import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type Database from "better-sqlite3";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { startServer, type RunningServer } from "./helpers/server";
import { ApiClient, type FixtureShape, type ProjectDetailShape } from "./helpers/http";
import { initializeRatingConfig, RATING_CONFIG_META_KEY } from "@/lib/rating-config";

/**
 * 旅程 6：故障与重试。
 * - 写事务中途故障完整回滚：测试库注入 AFTER UPDATE 触发器（只改测试数据、
 *   不改产品代码），使 status→recorded 的 UPDATE 必然失败，断言整事务回滚。
 * - 评分服务失败：损坏测试库 meta 里的评分配置 → 比赛已保存、卡片已登记、
 *   积分 pending；新建/续排返回 409 rating_unavailable。
 * - 幂等重试：同 requestId 同 payload 重放不重复创建项目/批次；
 *   同 requestId 不同 payload → 409 idempotency_conflict。
 */

const PORT = 8916;

let db: TempDb;
let server: RunningServer;
let api: ApiClient;
let sqlite: Database.Database;

interface ScoreResponse {
  rating: { state: string; reason?: string };
  fixture: FixtureShape;
}

async function loadProject(id: number): Promise<ProjectDetailShape> {
  const res = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${id}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

function countMatches(): number {
  return (sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c;
}

beforeAll(async () => {
  db = makeTempDb("j6");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  sqlite = db.open();
});

afterAll(async () => {
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 6：故障回滚 / 评分降级 / 幂等重试", () => {
  it("写事务中途故障完整回滚（测试库触发器注入，不影响产品代码）", async () => {
    const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
      name: "E2E 旅程六回滚项目",
      playerIds: PLAYER_IDS.slice(0, 6).map(String),
      matches: 3,
      actorId: "1",
      requestId: crypto.randomUUID(),
    });
    expect(created.status).toBe(201);
    const projectId = created.body.project.id;
    const fixture = created.body.project.fixtures[0];

    // 注入故障：任何 status→'recorded' 的 fixture 更新都中止。
    sqlite
      .prepare(
        `CREATE TRIGGER e2e_inject_record_fault
         AFTER UPDATE OF status ON schedule_fixtures
         WHEN NEW.status = 'recorded'
         BEGIN
           SELECT RAISE(ABORT, 'e2e injected fault');
         END`
      )
      .run();

    const res = await api.post(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      {
        scoreA: 21,
        scoreB: 10,
        expectedFixtureRevision: fixture.revision,
        actorId: "1",
      }
    );
    expect(res.status).toBeGreaterThanOrEqual(400);
    expect(res.status).toBeLessThan(500);

    // 完整回滚：无比赛落库、场次仍 pending、revision 未动。
    expect(countMatches()).toBe(0);
    const project = await loadProject(projectId);
    const fx = project.fixtures.find((f) => f.id === fixture.id)!;
    expect(fx.status).toBe("pending");
    expect(fx.revision).toBe(fixture.revision);
    expect(fx.matchId).toBeNull();

    // 移除故障后原请求可成功重放。
    sqlite.prepare(`DROP TRIGGER e2e_inject_record_fault`).run();
    const retry = await api.post<ScoreResponse>(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      {
        scoreA: 21,
        scoreB: 10,
        expectedFixtureRevision: fixture.revision,
        actorId: "1",
      }
    );
    expect(retry.status).toBe(201);
    expect(countMatches()).toBe(1);
  });

  it("评分服务失败：比赛已保存、卡片已登记、积分 pending；创建/续排 409 rating_unavailable", async () => {
    // 初始化评分配置（glicko2 可用）。
    initializeRatingConfig({}, sqlite);
    const matchesBefore = countMatches();
    const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
      name: "E2E 旅程六评分项目",
      playerIds: PLAYER_IDS.slice(0, 6).map(String),
      matches: 3,
      actorId: "1",
      requestId: crypto.randomUUID(),
      model: "glicko2",
    });
    expect(created.status).toBe(201);
    const projectId = created.body.project.id;
    const fixture = created.body.project.fixtures[0];

    // 损坏评分配置（只改测试数据）。
    const original = (
      sqlite
        .prepare(`SELECT value FROM meta WHERE key = ?`)
        .get(RATING_CONFIG_META_KEY) as { value: string }
    ).value;
    sqlite.prepare(`UPDATE meta SET value = ? WHERE key = ?`).run(
      "{corrupted-json",
      RATING_CONFIG_META_KEY
    );

    try {
      // 记分：201、fixture recorded、rating pending、比赛已落库。
      const scored = await api.post<ScoreResponse>(
        `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
        {
          scoreA: 21,
          scoreB: 13,
          expectedFixtureRevision: fixture.revision,
          actorId: "1",
        }
      );
      expect(scored.status).toBe(201);
      expect(scored.body.fixture.status).toBe("recorded");
      expect(scored.body.rating.state).toBe("pending");
      const project = await loadProject(projectId);
      expect(project.fixtures.find((f) => f.id === fixture.id)!.status).toBe("recorded");
      // 比赛确实落库（在回滚用例已写入 1 场的基础上 +1）。
      expect(countMatches()).toBe(matchesBefore + 1);

      // 新建：409 rating_unavailable，不伪造胜率。
      const downCreate = await api.post("/api/schedule/projects", {
        name: "E2E 不应建成",
        playerIds: PLAYER_IDS.slice(0, 6).map(String),
        matches: 2,
        actorId: "1",
        requestId: crypto.randomUUID(),
        model: "glicko2",
      });
      expect(downCreate.status).toBe(409);
      expect((downCreate.body as { code: string }).code).toBe("rating_unavailable");

      // 续排：409 rating_unavailable，名单未被改动。
      const downRegen = await api.post(`/api/schedule/projects/${projectId}/generate`, {
        playerIds: PLAYER_IDS.slice(0, 6).map(String),
        matches: 2,
        expectedRevision: project.revision,
        actorId: "1",
        requestId: crypto.randomUUID(),
      });
      expect(downRegen.status).toBe(409);
      expect((downRegen.body as { code: string }).code).toBe("rating_unavailable");
      const unchanged = await loadProject(projectId);
      expect(unchanged.members).toHaveLength(6);
      expect(unchanged.fixtures.filter((f) => f.status === "pending")).toHaveLength(2);
    } finally {
      // 恢复评分配置，避免影响后续断言与清理。
      sqlite
        .prepare(`UPDATE meta SET value = ? WHERE key = ?`)
        .run(original, RATING_CONFIG_META_KEY);
    }
  });

  it("同 requestId 重试：不重复创建项目/批次；不同 payload 409 idempotency_conflict", async () => {
    const requestId = crypto.randomUUID();
    const payload = {
      name: "E2E 幂等项目",
      playerIds: PLAYER_IDS.slice(0, 6).map(String),
      matches: 3,
      avoidPlayedRematches: true,
      seed: 424242,
      actorId: "1",
      requestId,
    };
    const first = await api.post<{ project: ProjectDetailShape; replayed: boolean }>(
      "/api/schedule/projects",
      payload
    );
    expect(first.status).toBe(201);
    expect(first.body.replayed).toBe(false);
    const projectId = first.body.project.id;

    // 网络响应丢失后的安全重试：同 requestId + 同 payload → 同一项目。
    const replay = await api.post<{ project: ProjectDetailShape; replayed: boolean }>(
      "/api/schedule/projects",
      payload
    );
    expect(replay.status).toBe(201);
    expect(replay.body.replayed).toBe(true);
    expect(replay.body.project.id).toBe(projectId);
    expect(replay.body.project.fixtures).toHaveLength(3);

    const list = await api.get<{ projects: Array<{ id: number; name: string }> }>(
      "/api/schedule/projects"
    );
    // 只按本项目名过滤（DB 内还有同文件其它用例的项目）。
    expect(
      list.body.projects.filter((p) => p.name === "E2E 幂等项目")
    ).toHaveLength(1);

    // 同 requestId 不同 payload → 409 idempotency_conflict。
    const conflict = await api.post("/api/schedule/projects", {
      ...payload,
      name: "E2E 幂等冲突",
    });
    expect(conflict.status).toBe(409);
    expect((conflict.body as { code: string }).code).toBe("idempotency_conflict");

    // 续排批次幂等：同 requestId + 同 payload → 同 generation、不重复建场。
    const genPayload = {
      playerIds: PLAYER_IDS.slice(0, 6).map(String),
      matches: 2,
      avoidPlayedRematches: true,
      seed: 777,
      expectedRevision: first.body.project.revision,
      actorId: "1",
      requestId: crypto.randomUUID(),
    };
    const gen1 = await api.post<{
      generationId: number;
      fixtures: FixtureShape[];
      replayed: boolean;
    }>(`/api/schedule/projects/${projectId}/generate`, genPayload);
    expect(gen1.status).toBe(201);
    expect(gen1.body.replayed).toBe(false);
    expect(gen1.body.fixtures).toHaveLength(2);

    const gen2 = await api.post<{
      generationId: number;
      fixtures: FixtureShape[];
      replayed: boolean;
    }>(`/api/schedule/projects/${projectId}/generate`, genPayload);
    expect(gen2.status).toBe(201);
    expect(gen2.body.replayed).toBe(true);
    expect(gen2.body.generationId).toBe(gen1.body.generationId);
    expect(gen2.body.fixtures.map((f) => f.id)).toEqual(gen1.body.fixtures.map((f) => f.id));
    // 原始 3 场已被 gen1 替换为 2 场新待打；重试没有第三次建场。
    const project = await loadProject(projectId);
    expect(project.fixtures.filter((f) => f.status === "pending")).toHaveLength(2);
  });
});
