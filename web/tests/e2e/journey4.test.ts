import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { startServer, type RunningServer } from "./helpers/server";
import {
  ApiClient,
  type FixtureShape,
  type ProjectDetailShape,
} from "./helpers/http";
import { canonicalMatchKey } from "@/lib/scheduling/types";

/**
 * 旅程 4：「减少已打对阵重复」勾选开/关都已打保留；同分评分下构造
 * 有替代组合的场景，验证勾选后避开已打原阵容；4 人反复续排仍出解。
 */

const PORT = 8914;

let db: TempDb;
let server: RunningServer;
let api: ApiClient;
let projectId: number;
let playedFixture: FixtureShape;

async function loadProject(id: number): Promise<ProjectDetailShape> {
  const res = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${id}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

async function regen(
  id: number,
  playerIds: string[],
  matches: number,
  expectedRevision: number,
  extra: Record<string, unknown> = {}
) {
  return api.post<{ project: ProjectDetailShape; fixtures: FixtureShape[]; replayed: boolean }>(
    `/api/schedule/projects/${id}/generate`,
    {
      playerIds,
      matches,
      expectedRevision,
      actorId: "1",
      requestId: crypto.randomUUID(),
      ...extra,
    }
  );
}

beforeAll(async () => {
  db = makeTempDb("j4");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);

  // 6 人全新项目：无任何历史比赛，所有人同分（legacy 初始分），
  // 天然构成「同分评分 + 存在替代组合」的场景。
  const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
    name: "E2E 旅程四项目",
    playerIds: PLAYER_IDS.slice(0, 6).map(String),
    matches: 6,
    actorId: "1",
    requestId: crypto.randomUUID(),
  });
  expect(created.status).toBe(201);
  projectId = created.body.project.id;
});

afterAll(async () => {
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 4：减少已打对阵重复", () => {
  it("勾选开（默认）：已打保留，且多 seed 续排均避开已打原阵容", async () => {
    const project = await loadProject(projectId);
    const f1 = project.fixtures[0];

    const scored = await api.post<{ fixture: FixtureShape }>(
      `/api/schedule/projects/${projectId}/matches/${f1.id}/score`,
      {
        scoreA: 21,
        scoreB: 9,
        expectedFixtureRevision: f1.revision,
        actorId: "1",
        playedAt: "2026-10-05",
      }
    );
    expect(scored.status).toBe(201);
    playedFixture = scored.body.fixture;
    const playedKey = canonicalMatchKey(
      playedFixture.pa1,
      playedFixture.pa2,
      playedFixture.pb1,
      playedFixture.pb2
    );

    // 5 个不同 seed 连续续排（每场 1 场）：每场都不重复已打对阵，
    // 且已打场次原样保留。
    const roster = PLAYER_IDS.slice(0, 6).map(String);
    for (let seed = 101; seed <= 105; seed += 1) {
      const current = await loadProject(projectId);
      const res = await regen(projectId, roster, 1, current.revision, { seed });
      expect(res.status).toBe(201);
      expect(res.body.fixtures).toHaveLength(1);

      const kept = res.body.project.fixtures.find((f) => f.id === playedFixture.id)!;
      expect(kept.status).toBe("recorded");
      expect(kept.playedAt).toBe("2026-10-05");
      expect(canonicalMatchKey(kept.pa1, kept.pa2, kept.pb1, kept.pb2)).toBe(playedKey);

      const next = res.body.fixtures[0];
      const nextKey = canonicalMatchKey(next.pa1, next.pa2, next.pb1, next.pb2);
      expect(nextKey).not.toBe(playedKey);
    }
  });

  it("勾选关：已打同样保留（不删除、不改写）", async () => {
    const current = await loadProject(projectId);
    const playedKey = canonicalMatchKey(
      playedFixture.pa1,
      playedFixture.pa2,
      playedFixture.pb1,
      playedFixture.pb2
    );
    const res = await regen(
      projectId,
      PLAYER_IDS.slice(0, 6).map(String),
      2,
      current.revision,
      { seed: 777, avoidPlayedRematches: false }
    );
    expect(res.status).toBe(201);

    const kept = res.body.project.fixtures.find((f) => f.id === playedFixture.id)!;
    expect(kept.status).toBe("recorded");
    expect(canonicalMatchKey(kept.pa1, kept.pa2, kept.pb1, kept.pb2)).toBe(playedKey);
    expect(res.body.fixtures).toHaveLength(2);
  });

  it("4 人反复续排仍出解：每轮 2 场、四人都出场且本轮对阵不重复", async () => {
    const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
      name: "E2E 旅程四四人项目",
      playerIds: PLAYER_IDS.slice(0, 4).map(String),
      matches: 2,
      actorId: "1",
      requestId: crypto.randomUUID(),
    });
    expect(created.status).toBe(201);
    const fourId = created.body.project.id;
    const roster = PLAYER_IDS.slice(0, 4).map(String);

    for (let round = 1; round <= 3; round += 1) {
      const current = await loadProject(fourId);
      const res = await regen(fourId, roster, 2, current.revision, { seed: 900 + round });
      expect(res.status).toBe(201);
      const fixtures = res.body.fixtures;
      expect(fixtures).toHaveLength(2);

      // 本轮两条对阵互不相同（4 人 2 场必然重新组合）。
      const keys = fixtures.map((f) => canonicalMatchKey(f.pa1, f.pa2, f.pb1, f.pb2));
      expect(new Set(keys).size).toBe(2);

      // 四人都恰好出场 2 次。
      const counts = new Map<string, number>();
      for (const f of fixtures) {
        for (const pid of [f.pa1, f.pa2, f.pb1, f.pb2]) {
          counts.set(pid, (counts.get(pid) ?? 0) + 1);
        }
      }
      for (const pid of roster) {
        expect(counts.get(pid)).toBe(2);
      }

      // 引擎确实出了「解」：待打为 2、名单仍是 4 人。
      const after = res.body.project;
      expect(after.members).toHaveLength(4);
      expect(after.fixtures.filter((f) => f.status === "pending")).toHaveLength(2);
    }
  });
});
