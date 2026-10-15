import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type Database from "better-sqlite3";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { ADMIN_PASSWORD, startServer, type RunningServer } from "./helpers/server";
import {
  ApiClient,
  type FixtureShape,
  type ProjectDetailShape,
} from "./helpers/http";

/**
 * 旅程 7：played_unscored 补分保持原日期、出场不重复计数；两条撤回入口
 * （/api/matches?id= 与 /api/matches/[id]）均解除比分保留已打并可重新登记；
 * admin 改分（x-admin-key）同步项目刷新；冲突人员合并（同项目双成员记录）
 * 拒绝且无部分改写。
 */

const PORT = 8917;

let db: TempDb;
let server: RunningServer;
let api: ApiClient;
let sqlite: Database.Database;
let projectId: number;
let fixture: FixtureShape;

async function loadProject(): Promise<ProjectDetailShape> {
  const res = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${projectId}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

function fixtureNow(): FixtureShape {
  const project = sqlite
    .prepare(`SELECT revision, status, match_id AS matchId, played_at AS playedAt FROM schedule_fixtures WHERE id = ?`)
    .get(fixture.id) as {
    revision: number;
    status: FixtureShape["status"];
    matchId: number | null;
    playedAt: string | null;
  };
  return { ...fixture, ...project };
}

beforeAll(async () => {
  db = makeTempDb("j7");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  sqlite = db.open();

  const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
    name: "E2E 旅程七项目",
    playerIds: PLAYER_IDS.slice(0, 8).map(String),
    matches: 8,
    actorId: "1",
    requestId: crypto.randomUUID(),
  });
  expect(created.status).toBe(201);
  projectId = created.body.project.id;
  fixture = created.body.project.fixtures[0];
});

afterAll(async () => {
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 7：补分日期 / 双撤回入口 / admin 改分 / 合并拒绝", () => {
  it("played_unscored 补分保持比赛原日期、出场不重复计数", async () => {
    const played = await api.post<{ fixture: FixtureShape }>(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/played`,
      { expectedFixtureRevision: fixture.revision, actorId: "1", playedAt: "2026-10-02" }
    );
    expect(played.status).toBe(200);
    expect(played.body.fixture.status).toBe("played_unscored");

    const scored = await api.post<{ id: number; fixture: FixtureShape }>(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      {
        // 不传 playedAt：必须沿用「标记已打」的 2026-10-02，而不是今天。
        scoreA: 21,
        scoreB: 16,
        expectedFixtureRevision: played.body.fixture.revision,
        actorId: "1",
      }
    );
    expect(scored.status).toBe(201);
    expect(scored.body.fixture.playedAt).toBe("2026-10-02");

    const match = sqlite
      .prepare(`SELECT played_at AS playedAt FROM matches WHERE id = ?`)
      .get(scored.body.id) as { playedAt: string };
    expect(match.playedAt).toBe("2026-10-02");

    // 出场不重复计数：四名出场者各计 1 个已打场次（不是标记+登记两次）。
    const project = await loadProject();
    const lineup = [fixture.pa1, fixture.pa2, fixture.pb1, fixture.pb2];
    for (const pid of lineup) {
      const member = project.members.find((m) => m.playerId === pid)!;
      expect(member.actualPlayed).toBe(1);
    }
    const nonLineup = project.members.find((m) => !lineup.includes(m.playerId))!;
    expect(nonLineup.actualPlayed).toBe(0);
  });

  it("撤回入口一：DELETE /api/matches?id= 解除比分、保留已打、可重新登记", async () => {
    const current = fixtureNow();
    expect(current.status).toBe("recorded");
    const matchId = current.matchId!;

    const del = await api.delete(`/api/matches?id=${matchId}`);
    expect(del.status).toBe(200);

    const after = fixtureNow();
    expect(after.status).toBe("played_unscored");
    expect(after.matchId).toBeNull();
    expect(after.playedAt).toBe("2026-10-02");
    const matchRow = sqlite
      .prepare(`SELECT 1 FROM matches WHERE id = ?`)
      .get(matchId);
    expect(matchRow).toBeUndefined();

    // 重新登记成功。
    const rescored = await api.post(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      {
        scoreA: 21,
        scoreB: 19,
        expectedFixtureRevision: after.revision,
        actorId: "2",
      }
    );
    expect(rescored.status).toBe(201);
    expect((rescored.body as { fixture: FixtureShape }).fixture.playedAt).toBe("2026-10-02");
  });

  it("撤回入口二：DELETE /api/matches/[id] 同样解除比分并保留已打", async () => {
    const current = fixtureNow();
    expect(current.status).toBe("recorded");
    const matchId = current.matchId!;

    const del = await api.delete(`/api/matches/${matchId}`);
    expect(del.status).toBe(200);

    const after = fixtureNow();
    expect(after.status).toBe("played_unscored");
    expect(after.matchId).toBeNull();
    expect(after.playedAt).toBe("2026-10-02");

    const rescored = await api.post<{ id: number }>(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      {
        scoreA: 21,
        scoreB: 14,
        expectedFixtureRevision: after.revision,
        actorId: "1",
      }
    );
    expect(rescored.status).toBe(201);
    (fixture as { matchId: number | null }).matchId = rescored.body.id;
  });

  it("admin 改分（x-admin-key）同步项目刷新", async () => {
    const before = await loadProject();
    const beforeRevision = before.revision;
    const matchId = fixtureNow().matchId!;
    const fixtureRevisionBefore = fixtureNow().revision;

    // 无 admin key → 401。
    const unauth = await api.patch(`/api/admin/matches/${matchId}`, {
      scoreA: 22,
      scoreB: 20,
    });
    expect(unauth.status).toBe(401);

    const patched = await api.patch(
      `/api/admin/matches/${matchId}`,
      { scoreA: 22, scoreB: 20 },
      { "x-admin-key": ADMIN_PASSWORD }
    );
    expect(patched.status).toBe(200);

    // 项目 revision 被同步 +1；fixture 仍 recorded，比分可读出新值。
    const after = await loadProject();
    expect(after.revision).toBe(beforeRevision + 1);
    const fx = after.fixtures.find((f) => f.id === fixture.id)!;
    expect(fx.status).toBe("recorded");
    expect(fx.scoreA).toBe(22);
    expect(fx.scoreB).toBe(20);
    expect(fx.revision).toBe(fixtureRevisionBefore);
    expect(fx.matchId).toBe(matchId);
  });

  it("冲突人员合并（同项目双成员记录）拒绝且无部分改写", async () => {
    // 构造「两名球员同属一个项目、但从不在同一场次出现」的确定性场景，
    // 让合并预检确定性地走到「同项目双成员记录」分支：
    // 每轮造两名全新球员 + 一个 10 人 1 场项目；若唯一场次把两人排在一起
    // 就换新球员重造（superseded 行也参与对撞预检，不能留下同场记录）。
    let mergeProjectId = 0;
    let fromId = 0;
    let toId = 0;
    let revisionBefore = 0;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      const px = await api.post<{ id: number }>("/api/players", {
        name: `E2E-M${attempt}A`,
      });
      const py = await api.post<{ id: number }>("/api/players", {
        name: `E2E-M${attempt}B`,
      });
      expect(px.status).toBe(201);
      expect(py.status).toBe(201);
      const roster = [px.body.id, py.body.id, 1, 2, 3, 4, 5, 6, 7, 8].map(String);
      const created = await api.post<{ project: ProjectDetailShape }>(
        "/api/schedule/projects",
        {
          name: `E2E 旅程七合并${attempt}`,
          playerIds: roster,
          matches: 1,
          actorId: "1",
          requestId: crypto.randomUUID(),
        }
      );
      expect(created.status).toBe(201);
      const fx = created.body.project.fixtures[0];
      const slots = [fx.pa1, fx.pa2, fx.pb1, fx.pb2];
      if (slots.includes(String(px.body.id)) && slots.includes(String(py.body.id))) {
        continue; // 同场对撞会抢先触发，换一对全新球员重造。
      }
      mergeProjectId = created.body.project.id;
      fromId = px.body.id;
      toId = py.body.id;
      revisionBefore = created.body.project.revision;
      break;
    }
    expect(mergeProjectId).toBeGreaterThan(0);

    // 成员 A 的 fixture 出场行（用于事后比对无改写）。
    const membersBefore = JSON.stringify(
      sqlite
        .prepare(
          `SELECT player_id AS pid, active, fairness_base AS fb, played_at_join AS paj
           FROM schedule_project_players WHERE project_id = ? ORDER BY player_id`
        )
        .all(mergeProjectId)
    );
    const playersBefore = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM players`).get() as { c: number }
    ).c;

    // 无 admin key → 401，不做任何检查改写。
    const unauth = await api.post(`/api/admin/players/${fromId}/merge`, { toId });
    expect(unauth.status).toBe(401);

    const res = await api.post(
      `/api/admin/players/${fromId}/merge`,
      { toId },
      { "x-admin-key": ADMIN_PASSWORD }
    );
    expect(res.status).toBe(400);
    expect((res.body as { error: string }).error).toContain("均有成员记录");

    // 无部分改写：两名球员都还在；成员行未动；revision 未动。
    const playersAfter = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM players`).get() as { c: number }
    ).c;
    expect(playersAfter).toBe(playersBefore);
    const membersAfter = JSON.stringify(
      sqlite
        .prepare(
          `SELECT player_id AS pid, active, fairness_base AS fb, played_at_join AS paj
           FROM schedule_project_players WHERE project_id = ? ORDER BY player_id`
        )
        .all(mergeProjectId)
    );
    expect(membersAfter).toBe(membersBefore);
    const detail = await api.get<{ project: ProjectDetailShape }>(
      `/api/schedule/projects/${mergeProjectId}`
    );
    expect(detail.body.project.revision).toBe(revisionBefore);
  });
});
