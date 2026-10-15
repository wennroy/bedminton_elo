import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("next/cache", () => ({
  revalidatePath: vi.fn(),
}));

import { randomUUID } from "node:crypto";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addPlayer } from "@/lib/repo";
import * as ratingService from "@/lib/rating-service";
import { initializeRatingConfig } from "@/lib/rating-config";
import {
  createProjectService,
  markPlayedService,
  regenerateProjectService,
} from "@/lib/schedule-service";
import { getProject } from "@/lib/schedule-projects";
import { POST } from "./route";

function params(projectId: number, fixtureId: number) {
  return {
    params: Promise.resolve({ id: String(projectId), fixtureId: String(fixtureId) }),
  };
}

function postRequest(projectId: number, fixtureId: number, body: object): Request {
  return new Request(
    `http://localhost/api/schedule/projects/${projectId}/matches/${fixtureId}/score`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
}

describe.sequential("schedule fixture score API（项目记分）", () => {
  let dbPath: string;
  let players: string[] = [];
  let projectId = 0;
  let fixtures: Array<{ id: number; revision: number }> = [];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-schedule-score-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    players = [
      String(addPlayer("A")),
      String(addPlayer("B")),
      String(addPlayer("C")),
      String(addPlayer("D")),
    ];
    const created = createProjectService({
      name: "记分项目",
      playerIds: players,
      matches: 2,
      seed: 42,
      model: "legacy",
      actorId: players[0],
      requestId: randomUUID(),
      db: getDb(),
    });
    projectId = created.project.id;
    fixtures = created.project.fixtures.map((f) => ({ id: f.id, revision: f.revision }));
  });

  afterEach(() => {
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
  });

  it("POST 201：pending → recorded，返回比分与评分字段，详情一致", async () => {
    const response = await POST(
      postRequest(projectId, fixtures[0].id, {
        scoreA: 21,
        scoreB: 12,
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[0],
        playedAt: "2026-08-12",
      }),
      params(projectId, fixtures[0].id)
    );
    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.fixture.status).toBe("recorded");
    expect(json.fixture.scoreA).toBe(21);
    expect(json.fixture.scoreB).toBe(12);
    expect(json.fixture.playedAt).toBe("2026-08-12");
    expect(json.rating).toEqual({
      state: "not_effective",
      model: "glicko2",
      reason: "rating config not initialized",
    });
    expect(json.before).toHaveLength(4);
    expect(json.after).toHaveLength(4);

    // 详情展示一致：recorded 带比分，members actualPlayed +1。
    const detail = getProject(projectId, getDb());
    const recorded = detail.fixtures.find((f) => f.id === fixtures[0].id)!;
    expect(recorded.status).toBe("recorded");
    expect(recorded.matchId).toBe(json.id);
    expect(recorded.scoreA).toBe(21);
    expect(detail.members.find((m) => m.playerId === players[0])?.actualPlayed).toBe(1);
  });

  it("POST 201：played_unscored → recorded 保留原日期（playedAt 传了也忽略）", async () => {
    const played = markPlayedService({
      projectId,
      fixtureId: fixtures[0].id,
      expectedFixtureRevision: fixtures[0].revision,
      actorId: players[0],
      playedAt: "2026-08-05",
      db: getDb(),
    });

    const response = await POST(
      postRequest(projectId, fixtures[0].id, {
        scoreA: 21,
        scoreB: 9,
        expectedFixtureRevision: played.fixture.revision,
        actorId: players[0],
        playedAt: "2026-01-01",
      }),
      params(projectId, fixtures[0].id)
    );
    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.fixture.status).toBe("recorded");
    expect(json.fixture.playedAt).toBe("2026-08-05");
  });

  it("POST 已 recorded → 409 already_recorded 携带当前比分；不重复插比赛", async () => {
    const first = await POST(
      postRequest(projectId, fixtures[0].id, {
        scoreA: 21,
        scoreB: 9,
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    const firstJson = await first.json();

    const second = await POST(
      postRequest(projectId, fixtures[0].id, {
        scoreA: 18,
        scoreB: 21,
        expectedFixtureRevision: firstJson.fixture.revision,
        actorId: players[1],
      }),
      params(projectId, fixtures[0].id)
    );
    expect(second.status).toBe(409);
    const secondJson = await second.json();
    expect(secondJson.code).toBe("already_recorded");
    expect(secondJson.fixture.scoreA).toBe(21);
    expect(secondJson.fixture.scoreB).toBe(9);
    expect((getDb().prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
  });

  it("POST superseded → 409 fixture_superseded；未知场次 → 404", async () => {
    regenerateProjectService({
      projectId,
      playerIds: players,
      matches: 0,
      expectedRevision: getProject(projectId, getDb()).revision,
      actorId: players[0],
      requestId: randomUUID(),
      db: getDb(),
    });
    const superseded = getProject(projectId, getDb()).superseded[0];
    const response = await POST(
      postRequest(projectId, superseded.id, {
        scoreA: 21,
        scoreB: 9,
        expectedFixtureRevision: superseded.revision,
        actorId: players[0],
      }),
      params(projectId, superseded.id)
    );
    expect(response.status).toBe(409);
    expect((await response.json()).code).toBe("fixture_superseded");

    const missing = await POST(
      postRequest(projectId, 99999, {
        scoreA: 21,
        scoreB: 9,
        expectedFixtureRevision: 0,
        actorId: players[0],
      }),
      params(projectId, 99999)
    );
    expect(missing.status).toBe(404);
  });

  it("POST 非法比分 400 且不落库（相等/负数/小数）", async () => {
    for (const [scoreA, scoreB] of [
      [21, 21],
      [-1, 9],
      [21.5, 9],
    ]) {
      const response = await POST(
        postRequest(projectId, fixtures[0].id, {
          scoreA,
          scoreB,
          expectedFixtureRevision: fixtures[0].revision,
          actorId: players[0],
        }),
        params(projectId, fixtures[0].id)
      );
      expect(response.status).toBe(400);
    }
    expect((getDb().prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(0);
  });

  it("POST 评分服务故障：201 + rating pending，比赛已保存、场次 recorded", async () => {
    initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, getDb());
    const spy = vi
      .spyOn(ratingService, "loadGlickoSnapshot")
      .mockImplementation(() => {
        throw new Error("glicko boom");
      });
    try {
      const response = await POST(
        postRequest(projectId, fixtures[0].id, {
          scoreA: 21,
          scoreB: 9,
          expectedFixtureRevision: fixtures[0].revision,
          actorId: players[0],
        }),
        params(projectId, fixtures[0].id)
      );
      expect(response.status).toBe(201);
      const json = await response.json();
      expect(json.rating.state).toBe("pending");
      expect(json.rating.reason).toContain("glicko boom");
      expect(json.fixture.status).toBe("recorded");
      expect((getDb().prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
    } finally {
      spy.mockRestore();
    }
  });

  it("POST 并发：旧 revision 重放 → 409 读到已有比分，另一场不受影响", async () => {
    const first = await POST(
      postRequest(projectId, fixtures[0].id, {
        scoreA: 21,
        scoreB: 9,
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    expect(first.status).toBe(201);

    const stale = await POST(
      postRequest(projectId, fixtures[0].id, {
        scoreA: 18,
        scoreB: 21,
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[1],
      }),
      params(projectId, fixtures[0].id)
    );
    expect(stale.status).toBe(409);
    expect((await stale.json()).code).toBe("already_recorded");
    expect((getDb().prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);

    // 另一场仍 pending 且 revision 不变。
    const detail = getProject(projectId, getDb());
    const other = detail.fixtures.find((f) => f.id === fixtures[1].id)!;
    expect(other.status).toBe("pending");
    expect(other.revision).toBe(fixtures[1].revision);
  });
});
