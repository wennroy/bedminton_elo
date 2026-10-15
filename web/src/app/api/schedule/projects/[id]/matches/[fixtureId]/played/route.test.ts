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
import { createProjectService } from "@/lib/schedule-service";
import { getProject } from "@/lib/schedule-projects";
import { shanghaiLocalDateFromInstant } from "@/lib/ratings/calendar";
import { POST } from "./route";

function params(projectId: number, fixtureId: number) {
  return {
    params: Promise.resolve({ id: String(projectId), fixtureId: String(fixtureId) }),
  };
}

function postRequest(projectId: number, fixtureId: number, body: object): Request {
  return new Request(
    `http://localhost/api/schedule/projects/${projectId}/matches/${fixtureId}/played`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }
  );
}

function shanghaiToday(): string {
  return shanghaiLocalDateFromInstant(new Date().toISOString());
}

describe.sequential("schedule fixture played API（登记已打）", () => {
  let dbPath: string;
  let players: string[] = [];
  let projectId = 0;
  let fixtures: Array<{ id: number; revision: number }> = [];

  beforeEach(async () => {
    closeDb();
    dbPath = join(tmpdir(), `test-schedule-played-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    players = [
      String(addPlayer("A")),
      String(addPlayer("B")),
      String(addPlayer("C")),
      String(addPlayer("D")),
    ];
    const created = createProjectService({
      name: "已打项目",
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

  it("POST 200：pending → played_unscored，缺省 playedAt 为上海当天", async () => {
    const response = await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    expect(response.status).toBe(200);
    const json = await response.json();
    expect(json.fixture.status).toBe("played_unscored");
    expect(json.fixture.playedAt).toBe(shanghaiToday());
    expect(json.fixture.matchId).toBeNull();
  });

  it("POST 200：显式 playedAt", async () => {
    const response = await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[1],
        playedAt: "2026-08-11",
      }),
      params(projectId, fixtures[0].id)
    );
    expect(response.status).toBe(200);
    expect((await response.json()).fixture.playedAt).toBe("2026-08-11");
  });

  it("已 played_unscored 再标记 → 409 state_conflict；不产生比赛行", async () => {
    const first = await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    const firstJson = await first.json();

    const again = await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: firstJson.fixture.revision,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    expect(again.status).toBe(409);
    expect((await again.json()).code).toBe("state_conflict");
    expect((getDb().prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(0);
  });

  it("wrong revision → 409 revision_conflict 携带当前 fixture", async () => {
    const response = await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: fixtures[0].revision + 10,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    expect(response.status).toBe(409);
    const json = await response.json();
    expect(json.code).toBe("revision_conflict");
    expect(json.fixture.id).toBe(fixtures[0].id);
    expect(json.fixture.revision).toBe(fixtures[0].revision);
  });

  it("superseded 场次 → 409 fixture_superseded；recorded 场次 → 409 already_recorded", async () => {
    // 先让 fixtures[0] 走完整链路变成 recorded。
    await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    const { recordFixtureScoreService } = await import("@/lib/schedule-service");
    const detail1 = getProject(projectId, getDb());
    const fx0 = detail1.fixtures[0];
    const scored = recordFixtureScoreService({
      projectId,
      fixtureId: fx0.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: fx0.revision,
      actorId: players[0],
      db: getDb(),
    });

    const recorded = await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: scored.fixture!.revision,
        actorId: players[0],
      }),
      params(projectId, fixtures[0].id)
    );
    expect(recorded.status).toBe(409);
    expect((await recorded.json()).code).toBe("already_recorded");

    // 续排（matches=0）把 fixtures[1] superseded。
    const { regenerateProjectService } = await import("@/lib/schedule-service");
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
        expectedFixtureRevision: superseded.revision,
        actorId: players[0],
      }),
      params(projectId, superseded.id)
    );
    expect(response.status).toBe(409);
    expect((await response.json()).code).toBe("fixture_superseded");
  });

  it("未知场次 / 属于另一项目的场次 → 404；非法日期 400", async () => {
    const missing = await POST(
      postRequest(projectId, 99999, {
        expectedFixtureRevision: 0,
        actorId: players[0],
      }),
      params(projectId, 99999)
    );
    expect(missing.status).toBe(404);
    expect((await missing.json()).code).toBe("not_found");

    const other = createProjectService({
      name: "另一项目",
      playerIds: players,
      matches: 1,
      seed: 1,
      model: "legacy",
      actorId: players[0],
      requestId: randomUUID(),
      db: getDb(),
    });
    const foreign = other.project.fixtures[0];
    const cross = await POST(
      postRequest(projectId, foreign.id, {
        expectedFixtureRevision: foreign.revision,
        actorId: players[0],
      }),
      params(projectId, foreign.id)
    );
    expect(cross.status).toBe(404);

    const badDate = await POST(
      postRequest(projectId, fixtures[0].id, {
        expectedFixtureRevision: fixtures[0].revision,
        actorId: players[0],
        playedAt: "2026-02-30",
      }),
      params(projectId, fixtures[0].id)
    );
    expect(badDate.status).toBe(400);
  });
});
