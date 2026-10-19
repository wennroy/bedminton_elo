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
import { createProjectService, recordFixtureScoreService } from "@/lib/schedule-service";
import { getProject } from "@/lib/schedule-projects";
import { POST } from "./route";

function params(id: number) {
  return { params: Promise.resolve({ id: String(id) }) };
}

function postRequest(id: number, body: object): Request {
  return new Request(`http://localhost/api/schedule/projects/${id}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe.sequential("schedule project generate API（续排）", () => {
  let dbPath: string;
  let players: string[] = [];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-schedule-generate-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    players = [
      String(addPlayer("A")),
      String(addPlayer("B")),
      String(addPlayer("C")),
      String(addPlayer("D")),
      String(addPlayer("E")),
    ];
  });

  afterEach(() => {
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
  });

  function createProject() {
    return createProjectService({
      name: "续排项目",
      playerIds: players.slice(0, 4),
      matches: 2,
      seed: 42,
      model: "legacy",
      actorId: players[0],
      requestId: randomUUID(),
      db: getDb(),
    });
  }

  it("郭老师方法用同一续排接口开启下一轮，拒绝未完成轮次与名单改动", async () => {
    const roster = [...players, ...["F", "G", "H"].map((name) => String(addPlayer(name)))];
    const created = createProjectService({mode: "guo", playerIds: roster, actorId: roster[0], requestId: randomUUID(), model: "legacy", db: getDb()}).project;
    const payload = {actorId: roster[0], requestId: randomUUID(), expectedRevision: created.revision};
    const blocked = await POST(postRequest(created.id, payload), params(created.id));
    expect(blocked.status).toBe(409);
    expect((await blocked.json()).code).toBe("state_conflict");
    const bypass = await POST(postRequest(created.id, {...payload, matches:0, playerIds:roster.slice(1)}), params(created.id));
    expect(bypass.status).toBe(400);
    for (const f of created.fixtures) {
      recordFixtureScoreService({projectId:created.id,fixtureId:f.id,scoreA:21,scoreB:18,expectedFixtureRevision:0,actorId:roster[0],db:getDb()});
    }
    const finished = getProject(created.id,getDb());
    const nextPayload = {...payload, expectedRevision:finished.revision};
    const next = await POST(postRequest(created.id,nextPayload),params(created.id));
    expect(next.status).toBe(201);
    const json = await next.json();
    expect(json.project.guo.rounds).toHaveLength(2);
    expect(json.fixtures).toHaveLength(6);
    expect(json.supersededCount).toBe(0);
    const retry = await POST(postRequest(created.id,nextPayload),params(created.id));
    expect((await retry.json()).replayed).toBe(true);
  });

  it("POST 201 续排：旧 pending superseded，新 fixtures 从 next_fixture_no 续号", async () => {
    const created = createProject();
    const response = await POST(
      postRequest(created.project.id, {
        playerIds: players.slice(0, 4),
        matches: 2,
        seed: 7,
        expectedRevision: created.project.revision,
        actorId: players[0],
        requestId: randomUUID(),
      }),
      params(created.project.id)
    );
    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.replayed).toBe(false);
    expect(json.fixtures).toHaveLength(2);
    expect(json.supersededCount).toBe(2);
    expect(json.fixtures.map((f: { displayNo: number }) => f.displayNo)).toEqual([3, 4]);
    expect(json.project.revision).toBe(created.project.revision + 1);
  });

  it("POST 201 matches=0：只保存名单（可换名单）并清空未打", async () => {
    const created = createProject();
    const response = await POST(
      postRequest(created.project.id, {
        playerIds: [players[0], players[2], players[3], players[4]],
        matches: 0,
        expectedRevision: created.project.revision,
        actorId: players[0],
        requestId: randomUUID(),
      }),
      params(created.project.id)
    );
    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.fixtures).toEqual([]);
    expect(json.project.fixtures).toHaveLength(0);
    expect(json.project.superseded).toHaveLength(2);
    expect(json.project.members.map((m: { playerId: string }) => m.playerId).sort()).toEqual(
      [players[0], players[2], players[3], players[4]].sort()
    );
  });

  it("POST 幂等重放 + 同 requestId 不同 payload 409", async () => {
    const created = createProject();
    const payload = {
      playerIds: players.slice(0, 4),
      matches: 1,
      seed: 11,
      expectedRevision: created.project.revision,
      actorId: players[0],
      requestId: randomUUID(),
    };
    const first = await POST(postRequest(created.project.id, payload), params(created.project.id));
    expect(first.status).toBe(201);
    const firstJson = await first.json();

    const replay = await POST(postRequest(created.project.id, payload), params(created.project.id));
    const replayJson = await replay.json();
    expect(replayJson.replayed).toBe(true);
    expect(replayJson.fixtures.map((f: { id: number }) => f.id)).toEqual(
      firstJson.fixtures.map((f: { id: number }) => f.id)
    );

    const conflict = await POST(
      postRequest(created.project.id, { ...payload, seed: 12 }),
      params(created.project.id)
    );
    expect(conflict.status).toBe(409);
    expect((await conflict.json()).code).toBe("idempotency_conflict");
  });

  it("POST revision CAS 冲突 409 携带 projectRevision；未知项目 404", async () => {
    const created = createProject();
    const stale = await POST(
      postRequest(created.project.id, {
        playerIds: players.slice(0, 4),
        matches: 1,
        expectedRevision: created.project.revision + 1,
        actorId: players[0],
        requestId: randomUUID(),
      }),
      params(created.project.id)
    );
    expect(stale.status).toBe(409);
    const staleJson = await stale.json();
    expect(staleJson.code).toBe("revision_conflict");
    expect(staleJson.projectRevision).toBe(created.project.revision);

    const missing = await POST(
      postRequest(9999, {
        playerIds: players.slice(0, 4),
        matches: 1,
        expectedRevision: 0,
        actorId: players[0],
        requestId: randomUUID(),
      }),
      params(9999)
    );
    expect(missing.status).toBe(404);
  });

  it("POST 已打过场次的项目续排：历史计入，新批次避开（avoidPlayedRematches 默认开）", async () => {
    const created = createProject();
    const detail = getProject(created.project.id, getDb());
    const fixture = detail.fixtures[0];

    // 记一场已打（直接 repo 层记分走服务太重，这里用 markPlayed+score 服务组合的最简路径）。
    const { recordFixtureScoreService } = await import("@/lib/schedule-service");
    recordFixtureScoreService({
      projectId: created.project.id,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: fixture.revision,
      actorId: players[0],
      playedAt: "2026-08-01",
      db: getDb(),
    });

    const after = getProject(created.project.id, getDb());
    expect(after.members.find((m) => m.playerId === players[0])?.actualPlayed).toBe(1);

    const response = await POST(
      postRequest(created.project.id, {
        playerIds: players.slice(0, 4),
        matches: 2,
        seed: 5,
        expectedRevision: after.revision,
        actorId: players[0],
        requestId: randomUUID(),
      }),
      params(created.project.id)
    );
    expect(response.status).toBe(201);
    // played fixture 未被 superseded（仍有效）。
    const json = await response.json();
    expect(json.project.superseded).toHaveLength(1);
    const playedStill = json.project.fixtures.find((f: { id: number }) => f.id === fixture.id);
    expect(playedStill.status).toBe("recorded");
  });
});
