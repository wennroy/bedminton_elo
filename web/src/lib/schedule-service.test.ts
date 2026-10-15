import { describe, it, expect, vi } from "vitest";

vi.mock("next/cache", () => ({
  revalidatePath: vi.fn(),
}));

import { randomUUID } from "node:crypto";
import { tmpdir } from "os";
import { join } from "path";
import Database from "better-sqlite3";
import { createDb } from "./db";
import { addPlayer } from "./repo";
import { predictElo } from "./elo";
import * as ratingService from "./rating-service";
import {
  createProjectService,
  regenerateProjectService,
  renameProjectService,
  markPlayedService,
  recordFixtureScoreService,
  mergePlayerWithScheduleRefs,
  scheduleReferenceBlockReason,
  noteMatchScoreEdited,
  toScheduleHttp,
  ScheduleServiceError,
  shanghaiToday,
} from "./schedule-service";
import { getProject } from "./schedule-projects";
import { shanghaiLocalDateFromInstant } from "./ratings/calendar";
import { initializeRatingConfig } from "./rating-config";

function mkDb(): Database.Database {
  const path = join(
    tmpdir(),
    `schedule-service-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.db`
  );
  return createDb(path);
}

function seedPlayers(db: Database.Database, n: number): string[] {
  const ids: string[] = [];
  for (let i = 0; i < n; i++) {
    ids.push(String(addPlayer(`SS-P${i}-${Math.random().toString(36).slice(2, 6)}`, db)));
  }
  // 服务层会数值排序；这里也排序使断言稳定。
  return ids.sort((a, b) => Number(a) - Number(b));
}

function createLegacyProject(
  db: Database.Database,
  playerIds: string[],
  overrides: Record<string, unknown> = {}
) {
  return createProjectService({
    name: "服务测试",
    playerIds,
    matches: 2,
    seed: 42,
    model: "legacy",
    actorId: playerIds[0],
    requestId: randomUUID(),
    db,
    ...overrides,
  });
}

function expectServiceError(
  fn: () => unknown,
  status: number,
  code: string
): { status: number; body: Record<string, unknown> } {
  try {
    fn();
  } catch (err) {
    // 服务层 ScheduleServiceError 与 repo 层 ScheduleProjectError 都经映射器出 HTTP 语义。
    const http = toScheduleHttp(err);
    expect(http.status).toBe(status);
    expect(http.body.code).toBe(code);
    return http;
  }
  throw new Error(`期望 ${status}/${code}，但没有抛错`);
}

describe("createProjectService", () => {
  it("happy path（legacy）：返回项目详情、首批场次与姓名映射", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 5);
    const result = createLegacyProject(db, ids.slice(0, 4));

    expect(result.replayed).toBe(false);
    expect(result.project.name).toBe("服务测试");
    expect(result.project.model).toBe("legacy");
    expect(result.project.fixtures).toHaveLength(2);
    expect(result.project.fixtures.map((f) => f.displayNo)).toEqual([1, 2]);
    expect(result.project.fixtures.every((f) => f.status === "pending")).toBe(true);
    expect(result.project.members).toHaveLength(4);
    expect(result.project.members.every((m) => m.actualPlayed === 0)).toBe(true);
    expect(Object.keys(result.names)).toHaveLength(4);
    // 缺省 avoidPlayedRematches=true 落进 input_json 快照。
    const input = result.project.latestGeneration?.input as {
      avoidPlayedRematches: boolean;
      matches: number;
      seed: number;
    };
    expect(input.avoidPlayedRematches).toBe(true);
    expect(input.matches).toBe(2);
    expect(input.seed).toBe(42);
  });

  it("缺省 name 使用上海当天「YYYY-MM-DD 配对」", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const result = createProjectService({
      playerIds: ids,
      matches: 1,
      model: "legacy",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    expect(result.project.name).toBe(
      `${shanghaiLocalDateFromInstant(new Date().toISOString())} 配对`
    );
  });

  it("同 payload + 同 seed 的两次创建（不同 requestId）场次逐比特一致", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const first = createLegacyProject(db, ids);
    const second = createLegacyProject(db, ids);
    const lineup = (r: typeof first) =>
      r.project.fixtures.map((f) => [f.pa1, f.pa2, f.pb1, f.pb2]);
    expect(lineup(second)).toEqual(lineup(first));
  });

  it("幂等重放：同 requestId 同 payload 返回已建项目；不同 payload 409", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const requestId = randomUUID();
    const payload = {
      name: "幂等项目",
      playerIds: ids,
      matches: 2,
      seed: 7,
      model: "legacy",
      actorId: ids[0],
      requestId,
      db,
    };
    const first = createProjectService(payload);
    expect(first.replayed).toBe(false);

    const replay = createProjectService(payload);
    expect(replay.replayed).toBe(true);
    expect(replay.project.id).toBe(first.project.id);
    expect(replay.project.fixtures.map((f) => f.id)).toEqual(
      first.project.fixtures.map((f) => f.id)
    );

    expectServiceError(
      () => createProjectService({ ...payload, matches: 3 }),
      409,
      "idempotency_conflict"
    );
  });

  it.each([
    [{ matches: 0 }, "matches"],
    [{ matches: 100 }, "matches"],
    [{ matches: 1.5 }, "matches"],
    [{ playerIds: ["a", "b", "c", "d"] }, "actorId"],
    [{ playerIds: [] }, "playerIds"],
  ])("非法入参 400：%o", (overrides, field) => {
    const db = mkDb();
    const ids = seedPlayers(db, 6);
    const base = {
      playerIds: ids.slice(0, 4),
      matches: 2,
      model: "legacy",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    };
    expectServiceError(
      () => createProjectService({ ...base, ...overrides }),
      400,
      "validation_error"
    );
  });

  it("重复/未知/人数不足的球员 ID 400", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 5);
    const base = {
      matches: 1,
      model: "legacy",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    };
    // 重复 ID。
    expectServiceError(
      () =>
        createProjectService({
          ...base,
          playerIds: [ids[0], ids[0], ids[1], ids[2]],
        }),
      400,
      "validation_error"
    );
    // 未知 ID。
    expectServiceError(
      () =>
        createProjectService({
          ...base,
          playerIds: [ids[0], ids[1], ids[2], "99999"],
        }),
      400,
      "player_not_found"
    );
    // 3 人不足 4 人。
    expectServiceError(
      () => createProjectService({ ...base, playerIds: ids.slice(0, 3) }),
      400,
      "validation_error"
    );
  });

  it("actorId 非法/不存在 400；requestId 非 UUID 400", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const base = { playerIds: ids, matches: 1, model: "legacy", db };
    expectServiceError(
      () =>
        createProjectService({
          ...base,
          actorId: "abc",
          requestId: randomUUID(),
        }),
      400,
      "validation_error"
    );
    expectServiceError(
      () =>
        createProjectService({
          ...base,
          actorId: "99999",
          requestId: randomUUID(),
        }),
      400,
      "player_not_found"
    );
    expectServiceError(
      () =>
        createProjectService({ ...base, actorId: ids[0], requestId: "nope" }),
      400,
      "validation_error"
    );
  });

  it("seed 越界 400；显式非法 model 400", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const base = {
      playerIds: ids,
      matches: 1,
      model: "legacy",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    };
    expectServiceError(
      () => createProjectService({ ...base, seed: -1 }),
      400,
      "validation_error"
    );
    expectServiceError(
      () => createProjectService({ ...base, seed: 0x100000000 }),
      400,
      "validation_error"
    );
    expectServiceError(
      () => createProjectService({ ...base, model: "trueskill" }),
      400,
      "validation_error"
    );
  });

  it("glicko2 快照不可用 → 409 rating_unavailable（不静默退回 legacy）", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    expectServiceError(
      () =>
        createProjectService({
          playerIds: ids,
          matches: 1,
          model: "glicko2",
          actorId: ids[0],
          requestId: randomUUID(),
          db,
        }),
      409,
      "rating_unavailable"
    );
  });

  it("glicko2 ready 时可创建（胜率回调注入不抛错）", () => {
    const db = mkDb();
    initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, db);
    const ids = seedPlayers(db, 4);
    const result = createProjectService({
      playerIds: ids,
      matches: 1,
      model: "glicko2",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    expect(result.project.model).toBe("glicko2");
    expect(result.project.fixtures).toHaveLength(1);
  });
});

describe("regenerateProjectService", () => {
  it("续排：旧 pending 全部 superseded，新 fixtures 生成，revision CAS 通过", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 5);
    const created = createLegacyProject(db, ids.slice(0, 4));
    const before = getProject(created.project.id, db);

    const result = regenerateProjectService({
      projectId: created.project.id,
      playerIds: ids.slice(0, 4),
      matches: 2,
      seed: 99,
      expectedRevision: before.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });

    expect(result.replayed).toBe(false);
    expect(result.fixtures).toHaveLength(2);
    expect(result.supersededCount).toBe(2);
    expect(result.project.revision).toBe(before.revision + 1);
    expect(result.project.superseded).toHaveLength(2);
    // 新旧 display_no 不复用：新批次从 3 开始。
    expect(result.project.fixtures.map((f) => f.displayNo)).toEqual([3, 4]);
  });

  it("matches=0 合法：只保存名单、清空未打、跳过求解", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 6);
    const created = createLegacyProject(db, ids.slice(0, 4));
    // 换一个名单（加入 2 人、去掉 1 人）。
    const result = regenerateProjectService({
      projectId: created.project.id,
      playerIds: [ids[0], ids[2], ids[3], ids[4], ids[5]],
      matches: 0,
      expectedRevision: created.project.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });

    expect(result.fixtures).toEqual([]);
    expect(result.project.fixtures).toHaveLength(0);
    expect(result.project.superseded).toHaveLength(2);
    expect(result.project.members.map((m) => m.playerId).sort()).toEqual(
      [ids[0], ids[2], ids[3], ids[4], ids[5]].sort()
    );
    // 新成员 actualPlayed=0；离开者不在 active 名单。
    expect(
      result.project.members.find((m) => m.playerId === ids[4])?.actualPlayed
    ).toBe(0);
  });

  it("matches=0 允许 <4 人保存名单；空名单拒绝；matches>0 仍要求 ≥4 人", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);

    // 现场减到 3 人：保存名单、清空未打（合同「保存名单，暂不安排比赛」）。
    const result = regenerateProjectService({
      projectId: created.project.id,
      playerIds: ids.slice(0, 3),
      matches: 0,
      expectedRevision: created.project.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    expect(result.fixtures).toEqual([]);
    expect(result.project.members.map((m) => m.playerId).sort()).toEqual(
      ids.slice(0, 3).sort()
    );

    expectServiceError(
      () =>
        regenerateProjectService({
          projectId: created.project.id,
          playerIds: [],
          matches: 0,
          expectedRevision: result.project.revision,
          actorId: ids[0],
          requestId: randomUUID(),
          db,
        }),
      400,
      "validation_error"
    );
    expectServiceError(
      () =>
        regenerateProjectService({
          projectId: created.project.id,
          playerIds: ids.slice(0, 3),
          matches: 1,
          expectedRevision: result.project.revision,
          actorId: ids[0],
          requestId: randomUUID(),
          db,
        }),
      400,
      "validation_error"
    );
  });

  it("打过的场次在续排时计入 effectiveAppearances 与历史对阵", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    const fixture = created.project.fixtures[0];
    recordFixtureScoreService({
      projectId: created.project.id,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 10,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      playedAt: "2026-08-01",
      db,
    });

    const after = getProject(created.project.id, db);
    expect(
      after.members.find((m) => m.playerId === ids[0])?.actualPlayed
    ).toBe(1);
    expect(after.members.find((m) => m.playerId === ids[0])?.pendingCount).toBe(
      1
    );

    // 幂等重放校验续排 input 快照带历史计数。
    const regen = regenerateProjectService({
      projectId: created.project.id,
      playerIds: ids,
      matches: 1,
      seed: 5,
      expectedRevision: after.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    const input = regen.project.latestGeneration?.input as {
      effectiveAppearances: Record<string, number>;
      playedMatchCounts: Record<string, number>;
    };
    expect(Object.values(input.effectiveAppearances).some((v) => v >= 1)).toBe(
      true
    );
    expect(Object.keys(input.playedMatchCounts)).toHaveLength(1);
  });

  it("revision CAS 冲突 409 携带当前 projectRevision", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    const http = expectServiceError(
      () =>
        regenerateProjectService({
          projectId: created.project.id,
          playerIds: ids,
          matches: 1,
          expectedRevision: created.project.revision + 5,
          actorId: ids[0],
          requestId: randomUUID(),
          db,
        }),
      409,
      "revision_conflict"
    );
    expect(http.body.projectRevision).toBe(created.project.revision);
  });

  it("幂等重放：同 requestId 同 payload 返回既有 fixtures；不同 payload 409", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    const payload = {
      projectId: created.project.id,
      playerIds: ids,
      matches: 1,
      seed: 11,
      expectedRevision: created.project.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    };
    const first = regenerateProjectService(payload);
    expect(first.replayed).toBe(false);

    const replay = regenerateProjectService(payload);
    expect(replay.replayed).toBe(true);
    expect(replay.fixtures.map((f) => f.id)).toEqual(
      first.fixtures.map((f) => f.id)
    );
    // 重放不产生新 superseded。
    expect(replay.supersededCount).toBe(0);

    expectServiceError(
      () => regenerateProjectService({ ...payload, seed: 12 }),
      409,
      "idempotency_conflict"
    );
  });

  it("未知项目 404；matches 越界/expectedRevision 缺失 400", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    expectServiceError(
      () =>
        regenerateProjectService({
          projectId: 9999,
          playerIds: ids,
          matches: 1,
          expectedRevision: 0,
          actorId: ids[0],
          requestId: randomUUID(),
          db,
        }),
      404,
      "not_found"
    );

    const created = createLegacyProject(db, ids);
    expectServiceError(
      () =>
        regenerateProjectService({
          projectId: created.project.id,
          playerIds: ids,
          matches: 100,
          expectedRevision: created.project.revision,
          actorId: ids[0],
          requestId: randomUUID(),
          db,
        }),
      400,
      "validation_error"
    );
  });
});

describe("renameProjectService", () => {
  it("改名成功：revision bump、updated_at 刷新，可同名", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids, { name: "原名" });

    const renamed = renameProjectService({
      projectId: created.project.id,
      name: "  新名字  ",
      expectedRevision: created.project.revision,
      actorId: ids[0],
      db,
    });
    expect(renamed.name).toBe("新名字");
    expect(renamed.revision).toBe(created.project.revision + 1);
    expect(getProject(created.project.id, db).name).toBe("新名字");
  });

  it("revision CAS 冲突 409 携带当前 projectRevision；未知项目 404", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);

    const http = expectServiceError(
      () =>
        renameProjectService({
          projectId: created.project.id,
          name: "x",
          expectedRevision: created.project.revision + 1,
          actorId: ids[0],
          db,
        }),
      409,
      "revision_conflict"
    );
    expect(http.body.projectRevision).toBe(created.project.revision);

    expectServiceError(
      () =>
        renameProjectService({
          projectId: 424242,
          name: "x",
          expectedRevision: 0,
          actorId: ids[0],
          db,
        }),
      404,
      "not_found"
    );
  });

  it("空名/超长名 400；不隐式改名单", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    expectServiceError(
      () =>
        renameProjectService({
          projectId: created.project.id,
          name: "   ",
          expectedRevision: created.project.revision,
          actorId: ids[0],
          db,
        }),
      400,
      "validation_error"
    );
    expectServiceError(
      () =>
        renameProjectService({
          projectId: created.project.id,
          name: "长".repeat(61),
          expectedRevision: created.project.revision,
          actorId: ids[0],
          db,
        }),
      400,
      "validation_error"
    );
    expect(getProject(created.project.id, db).members).toHaveLength(4);
  });
});

describe("markPlayedService", () => {
  function setup() {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    const fixture = created.project.fixtures[0];
    return { db, ids, projectId: created.project.id, fixture };
  }

  it("pending → played_unscored：缺省 playedAt 为上海当天", () => {
    const { db, ids, projectId, fixture } = setup();
    const result = markPlayedService({
      projectId,
      fixtureId: fixture.id,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      db,
    });
    expect(result.fixture.status).toBe("played_unscored");
    expect(result.fixture.playedAt).toBe(shanghaiToday());
    expect(result.fixture.markedBy).toBe(ids[0]);
  });

  it("显式 playedAt 生效；非法日期 400", () => {
    const { db, ids, projectId, fixture } = setup();
    const result = markPlayedService({
      projectId,
      fixtureId: fixture.id,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      playedAt: "2026-08-10",
      db,
    });
    expect(result.fixture.playedAt).toBe("2026-08-10");

    const other = getProject(projectId, db).fixtures[1];
    expectServiceError(
      () =>
        markPlayedService({
          projectId,
          fixtureId: other.id,
          expectedFixtureRevision: other.revision,
          actorId: ids[0],
          playedAt: "2026-02-30",
          db,
        }),
      400,
      "validation_error"
    );
  });

  it("重复标记/已 recorded/已 superseded/wrong revision 各自 409 语义", () => {
    const { db, ids, projectId, fixture } = setup();
    const played = markPlayedService({
      projectId,
      fixtureId: fixture.id,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      playedAt: "2026-08-10",
      db,
    });
    // 已 played_unscored 再标记 → 409 state_conflict。
    expectServiceError(
      () =>
        markPlayedService({
          projectId,
          fixtureId: fixture.id,
          expectedFixtureRevision: played.fixture.revision,
          actorId: ids[0],
          db,
        }),
      409,
      "state_conflict"
    );

    const scored = recordFixtureScoreService({
      projectId,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: played.fixture.revision,
      actorId: ids[0],
      db,
    });
    // 已 recorded → 409 already_recorded。
    expectServiceError(
      () =>
        markPlayedService({
          projectId,
          fixtureId: fixture.id,
          expectedFixtureRevision: scored.fixture!.revision,
          actorId: ids[0],
          db,
        }),
      409,
      "already_recorded"
    );

    // wrong revision（仍 pending 的另一场）→ 409 revision_conflict 携带当前 fixture。
    const pendingOther = getProject(projectId, db).fixtures.find(
      (f) => f.id !== fixture.id
    )!;
    const http = expectServiceError(
      () =>
        markPlayedService({
          projectId,
          fixtureId: pendingOther.id,
          expectedFixtureRevision: 999,
          actorId: ids[0],
          db,
        }),
      409,
      "revision_conflict"
    );
    expect((http.body.fixture as { id: number }).id).toBe(pendingOther.id);
  });

  it("superseded 场次 409 fixture_superseded", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    const fixture = created.project.fixtures[0];
    regenerateProjectService({
      projectId: created.project.id,
      playerIds: ids,
      matches: 0,
      expectedRevision: created.project.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    const superseded = getProject(created.project.id, db).superseded[0];
    expectServiceError(
      () =>
        markPlayedService({
          projectId: created.project.id,
          fixtureId: superseded.id,
          expectedFixtureRevision: superseded.revision,
          actorId: ids[0],
          db,
        }),
      409,
      "fixture_superseded"
    );
  });

  it("未知场次/不属于该项目的场次 404", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const first = createLegacyProject(db, ids);
    const second = createLegacyProject(db, ids);
    const foreign = first.project.fixtures[0];

    expectServiceError(
      () =>
        markPlayedService({
          projectId: second.project.id,
          fixtureId: foreign.id,
          expectedFixtureRevision: foreign.revision,
          actorId: ids[0],
          db,
        }),
      404,
      "not_found"
    );
    expectServiceError(
      () =>
        markPlayedService({
          projectId: first.project.id,
          fixtureId: 99999,
          expectedFixtureRevision: 0,
          actorId: ids[0],
          db,
        }),
      404,
      "not_found"
    );
  });
});

describe("recordFixtureScoreService（项目记分）", () => {
  function setup() {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    const fixture = created.project.fixtures[0];
    return { db, ids, projectId: created.project.id, fixture };
  }

  it("已 recorded 再记分 → 409 already_recorded（携带当前比分）；不产生新比赛", () => {
    const { db, ids, projectId, fixture } = setup();
    const first = recordFixtureScoreService({
      projectId,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      db,
    });

    const http = expectServiceError(
      () =>
        recordFixtureScoreService({
          projectId,
          fixtureId: fixture.id,
          scoreA: 18,
          scoreB: 21,
          expectedFixtureRevision: first.fixture!.revision,
          actorId: ids[0],
          db,
        }),
      409,
      "already_recorded"
    );
    expect((http.body.fixture as { scoreA: number | null }).scoreA).toBe(21);
    // 只有一条比赛。
    expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
  });

  it("superseded 场次记分 409 fixture_superseded", () => {
    const { db, ids, projectId } = setup();
    regenerateProjectService({
      projectId,
      playerIds: ids,
      matches: 0,
      expectedRevision: getProject(projectId, db).revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    const superseded = getProject(projectId, db).superseded[0];
    expectServiceError(
      () =>
        recordFixtureScoreService({
          projectId,
          fixtureId: superseded.id,
          scoreA: 21,
          scoreB: 9,
          expectedFixtureRevision: superseded.revision,
          actorId: ids[0],
          db,
        }),
      409,
      "fixture_superseded"
    );
  });

  it.each([
    [{ scoreA: 21, scoreB: 21 }, "equal"],
    [{ scoreA: -1, scoreB: 9 }, "negative"],
    [{ scoreA: 21.5, scoreB: 9 }, "float"],
  ])("非法比分 400 且不落库：%s", (scores) => {
    const { db, ids, projectId, fixture } = setup();
    expectServiceError(
      () =>
        recordFixtureScoreService({
          projectId,
          fixtureId: fixture.id,
          ...scores,
          expectedFixtureRevision: fixture.revision,
          actorId: ids[0],
          db,
        }),
      400,
      "validation_error"
    );
    expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(0);
  });

  it("评分服务故障：比赛已保存、场次 recorded、rating pending", () => {
    const { db, ids, projectId, fixture } = setup();
    initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, db);
    const spy = vi
      .spyOn(ratingService, "loadGlickoSnapshot")
      .mockImplementation(() => {
        throw new Error("glicko boom");
      });
    try {
      const result = recordFixtureScoreService({
        projectId,
        fixtureId: fixture.id,
        scoreA: 21,
        scoreB: 9,
        expectedFixtureRevision: fixture.revision,
        actorId: ids[0],
        db,
      });
      expect(result.rating.state).toBe("pending");
      expect(result.fixture?.status).toBe("recorded");
      expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
    } finally {
      spy.mockRestore();
    }
  });

  it("并发语义：第二个请求带着旧 revision 记分 → 409 读到已有比分", () => {
    const { db, ids, projectId, fixture } = setup();
    const first = recordFixtureScoreService({
      projectId,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      db,
    });
    expect(first.fixture?.status).toBe("recorded");

    // 第二个请求仍持有旧 revision：preflight 直接 409 already_recorded。
    const http = expectServiceError(
      () =>
        recordFixtureScoreService({
          projectId,
          fixtureId: fixture.id,
          scoreA: 18,
          scoreB: 21,
          expectedFixtureRevision: fixture.revision,
          actorId: ids[1],
          db,
        }),
      409,
      "already_recorded"
    );
    expect((http.body.fixture as { scoreB: number | null }).scoreB).toBe(9);
    expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
  });

  it("不同 fixture 互不影响（各自 revision CAS）", () => {
    const { db, ids, projectId, fixture } = setup();
    const other = getProject(projectId, db).fixtures[1];
    const scored = recordFixtureScoreService({
      projectId,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      db,
    });
    expect(scored.fixture?.status).toBe("recorded");
    // 另一场仍是 pending 且 revision 未变。
    const detail = getProject(projectId, db);
    const untouched = detail.fixtures.find((f) => f.id === other.id);
    expect(untouched?.status).toBe("pending");
    expect(untouched?.revision).toBe(other.revision);
  });
});

describe("旧入口联动服务", () => {
  it("noteMatchScoreEdited：被 fixture 引用的比赛改分后项目 revision++，未引用则无操作", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createLegacyProject(db, ids);
    const fixture = created.project.fixtures[0];
    const scored = recordFixtureScoreService({
      projectId: created.project.id,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      db,
    });
    const revisionBefore = getProject(created.project.id, db).revision;

    noteMatchScoreEdited(scored.id, db);
    expect(getProject(created.project.id, db).revision).toBe(revisionBefore + 1);
    // fixture 保持 recorded、比分未动。
    const after = getProject(created.project.id, db).fixtures[0];
    expect(after.status).toBe("recorded");
    expect(after.scoreA).toBe(21);

    noteMatchScoreEdited(999999, db);
    expect(getProject(created.project.id, db).revision).toBe(revisionBefore + 1);
  });

  it("mergePlayerWithScheduleRefs：无冲突合并迁移全部排阵引用并 bump 项目", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 6);
    const from = ids[4];
    const to = ids[5];
    const created = createLegacyProject(db, [ids[0], ids[1], ids[2], ids[3], from].sort((a, b) => Number(a) - Number(b)));
    const projectId = created.project.id;
    const fixture = created.project.fixtures[0];
    const revisionBefore = created.project.revision;

    // from 打一场真实比赛（from 与 to 不同场）。
    const matchId = db
      .prepare(
        `INSERT INTO matches (pa1, pa2, pb1, pb2, score_a, score_b, played_at, entered_by)
         VALUES (?, ?, ?, ?, 21, 15, '2026-08-01', ?)`
      )
      .run(Number(from), Number(ids[0]), Number(ids[1]), Number(ids[2]), Number(ids[0]))
      .lastInsertRowid as number;

    const result = mergePlayerWithScheduleRefs(Number(from), Number(to), db);
    expect(result.affectedProjects).toContain(projectId);

    // matches 引用迁移。
    const match = db
      .prepare(`SELECT pa1, entered_by FROM matches WHERE id = ?`)
      .get(matchId) as { pa1: number; entered_by: number | null };
    expect(match.pa1).toBe(Number(to));
    // 成员行迁移。
    const member = db
      .prepare(
        `SELECT 1 AS x FROM schedule_project_players WHERE project_id = ? AND player_id = ?`
      )
      .get(projectId, Number(to));
    expect(member).toBeDefined();
    // fixture 槽位迁移（TEXT）。
    const fx = getProject(projectId, db).fixtures.find((f) => f.id === fixture.id)!;
    const slots = [fx.pa1, fx.pa2, fx.pb1, fx.pb2];
    expect(slots).not.toContain(from);
    // created_by 迁移。
    expect(getProject(projectId, db).createdBy).not.toBe(from);
    // 项目 revision++。
    expect(getProject(projectId, db).revision).toBe(revisionBefore + 1);
    // 旧球员已删除。
    expect(
      db.prepare(`SELECT 1 FROM players WHERE id = ?`).get(Number(from))
    ).toBeUndefined();
  });

  it("merge 三类冲突全部拒绝且无部分改写", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 6);
    const [a, b, c, d, e, f] = ids;

    // 冲突 1：合并后真实比赛同场双人。
    db.prepare(
      `INSERT INTO matches (pa1, pa2, pb1, pb2, score_a, score_b, played_at)
       VALUES (?, ?, ?, ?, 21, 15, '2026-08-01')`
    ).run(Number(a), Number(b), Number(c), Number(d));
    const playersBefore = (db.prepare(`SELECT COUNT(*) AS c FROM players`).get() as { c: number }).c as number;
    expectServiceError(
      () => mergePlayerWithScheduleRefs(Number(a), Number(b), db),
      400,
      "state_conflict"
    );
    expect((db.prepare(`SELECT COUNT(*) AS c FROM players`).get() as { c: number }).c).toBe(playersBefore);
    const m = db.prepare(`SELECT pa1, pa2 FROM matches`).get() as { pa1: number; pa2: number };
    expect([m.pa1, m.pa2]).toEqual([Number(a), Number(b)]);

    // 清理比赛，准备 fixture 冲突：名单不含 b，再把 b 塞进 a 所在场次的非 a 槽位。
    db.prepare(`DELETE FROM matches`).run();
    const created = createLegacyProject(db, [a, c, d, e].sort((x, y) => Number(x) - Number(y)));
    const fixture = created.project.fixtures[0];
    const cols = ["pa1", "pa2", "pb1", "pb2"] as const;
    const slotValues = cols.map((col) => (fixture as unknown as Record<string, string>)[col]);
    const slotIndex = slotValues.findIndex((v) => v !== a);
    const targetCol = cols[slotIndex];
    db.prepare(`UPDATE schedule_fixtures SET ${targetCol} = ? WHERE id = ?`).run(b, fixture.id);
    const revisionBefore = getProject(created.project.id, db).revision;
    expectServiceError(
      () => mergePlayerWithScheduleRefs(Number(a), Number(b), db),
      400,
      "state_conflict"
    );
    // 无部分改写：槽位仍是 b、revision 未变。
    const fxAfter = getProject(created.project.id, db).fixtures.find((x) => x.id === fixture.id)!;
    expect((fxAfter as unknown as Record<string, string>)[targetCol]).toBe(b);
    expect(getProject(created.project.id, db).revision).toBe(revisionBefore);

    // 冲突 3：同项目双人成员。
    const created2 = createLegacyProject(db, [c, d, e, f].sort((x, y) => Number(x) - Number(y)));
    void created2;
    expectServiceError(
      () => mergePlayerWithScheduleRefs(Number(c), Number(e), db),
      400,
      "state_conflict"
    );

    // 目标不存在 400。
    expectServiceError(
      () => mergePlayerWithScheduleRefs(Number(c), 99999, db),
      400,
      "player_not_found"
    );
  });

  it("scheduleReferenceBlockReason：有排阵引用返回说明，无引用返回 null", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    createLegacyProject(db, ids);
    expect(scheduleReferenceBlockReason(Number(ids[0]), db)).toContain(
      "排阵项目"
    );
    const spare = addPlayer("无引用球员", db);
    expect(scheduleReferenceBlockReason(spare, db)).toBeNull();
  });
});

describe("toScheduleHttp", () => {
  it("ScheduleServiceError 直通；DB busy 映射 503 可重试", () => {
    const busy = toScheduleHttp(
      Object.assign(new Error("database is locked"), { code: "SQLITE_BUSY" })
    );
    expect(busy.status).toBe(503);
    expect(busy.body.code).toBe("db_busy");
    expect(busy.body.retryable).toBe(true);

    const internal = toScheduleHttp(new Error("boom"));
    expect(internal.status).toBe(500);
    expect(internal.body.code).toBe("internal_error");
  });
});

describe("生成快照与逐场胜率（generated_win_rate / input_json rating）", () => {
  it("创建项目：逐场胜率落库且在 [0,1]，可用快照评分复算一致", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const result = createLegacyProject(db, ids, { matches: 3 });

    const fixtures = result.project.fixtures;
    expect(fixtures).toHaveLength(3);

    const gen = db
      .prepare(
        `SELECT input_json FROM schedule_generations ORDER BY id DESC LIMIT 1`
      )
      .get() as { input_json: string };
    const snapshot = JSON.parse(gen.input_json) as {
      rating: {
        model: string;
        asOf: string;
        playerRatings: Record<string, number>;
      };
      budget: { restarts: number; proposalsPerRestart: number };
    };
    expect(snapshot.rating.model).toBe("legacy");
    expect(snapshot.rating.asOf).toBeTruthy();
    expect(Object.keys(snapshot.rating.playerRatings).sort()).toEqual(
      [...ids].sort()
    );
    expect(snapshot.budget).toEqual({
      restarts: 4,
      proposalsPerRestart: 3000,
    });

    for (const f of fixtures) {
      expect(f.generatedWinRate).not.toBeNull();
      const p = f.generatedWinRate as number;
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
      const recomputed = predictElo(
        f.pa1,
        f.pa2,
        f.pb1,
        f.pb2,
        snapshot.rating.playerRatings
      ).teamAWin;
      // 同一快照评分可复算出与落库一致的生成胜率（不冒充最新预测）。
      expect(p).toBeCloseTo(recomputed, 12);
    }
  });

  it("matches=0 保存名单不需要评分快照；恢复后续排新批次带胜率", () => {
    const db = mkDb();
    initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, db);
    const ids = seedPlayers(db, 4);
    const created = createProjectService({
      playerIds: ids,
      matches: 1,
      model: "glicko2",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    const before = getProject(created.project.id, db);

    // 快照随后不可用：matches>0 拒绝，matches=0 仍可保存名单。
    const spy = vi
      .spyOn(ratingService, "loadGlickoSnapshot")
      .mockImplementation(() => {
        throw new Error("glicko boom");
      });
    expectServiceError(
      () =>
        regenerateProjectService({
          projectId: created.project.id,
          playerIds: ids,
          matches: 1,
          seed: 7,
          expectedRevision: before.revision,
          actorId: ids[0],
          requestId: randomUUID(),
          db,
        }),
      409,
      "rating_unavailable"
    );
    const saved = regenerateProjectService({
      projectId: created.project.id,
      playerIds: ids,
      matches: 0,
      seed: 7,
      expectedRevision: before.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    expect(saved.project.revision).toBe(before.revision + 1);
    spy.mockRestore();

    const regen = regenerateProjectService({
      projectId: created.project.id,
      playerIds: ids,
      matches: 2,
      seed: 11,
      expectedRevision: saved.project.revision,
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    expect(regen.fixtures).toHaveLength(2);
    for (const f of regen.fixtures) {
      expect(f.generatedWinRate).not.toBeNull();
    }
  });
});
