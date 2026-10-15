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
import * as repo from "./repo";
import * as ratingService from "./rating-service";
import {
  recordMatch,
  withdrawMatch,
  type MatchRatingField,
} from "./match-recording";
import {
  createProject,
  getProject,
  markPlayed,
  type FixtureDto,
} from "./schedule-projects";
import { recordFixtureScoreService } from "./schedule-service";
import type { SchedulingResult } from "./scheduling/types";

function mkDb(): Database.Database {
  const path = join(
    tmpdir(),
    `match-recording-test-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.db`
  );
  return createDb(path);
}

function seedPlayers(db: Database.Database, n: number): string[] {
  const ids: string[] = [];
  for (let i = 0; i < n; i++) {
    ids.push(String(addPlayer(`MR-P${i}`, db)));
  }
  return ids;
}

function engineResult(
  schedule: Array<[string, string, string, string]>
): SchedulingResult {
  return {
    schedule: schedule.map(([a1, a2, b1, b2]) => ({ a1, a2, b1, b2 })),
    metrics: {
      J: 0,
      imbalancedMatchCount: 0,
      totalImbalanceExcess: 0,
      directRepeatCount: 0,
    },
    algorithmVersion: "quota-local-v1",
    objectiveVersion: "balanced-v1",
    seed: 42,
    quotas: {},
    fairnessF: 0,
  };
}

function createProjectWith(
  db: Database.Database,
  playerIds: string[],
  schedule: Array<[string, string, string, string]>
) {
  return createProject(
    {
      name: "录分测试",
      model: "legacy",
      createdBy: playerIds[0],
      requestId: randomUUID(),
      payloadHash: `hash-${Math.random()}`,
      playerIds,
      engineInput: { playerIds, matches: schedule.length, seed: 1 },
    },
    engineResult(schedule),
    db
  );
}

function fixtureOf(
  db: Database.Database,
  projectId: number,
  fixtureId: number
): FixtureDto {
  const detail = getProject(projectId, db);
  const fx =
    detail.fixtures.find((f) => f.id === fixtureId) ??
    detail.superseded.find((f) => f.id === fixtureId);
  if (!fx) throw new Error("fixture missing");
  return fx;
}

describe("match-recording（公共录分服务）", () => {
  it("普通录入：before/after 四人齐全，无配置时 rating not_effective", () => {
    const db = mkDb();
    const [a, b, c, d] = seedPlayers(db, 4);

    const result = recordMatch({
      pa1: Number(a),
      pa2: Number(b),
      pb1: Number(c),
      pb2: Number(d),
      scoreA: 21,
      scoreB: 15,
      playedAt: "2026-08-01",
      db,
    });

    expect(result.id).toBeTypeOf("number");
    expect(result.before).toHaveLength(4);
    expect(result.after).toHaveLength(4);
    expect(result.legacyRatingsAvailable).toBe(true);
    expect(result.rating).toEqual({
      state: "not_effective",
      model: "glicko2",
      reason: "rating config not initialized",
    });
    expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
  });

  it("校验失败抛 MatchValidationError（未知球员），不落库", () => {
    const db = mkDb();
    const [a, b, c] = seedPlayers(db, 3);

    expect(() =>
      recordMatch({
        pa1: Number(a),
        pa2: Number(b),
        pb1: Number(c),
        pb2: 99999,
        scoreA: 21,
        scoreB: 15,
        playedAt: "2026-08-01",
        db,
      })
    ).toThrowError(/Unknown player id: 99999/);
    expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(0);
  });

  it("评分服务故障：比赛已保存、rating 降级 pending", () => {
    const db = mkDb();
    const [a, b, c, d] = seedPlayers(db, 4);
    const spy = vi
      .spyOn(ratingService, "loadGlickoSnapshot")
      .mockImplementation(() => {
        throw new Error("glicko boom");
      });
    try {
      const result = recordMatch({
        pa1: Number(a),
        pa2: Number(b),
        pb1: Number(c),
        pb2: Number(d),
        scoreA: 21,
        scoreB: 15,
        playedAt: "2026-08-01",
        db,
      });
      expect(result.rating.state).toBe("pending");
      expect((result.rating as { reason: string }).reason).toContain(
        "glicko boom"
      );
      expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
    } finally {
      spy.mockRestore();
    }
  });

  it("legacy 附加计算故障：before/after 空数组 + 显式不可用标记，比赛仍保存", () => {
    const db = mkDb();
    const [a, b, c, d] = seedPlayers(db, 4);
    const spy = vi
      .spyOn(repo, "recomputeAllRatings")
      .mockImplementation(() => {
        throw new Error("legacy boom");
      });
    try {
      const result = recordMatch({
        pa1: Number(a),
        pa2: Number(b),
        pb1: Number(c),
        pb2: Number(d),
        scoreA: 21,
        scoreB: 15,
        playedAt: "2026-08-01",
        db,
      });
      expect(result.before).toEqual([]);
      expect(result.after).toEqual([]);
      expect(result.legacyRatingsAvailable).toBe(false);
      expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(1);
    } finally {
      spy.mockRestore();
    }
  });

  it("项目记分（fixtureLink）：matches 与 attachScore 同事务提交，场次转 recorded", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createProjectWith(db, ids, [
      [ids[0], ids[1], ids[2], ids[3]],
    ]);
    const detail = getProject(created.id, db);
    const fixture = detail.fixtures[0];
    const revisionBefore = detail.revision;

    const result = recordFixtureScoreService({
      projectId: created.id,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 12,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      playedAt: "2026-08-02",
      db,
    });

    expect(result.fixture?.status).toBe("recorded");
    expect(result.fixture?.matchId).toBe(result.id);
    expect(result.fixture?.scoreA).toBe(21);
    expect(result.fixture?.scoreB).toBe(12);
    expect(result.rating.state).toBe("not_effective");

    const after = fixtureOf(db, created.id, fixture.id);
    expect(after.status).toBe("recorded");
    expect(after.playedAt).toBe("2026-08-02");
    // attachScore 同时 bump 项目 revision。
    expect(getProject(created.id, db).revision).toBe(revisionBefore + 1);
  });

  it("撤回（withdrawMatch）：recorded → played_unscored，保留 played_at，可重新登记", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createProjectWith(db, ids, [
      [ids[0], ids[1], ids[2], ids[3]],
    ]);
    const fixture = getProject(created.id, db).fixtures[0];

    const scored = recordFixtureScoreService({
      projectId: created.id,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 12,
      expectedFixtureRevision: fixture.revision,
      actorId: ids[0],
      playedAt: "2026-08-03",
      db,
    });

    const withdrawn = withdrawMatch(scored.id, db);
    expect(withdrawn.fixture?.status).toBe("played_unscored");
    expect(withdrawn.fixture?.matchId).toBeNull();
    expect(withdrawn.fixture?.playedAt).toBe("2026-08-03");
    // matches 行已删除，played_at「打过」事实保留在场次上。
    expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(0);

    // 可重新登记比分（沿用保留的 played_at 口径）。
    const rescored = recordFixtureScoreService({
      projectId: created.id,
      fixtureId: fixture.id,
      scoreA: 18,
      scoreB: 21,
      expectedFixtureRevision: withdrawn.fixture!.revision,
      actorId: ids[1],
      db,
    });
    expect(rescored.fixture?.status).toBe("recorded");
    expect(rescored.fixture?.scoreA).toBe(18);
    expect(rescored.fixture?.scoreB).toBe(21);
    expect(rescored.fixture?.playedAt).toBe("2026-08-03");
  });

  it("撤回无关联的普通比赛：直接删除，fixture 为 null", () => {
    const db = mkDb();
    const [a, b, c, d] = seedPlayers(db, 4);
    const plain = recordMatch({
      pa1: Number(a),
      pa2: Number(b),
      pb1: Number(c),
      pb2: Number(d),
      scoreA: 21,
      scoreB: 15,
      playedAt: "2026-08-01",
      db,
    });

    const withdrawn = withdrawMatch(plain.id, db);
    expect(withdrawn.fixture).toBeNull();
    expect((db.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }).c).toBe(0);
  });

  it("已 played_unscored 的场次走服务记分：沿用原日期", () => {
    const db = mkDb();
    const ids = seedPlayers(db, 4);
    const created = createProjectWith(db, ids, [
      [ids[0], ids[1], ids[2], ids[3]],
    ]);
    const fixture = getProject(created.id, db).fixtures[0];
    const played = markPlayed(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: fixture.revision,
        actorId: ids[0],
        playedAt: "2026-08-05",
      },
      db
    );

    const scored = recordFixtureScoreService({
      projectId: created.id,
      fixtureId: fixture.id,
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: played.revision,
      actorId: ids[0],
      // played_unscored 状态下传了 playedAt 也忽略，沿用 2026-08-05。
      playedAt: "2026-01-01",
      db,
    });
    expect(scored.fixture?.status).toBe("recorded");
    expect(scored.fixture?.playedAt).toBe("2026-08-05");
  });
});

describe("match-recording（MatchRatingField 形状兼容）", () => {
  it("buildMatchRatingField 返回三种判别形状之一", () => {
    const db = mkDb();
    const [a, b, c, d] = seedPlayers(db, 4);
    const result = recordMatch({
      pa1: Number(a),
      pa2: Number(b),
      pb1: Number(c),
      pb2: Number(d),
      scoreA: 21,
      scoreB: 15,
      playedAt: "2026-08-01",
      db,
    });
    const states: Array<MatchRatingField["state"]> = [
      "ready",
      "pending",
      "not_effective",
    ];
    expect(states).toContain(result.rating.state);
  });
});
