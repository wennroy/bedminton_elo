import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { createHash } from "node:crypto";
import {
  getWeekRange,
  computeWeeklyStats,
  weeklyDataVersion,
  weeklyDataVersionContext,
  WeeklyRatingUnavailableError,
  type MatchWithNames,
} from "./weekly";
import { closeDb, getDb } from "@/lib/db";
import { addMatch, addPlayer, listMatchesByDate, listPlayers } from "@/lib/repo";
import { initializeRatingConfig, readRatingConfig } from "@/lib/rating-config";
import { loadGlickoSnapshot } from "@/lib/rating-service";
import { loadRatingView } from "@/lib/rating-view";
import { predictDoubles } from "@/lib/ratings/doubles";

const players = [
  { id: 1, name: "Alice" },
  { id: 2, name: "Bob" },
  { id: 3, name: "Carol" },
  { id: 4, name: "Dave" },
  { id: 5, name: "Eve" },
  { id: 6, name: "Frank" },
  { id: 7, name: "Grace" },
  { id: 8, name: "Hank" },
];

function match(
  id: number,
  date: string,
  a1: number,
  a2: number,
  b1: number,
  b2: number,
  scoreA: number,
  scoreB: number
): MatchWithNames {
  const nameOf = (id: number) => players.find((p) => p.id === id)?.name ?? "?";
  return {
    id,
    pa1: a1,
    pa2: a2,
    pb1: b1,
    pb2: b2,
    scoreA,
    scoreB,
    playedAt: date,
    enteredBy: null,
    createdAt: `${date}T10:00:00Z`,
    pa1Name: nameOf(a1),
    pa2Name: nameOf(a2),
    pb1Name: nameOf(b1),
    pb2Name: nameOf(b2),
  };
}

const matches = [
  match(1, "2024-01-01", 1, 2, 3, 4, 21, 18), // Mon week 1
  match(2, "2024-01-02", 1, 2, 3, 4, 21, 19), // Tue week 1
  match(3, "2024-01-03", 1, 3, 2, 4, 18, 21), // Wed week 1
  match(4, "2024-01-04", 1, 3, 2, 4, 21, 17), // Thu week 1
  match(5, "2024-01-05", 1, 2, 3, 4, 21, 16), // Fri week 1
  match(6, "2024-01-08", 1, 2, 3, 4, 21, 15), // Mon week 2
];

describe("weekly", () => {
  it("computes week range from any date", () => {
    const range = getWeekRange("2024-01-03"); // Wed
    expect(range.weekStart).toBe("2024-01-01");
    expect(range.weekEnd).toBe("2024-01-07");
    expect(range.weekNumber).toBe(1);
  });

  it("handles Sunday as end of week", () => {
    const range = getWeekRange("2024-01-07"); // Sun
    expect(range.weekStart).toBe("2024-01-01");
    expect(range.weekEnd).toBe("2024-01-07");
  });

  it("aggregates attendance", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    expect(stats.attendance).toHaveLength(4);
    const alice = stats.attendance.find((s) => s.playerId === 1);
    expect(alice?.matches).toBe(5);
  });

  it("finds win king", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    const top = stats.winKing[0];
    expect(top.playerId).toBe(1);
    expect(top.wins).toBe(4);
    expect(top.playerId).toBe(1);
  });

  it("computes elo changes", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    const alice = stats.eloChanges.find((s) => s.playerId === 1);
    expect(alice).toBeDefined();
    expect(alice!.change).not.toBe(0);
  });

  it("finds best pair with at least 3 matches", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    expect(stats.bestPair).not.toBeNull();
    expect(stats.bestPair!.total).toBe(3);
    expect(stats.bestPair!.winRate).toBe(1);
    expect(stats.bestPair!.playerA).toBe("Alice");
    expect(stats.bestPair!.playerB).toBe("Bob");
  });

  it("returns empty stats for week without matches", () => {
    const stats = computeWeeklyStats("2024-02-05", players, matches);
    expect(stats.attendance).toHaveLength(0);
    expect(stats.winKing).toHaveLength(0);
    expect(stats.eloChanges).toHaveLength(0);
    expect(stats.bestPair).toBeNull();
    expect(stats.fun).toEqual({
      closest: null,
      blowout: null,
      streakKing: null,
      upset: null,
    });
  });

  describe("fun stats", () => {
    it("finds the closest match (smallest score diff)", () => {
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.closest).not.toBeNull();
      expect(stats.fun.closest!.date).toBe("2024-01-02");
      expect(stats.fun.closest!.scoreA).toBe(21);
      expect(stats.fun.closest!.scoreB).toBe(19);
      expect(stats.fun.closest!.teamA).toEqual(["Alice", "Bob"]);
      expect(stats.fun.closest!.teamB).toEqual(["Carol", "Dave"]);
    });

    it("finds the blowout match (largest score diff)", () => {
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.blowout).not.toBeNull();
      expect(stats.fun.blowout!.date).toBe("2024-01-05");
      expect(stats.fun.blowout!.scoreA).toBe(21);
      expect(stats.fun.blowout!.scoreB).toBe(16);
    });

    it("closest tie: higher winner score wins even when later", () => {
      // 22:20 is closer than 21:19 (deuce games need a 2-point margin)
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 19),
        match(2, "2024-03-05", 1, 2, 3, 4, 22, 20),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.closest!.date).toBe("2024-03-05");
      expect(stats.fun.closest!.scoreA).toBe(22);
      expect(stats.fun.closest!.scoreB).toBe(20);
    });

    it("closest full tie: earliest match wins", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 19),
        match(2, "2024-03-05", 1, 2, 3, 4, 21, 19),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.closest!.date).toBe("2024-03-04");
    });

    it("blowout tie: lower loser score wins even when later", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 30, 9), // diff 21, loser 9
        match(2, "2024-03-05", 1, 2, 3, 4, 21, 0), // diff 21, loser 0
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.blowout!.date).toBe("2024-03-05");
      expect(stats.fun.blowout!.scoreB).toBe(0);
    });

    it("blowout full tie: earliest match wins", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 5),
        match(2, "2024-03-05", 1, 2, 3, 4, 21, 5),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.blowout!.date).toBe("2024-03-04");
    });

    it("finds the streak king (longest in-week win streak)", () => {
      // main fixture: Bob wins m1, m2, m3 in a row -> streak 3; Alice max 2
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.streakKing).toEqual({
        playerId: 2,
        name: "Bob",
        streak: 3,
      });
    });

    it("streak king is null when nobody wins twice in a row", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 18),
        match(2, "2024-03-05", 3, 4, 1, 2, 21, 18),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.streakKing).toBeNull();
    });

    it("finds the upset in the main fixture", () => {
      // m4 (2024-01-04): Alice+Carol beat Bob+Dave.
      // Pre-match averages: A=992, B=1008 -> winnerProb = 1/(1+10^(16/400)).
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.upset).not.toBeNull();
      expect(stats.fun.upset!.date).toBe("2024-01-04");
      expect(stats.fun.upset!.teamA).toEqual(["Alice", "Carol"]);
      expect(stats.fun.upset!.teamB).toEqual(["Bob", "Dave"]);
      expect(stats.fun.upset!.winnerWinProb).toBeCloseTo(
        1 / (1 + 10 ** (16 / 400)),
        10
      );
    });

    it("upset is null when every winner was favored (>= 50%)", () => {
      // single match between all-unknown players -> winner prob exactly 0.5
      const ms = [match(1, "2024-03-04", 1, 2, 3, 4, 21, 18)];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.upset).toBeNull();
      expect(stats.fun.streakKing).toBeNull();
      expect(stats.fun.closest!.scoreA).toBe(21);
      expect(stats.fun.blowout!.scoreA).toBe(21);
    });

    it("upset picks the lowest winner win prob within the week only", () => {
      // Disjoint groups so probabilities stay independent.
      // After h1+h2: Alice/Bob ~= 1015.63, Carol/Dave ~= 984.37.
      // w1: Grace+Hank (992) beat Eve+Frank (1008) -> prob 1/(1+10^0.04) ~= 0.4770
      // w2: Carol+Dave (984.37) beat Alice+Bob (1015.63) -> prob ~= 0.4551
      // w2 comes later but is the bigger upset.
      const ms = [
        match(1, "2024-01-01", 1, 2, 3, 4, 21, 10),
        match(2, "2024-01-02", 1, 2, 3, 4, 21, 10),
        match(3, "2024-01-03", 5, 6, 7, 8, 21, 10),
        match(4, "2024-01-08", 7, 8, 5, 6, 21, 10),
        match(5, "2024-01-09", 3, 4, 1, 2, 21, 10),
      ];
      const week1 = computeWeeklyStats("2024-01-01", players, ms);
      expect(week1.fun.upset).toBeNull(); // week-1 matches were all >= 50%

      const week2 = computeWeeklyStats("2024-01-08", players, ms);
      expect(week2.fun.upset!.date).toBe("2024-01-09");
      expect(week2.fun.upset!.teamA).toEqual(["Carol", "Dave"]);
      expect(week2.fun.upset!.teamB).toEqual(["Alice", "Bob"]);
      expect(week2.fun.upset!.winnerWinProb).toBeCloseTo(0.455129, 6);
    });
  });

  describe("weeklyDataVersion", () => {
    it("is deterministic for identical stats", () => {
      const a = computeWeeklyStats("2024-01-01", players, matches);
      const b = computeWeeklyStats("2024-01-01", players, matches);
      expect(weeklyDataVersion(a)).toBe(weeklyDataVersion(b));
    });

    it("changes when a match is added to the week", () => {
      const before = computeWeeklyStats("2024-01-01", players, matches);
      const after = computeWeeklyStats("2024-01-01", players, [
        ...matches,
        match(7, "2024-01-06", 5, 6, 7, 8, 21, 10),
      ]);
      expect(weeklyDataVersion(after)).not.toBe(weeklyDataVersion(before));
    });

    it("changes when a score changes", () => {
      const before = computeWeeklyStats("2024-01-01", players, matches);
      const modified = matches.map((m) =>
        m.id === 1 ? { ...m, scoreB: 20 } : m
      );
      const after = computeWeeklyStats("2024-01-01", players, modified);
      expect(weeklyDataVersion(after)).not.toBe(weeklyDataVersion(before));
    });

    it("changes when a player is renamed", () => {
      const before = computeWeeklyStats("2024-01-01", players, matches);
      const renamed = players.map((p) =>
        p.id === 1 ? { ...p, name: "Alicia" } : p
      );
      const after = computeWeeklyStats("2024-01-01", renamed, matches);
      expect(weeklyDataVersion(after)).not.toBe(weeklyDataVersion(before));
    });

    it("differs across weeks", () => {
      const week1 = computeWeeklyStats("2024-01-01", players, matches);
      const week2 = computeWeeklyStats("2024-01-08", players, matches);
      expect(weeklyDataVersion(week1)).not.toBe(weeklyDataVersion(week2));
    });

    it("legacy 无 context 时指纹公式与旧版逐位一致", () => {
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      const expected = createHash("sha1")
        .update(JSON.stringify(stats))
        .digest("hex")
        .slice(0, 16);
      expect(weeklyDataVersion(stats)).toBe(expected);
      expect(weeklyDataVersionContext(stats)).toBeUndefined();
      // legacy stats 不设 ratingReport 字段，序列化形状与旧格式一致。
      expect("ratingReport" in stats).toBe(false);
    });

    it("context 覆盖边界状态：同 stats 不同 asOfDate/区段指纹不同", () => {
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      const base = {
        model: "glicko2" as const,
        version: "p1:2026-07-01",
        inputHash: "hash-a",
        freshness: "fresh" as const,
        asOfDate: "2026-10-05",
        asOfSegmentId: "2026-10-05:2026-10-05",
        nextBoundary: "2026-10-11T16:00:00Z",
      };
      const a = weeklyDataVersion(stats, base);
      expect(weeklyDataVersion(stats, { ...base })).toBe(a);
      // 跨周界：DB 未变、stats 未变，仅边界状态变化也必须失效。
      expect(
        weeklyDataVersion(stats, { ...base, asOfDate: "2026-10-12" })
      ).not.toBe(a);
      expect(
        weeklyDataVersion(stats, {
          ...base,
          asOfSegmentId: "2026-10-12:2026-10-12",
        })
      ).not.toBe(a);
      // 输入指纹变化（改名/改分）也失效。
      expect(
        weeklyDataVersion(stats, { ...base, inputHash: "hash-b" })
      ).not.toBe(a);
    });
  });
});

describe("weekly glicko2 分支", () => {
  const FIRST_SEASON_START = "2026-07-01";

  let dbPath: string;
  let dbPlayers: number[];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `weekly-glicko-${Date.now()}-${Math.random()}.db`);
    process.env.DATABASE_URL = dbPath;
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    dbPlayers = [
      addPlayer("A"),
      addPlayer("B"),
      addPlayer("C"),
      addPlayer("D"),
      addPlayer("E"),
      addPlayer("F"),
      addPlayer("G"),
      addPlayer("H"),
    ];
  });

  afterEach(() => {
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
    delete process.env.DATABASE_URL;
  });

  function glickoStats(week: string, asOf: string) {
    return computeWeeklyStats(week, listPlayers(getDb()), listMatchesByDate(getDb()), {
      rating: "glicko2",
      asOf,
      db: getDb(),
    });
  }

  it("当前周 Estimated：区段状态 estimated、无 Final，eloChanges 仍是 ELO", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-10-05" });
    const stats = glickoStats("2026-10-05", "2026-10-06T20:00:00+08:00");

    expect(stats.weekStart).toBe("2026-10-05");
    expect(stats.weekEnd).toBe("2026-10-11");
    expect(stats.ratingReport).toBeDefined();
    const report = stats.ratingReport!;
    expect(report.freshness).toBe("fresh");
    expect(report.model).toBe("glicko2");
    expect(report.inputHash).toBeTypeOf("string");
    expect(report.segments).toHaveLength(1);
    const segment = report.segments[0];
    expect(segment.status).toBe("estimated");
    expect(segment.segmentId).toBe(report.segmentId);
    expect(segment.players).toHaveLength(4);
    const aReport = segment.players.find((p) => p.playerId === a)!;
    expect(aReport.matchesPlayed).toBe(1);
    expect(aReport.estimatedChange).not.toBe(0);
    expect(aReport.finalR).toBeNull();
    expect(segment.correction).toEqual({});
    expect(report.resets).toHaveLength(0);
    // eloChanges 保持旧 ELO 语义（不是 glicko 数值）。
    expect(stats.eloChanges).toHaveLength(4);
    expect(stats.eloChanges.find((s) => s.playerId === a)!.change).not.toBe(0);
  });

  it("过周 Final：已结算区段带 correction 与 finalR", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-10-05" });
    const stats = glickoStats("2026-10-05", "2026-10-13T12:00:00+08:00");

    const report = stats.ratingReport!;
    expect(report.segments).toHaveLength(1);
    const segment = report.segments[0];
    expect(segment.status).toBe("final");
    expect(segment.players).toHaveLength(4);
    for (const p of segment.players) {
      expect(p.finalR).not.toBeNull();
      // 展示取整口径：finalR = round(estimatedEnd + correction)，
      // 与 endEstimatedR + correction 的整数差在 ±1 内。
      expect(Math.abs(p.finalR! - (p.endEstimatedR + p.correction))).toBeLessThanOrEqual(1);
    }
    // correction 与投影一致（数据源是 weekSegments，不自行重放）。
    const view = loadRatingView({ rating: "glicko2", asOf: "2026-10-13T12:00:00+08:00", db: getDb() });
    if (view.model !== "glicko2" || view.freshness === "unavailable") {
      throw new Error("expected glicko2 view");
    }
    const projected = view.view.weekSegments.find((s) => s.segmentId === segment.segmentId)!;
    for (const [playerId, value] of Object.entries(segment.correction)) {
      expect(value).toBe(Math.round(projected.correction[playerId]));
    }
  });

  it("周中季界：跨季周两段分列，重置单独列出", () => {
    const [a, b, c, d, e, f, g, h] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-29" });
    addMatch({ pa1: e, pa2: f, pb1: g, pb2: h, scoreA: 18, scoreB: 21, playedAt: "2026-10-02" });
    const stats = glickoStats("2026-09-28", "2026-10-03T12:00:00+08:00");

    expect(stats.weekStart).toBe("2026-09-28");
    const report = stats.ratingReport!;
    expect(report.segments).toHaveLength(2);
    const [first, second] = report.segments;
    expect(first.seasonId).toBe("2026-07-01");
    expect(first.status).toBe("final");
    expect(first.players.map((p) => p.playerId).sort()).toEqual([a, b, c, d].sort());
    expect(second.seasonId).toBe("2026-10-01");
    expect(second.status).toBe("estimated");
    expect(second.players.map((p) => p.playerId).sort()).toEqual([e, f, g, h].sort());
    // 季界重置只挂在触发段（第一段，end=2026-10-01 季界）。
    expect(report.resets).toHaveLength(1);
    expect(report.resets[0].seasonId).toBe("2026-10-01");
    expect(report.resets[0].segmentId).toBe(first.segmentId);
    expect(report.resets[0].changes).toHaveLength(4);
  });

  it("空周：最后一场之后的周可查看，当前区段无比赛", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-29" });
    const stats = glickoStats("2026-10-05", "2026-10-06T20:00:00+08:00");

    expect(stats.attendance).toHaveLength(0);
    expect(stats.winKing).toHaveLength(0);
    expect(stats.bestPair).toBeNull();
    expect(stats.fun).toEqual({
      closest: null,
      blowout: null,
      streakKing: null,
      upset: null,
    });
    const report = stats.ratingReport!;
    expect(report.segments).toHaveLength(1);
    expect(report.segments[0].status).toBe("estimated");
    expect(report.segments[0].players).toHaveLength(0);
    expect(stats.weekNumber).toBe(40);
  });

  it("无新比赛的周界跨越：estimated 变 final，校准与 Final 出现且数值一致", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-10-05" });
    const inWeek = glickoStats("2026-10-05", "2026-10-06T20:00:00+08:00");
    const inReport = inWeek.ratingReport!;
    expect(inReport.segments[0].status).toBe("estimated");
    // 校准尚未发生：不显示为 0 的「正式校准」，无 Final 值。
    expect(
      inReport.segments[0].players.every((p) => p.finalR === null)
    ).toBe(true);

    // 没有新比赛，仅时钟越过周一界：同一段从 Estimated 变 Final。
    const nextWeek = glickoStats("2026-10-05", "2026-10-12T20:00:00+08:00");
    const report = nextWeek.ratingReport!;
    expect(report.segments).toHaveLength(1);
    expect(report.segments[0].status).toBe("final");
    for (const p of report.segments[0].players) {
      expect(p.finalR).not.toBeNull();
      expect(
        Math.abs(p.finalR! - (p.endEstimatedR + p.correction))
      ).toBeLessThanOrEqual(1);
    }
    // 内容指纹随边界状态变化（DB 未变也要失效，供 OG ETag 使用）。
    const etagA = weeklyDataVersion(inWeek, weeklyDataVersionContext(inWeek));
    const etagB = weeklyDataVersion(nextWeek, weeklyDataVersionContext(nextWeek));
    expect(etagB).not.toBe(etagA);
  });

  it("冷门用赛前 Estimated 概率：无周末 Final 泄漏，补录重算不变", () => {
    const [a, b, c, d] = dbPlayers;
    // 第 1 周：A/B 两连胜 C/D，建立实力差距。
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 10, playedAt: "2026-09-14" });
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 10, playedAt: "2026-09-15" });
    // 第 2 周：C/D 爆冷胜 A/B。
    addMatch({ pa1: c, pa2: d, pb1: a, pb2: b, scoreA: 21, scoreB: 19, playedAt: "2026-09-21" });

    const asOfAfter = "2026-09-28T12:00:00+08:00";
    const stats = glickoStats("2026-09-21", asOfAfter);
    const upset = stats.fun.upset;
    expect(upset).not.toBeNull();
    expect(upset!.date).toBe("2026-09-21");

    // 冷门概率 = matchId 对应的赛前 Estimated 概率（胜方侧）。
    const view = loadRatingView({ rating: "glicko2", asOf: asOfAfter, db: getDb() });
    if (view.model !== "glicko2" || view.freshness === "unavailable") {
      throw new Error("expected glicko2 view");
    }
    const preMatch = view.view.matchEstimatesById["3"];
    expect(preMatch).toBeDefined();
    expect(upset!.winnerWinProb).toBe(preMatch.preWinA); // C/D 在 A 队侧获胜
    expect(upset!.winnerWinProb).toBeLessThan(0.5);

    // 泄漏检查：用周 Final 之后的状态重算同一对阵，胜率更高（赛后变强），
    // 报告仍用赛前预估值。
    const config = readRatingConfig(getDb())!.config;
    const snapshot = loadGlickoSnapshot(getDb(), asOfAfter);
    if (snapshot.state !== "ready") throw new Error("expected ready");
    const postFinalProb = predictDoubles(
      [c, d],
      [a, b],
      snapshot.replay.lastFinal,
      config
    );
    expect(upset!.winnerWinProb).toBeLessThan(postFinalProb);

    // 补录（同周更晚日期）后重算：该场冷门概率仍是其赛前预估值，不变。
    addMatch({ pa1: a, pa2: c, pb1: b, pb2: d, scoreA: 21, scoreB: 12, playedAt: "2026-09-24" });
    const recomputed = glickoStats("2026-09-21", asOfAfter);
    expect(recomputed.fun.upset).not.toBeNull();
    expect(recomputed.fun.upset!.winnerWinProb).toBe(upset!.winnerWinProb);
  });

  it("服务 unavailable：抛 WeeklyRatingUnavailableError，不伪造数据", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-14" });
    // 未初始化配置的库另行验证（此处先清掉配置模拟缺失）。
    getDb().prepare(`DELETE FROM meta WHERE key = 'ratings.config.v1'`).run();
    expect(() =>
      computeWeeklyStats(
        "2026-09-14",
        listPlayers(getDb()),
        listMatchesByDate(getDb()),
        { rating: "glicko2", asOf: "2026-09-15T12:00:00+08:00", db: getDb() }
      )
    ).toThrow(WeeklyRatingUnavailableError);
  });

  it("指纹：同一边界内稳定，跨边界（DB 未变）失效，改名后变化", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-14" });

    const asOfA = "2026-09-16T12:00:00+08:00";
    const statsA = glickoStats("2026-09-14", asOfA);
    const contextA = weeklyDataVersionContext(statsA)!;
    const versionA = weeklyDataVersion(statsA, contextA);
    // 同边界内再取一次：指纹稳定（asOf 原文毫秒值不入指纹）。
    const statsA2 = glickoStats("2026-09-14", "2026-09-16T13:30:00+08:00");
    expect(weeklyDataVersion(statsA2, weeklyDataVersionContext(statsA2)!)).toBe(
      versionA
    );

    // 跨周界：DB 未变，asOf 进入下一周 → 指纹不同。
    const asOfB = "2026-09-23T12:00:00+08:00";
    const statsB = glickoStats("2026-09-14", asOfB);
    const versionB = weeklyDataVersion(statsB, weeklyDataVersionContext(statsB)!);
    expect(versionB).not.toBe(versionA);
    expect(weeklyDataVersionContext(statsB)!.asOfSegmentId).not.toBe(
      contextA.asOfSegmentId
    );

    // 改名后：输入指纹变化 → 指纹变化。
    getDb().prepare(`UPDATE players SET name = 'A2' WHERE id = ?`).run(a);
    const statsRenamed = glickoStats("2026-09-14", asOfA);
    expect(
      weeklyDataVersion(statsRenamed, weeklyDataVersionContext(statsRenamed)!)
    ).not.toBe(versionA);
  });
});
