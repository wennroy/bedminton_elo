import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addMatch, addPlayer, listMatchesByDate, listPlayers } from "@/lib/repo";
import { initializeRatingConfig } from "@/lib/rating-config";
import { computeWeeklyStats } from "@/lib/weekly";
import { loadRatingView, type LoadRatingViewResult } from "@/lib/rating-view";
import {
  buildSeasonPageData,
  buildSeasonStats,
  computeSeasonStats,
  listSeasonIds,
  loadSeasonRatingParams,
  WeeklyRatingUnavailableError,
} from "@/lib/season";

type ReadyView = Extract<
  LoadRatingViewResult,
  { model: "glicko2"; freshness: "fresh" | "stale" }
>;

function readyView(asOf: string): ReadyView {
  const view = loadRatingView({ rating: "glicko2", asOf, db: getDb() });
  if (view.model !== "glicko2" || view.freshness === "unavailable") {
    throw new Error("expected glicko2 view");
  }
  return view;
}

function seasonStats(seasonId: string, asOf: string) {
  return computeSeasonStats(
    readyView(asOf),
    seasonId,
    listPlayers(getDb()),
    listMatchesByDate(getDb())
  );
}

describe("season glicko2 分支", () => {
  const FIRST_SEASON_START = "2026-07-01";

  let dbPath: string;
  let dbPlayers: number[];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `season-glicko-${Date.now()}-${Math.random()}.db`);
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

  it("期中态：跨季短段归属正确，期末取当前 displayRating", () => {
    const [a, b, c, d, e, f, g, h] = dbPlayers;
    // 同一周（2026-09-28 起）跨季界：前两天属 Q3，后两天属 Q4。
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-29" });
    addMatch({ pa1: e, pa2: f, pb1: g, pb2: h, scoreA: 18, scoreB: 21, playedAt: "2026-10-02" });
    const asOf = "2026-10-03T12:00:00+08:00";

    const q3 = seasonStats("2026-07-01", asOf);
    expect(q3.matches).toBe(1);
    expect(q3.inProgress).toBe(false);
    expect(q3.attendance.map((s) => s.playerId).sort()).toEqual([a, b, c, d].sort());

    const q4 = seasonStats("2026-10-01", asOf);
    expect(q4.label).toBe("2026年Q4");
    expect(q4.start).toBe("2026-10-01");
    expect(q4.end).toBe("2026-12-31");
    expect(q4.inProgress).toBe(true);
    expect(q4.asOfLocalDate).toBe("2026-10-03");
    expect(q4.matches).toBe(1);
    expect(q4.attendance.map((s) => s.playerId).sort()).toEqual([e, f, g, h].sort());

    // 期末 = 当前 displayRating：G/H 获胜并列第 1，E/F 同队同初值并列第 3。
    const view = readyView(asOf);
    const endByPlayer = new Map(
      view.view.players.map((p) => [p.playerId, p.displayRating])
    );
    for (const row of q4.rating) {
      expect(row.endR).toBe(endByPlayer.get(row.playerId));
      expect(row.isNewcomer).toBe(true);
      expect(row.startR).toBeNull();
      expect(row.change).toBeNull();
    }
    const rankOf = (id: number) => q4.rating.find((r) => r.playerId === id)!.rank;
    expect(rankOf(g)).toBe(1);
    expect(rankOf(h)).toBe(1);
    expect(rankOf(e)).toBe(3);
    expect(rankOf(f)).toBe(3);
  });

  it("已结束季：期末取该季最后一个周 Final，首季参赛者全是新人", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-07-06" });
    addMatch({ pa1: a, pa2: c, pb1: b, pb2: d, scoreA: 21, scoreB: 12, playedAt: "2026-07-13" });
    const asOf = "2026-10-13T12:00:00+08:00";

    const q3 = seasonStats("2026-07-01", asOf);
    expect(q3.inProgress).toBe(false);
    expect(q3.freshness).toBe("fresh");
    expect(q3.matches).toBe(2);
    expect(q3.attendance.find((s) => s.playerId === a)!.matches).toBe(2);

    // 期末 = 该季最后一个 weekly_final 的取整值。
    const view = readyView(asOf);
    const lastFinal = new Map<number, number>();
    for (const point of view.view.points) {
      if (point.season === "2026-07-01" && point.kind === "weekly_final") {
        lastFinal.set(point.playerId, Math.round(point.r));
      }
    }
    expect(q3.rating).toHaveLength(4);
    for (const row of q3.rating) {
      expect(row.isNewcomer).toBe(true);
      expect(row.startR).toBeNull();
      expect(row.change).toBeNull();
      expect(row.endR).toBe(lastFinal.get(row.playerId));
    }
  });

  it("期初口径：Q4 取季首重置后值（season_reset 事件点）", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 10, playedAt: "2026-09-14" });
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 10, playedAt: "2026-09-15" });
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-10-05" });
    // asOf 进入 2027Q1：Q4 已结束（期末取该季最后一个周 Final，非当前值）。
    const asOf = "2027-01-05T12:00:00+08:00";

    const q4 = seasonStats("2026-10-01", asOf);
    expect(q4.inProgress).toBe(false);
    expect(q4.rating).toHaveLength(4);

    const view = readyView(asOf);
    const lastFinal = new Map<number, number>();
    for (const point of view.view.points) {
      if (point.season === "2026-10-01" && point.kind === "weekly_final") {
        lastFinal.set(point.playerId, Math.round(point.r));
      }
    }
    for (const row of q4.rating) {
      expect(row.isNewcomer).toBe(false);
      expect(row.endR).toBe(lastFinal.get(row.playerId));
      const resetPoint = view.view.points.find(
        (p) =>
          p.kind === "season_reset" &&
          p.season === "2026-10-01" &&
          p.playerId === row.playerId
      );
      expect(resetPoint).toBeDefined();
      // 期初 = 重置后值，涨跌 = 期末 - 重置后期初。
      expect(row.startR).toBe(Math.round(resetPoint!.r));
      expect(row.change).toBe(row.endR - row.startR!);
    }
  });

  it("空季可渲染：无比赛季各版块为空但出现在列表里", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-14" });
    const asOf = "2026-10-03T12:00:00+08:00";

    const view = readyView(asOf);
    // Q4 仅靠季首重置点进入列表（无比赛），倒序排第一。
    expect(listSeasonIds(view.view)).toEqual(["2026-10-01", "2026-07-01"]);

    const data = buildSeasonPageData("2026-10-01", { asOf, db: getDb() });
    expect(data.currentSeasonId).toBe("2026-10-01");
    expect(data.stats).not.toBeNull();
    const stats = data.stats!;
    expect(stats.matches).toBe(0);
    expect(stats.inProgress).toBe(true);
    expect(stats.attendance).toEqual([]);
    expect(stats.winKing).toEqual([]);
    expect(stats.rating).toEqual([]);
    expect(stats.bestPair).toBeNull();
    expect(stats.fun).toEqual({
      closest: null,
      blowout: null,
      streakKing: null,
      upset: null,
    });

    // 缺省回当前季（Q4 空季）。
    const defaultData = buildSeasonPageData(null, { asOf, db: getDb() });
    expect(defaultData.stats?.seasonId).toBe("2026-10-01");
  });

  it("未知季返回 null，由页面空态承接", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-14" });
    const asOf = "2026-09-16T12:00:00+08:00";

    const data = buildSeasonPageData("2026-01-01", { asOf, db: getDb() });
    expect(data.seasons).toEqual(["2026-07-01"]);
    expect(data.stats).toBeNull();
  });

  it("仅首赛季起点前的数据：赛季列表为空", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-06-15" });
    const asOf = "2026-06-20T12:00:00+08:00";

    const view = readyView(asOf);
    expect(listSeasonIds(view.view)).toEqual([]);
    const data = buildSeasonPageData(null, { asOf, db: getDb() });
    expect(data.currentSeasonId).toBeNull();
    expect(data.stats).toBeNull();
  });

  it("冷门概率口径：赛前 Estimated 胜率（<50%），与 matchEstimatesById 一致", () => {
    const [a, b, c, d] = dbPlayers;
    // 第 1 周：A/B 两连胜 C/D 建立差距；第 2 周：C/D 爆冷。
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 10, playedAt: "2026-09-14" });
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 10, playedAt: "2026-09-15" });
    addMatch({ pa1: c, pa2: d, pb1: a, pb2: b, scoreA: 21, scoreB: 19, playedAt: "2026-09-21" });
    const asOf = "2026-09-28T12:00:00+08:00";

    const q3 = seasonStats("2026-07-01", asOf);
    const upset = q3.fun.upset;
    expect(upset).not.toBeNull();
    expect(upset!.date).toBe("2026-09-21");
    expect(upset!.winnerWinProb).toBeLessThan(0.5);

    const view = readyView(asOf);
    const preMatch = view.view.matchEstimatesById["3"];
    expect(preMatch).toBeDefined();
    // C/D 在 A 队侧获胜：胜方胜率 = preWinA（与周报 glicko2 分支同口径）。
    expect(upset!.winnerWinProb).toBe(preMatch.preWinA);
  });

  it("与周报同种子数据下同周数字一致（出勤/战绩王/趣闻）", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-14" });
    addMatch({ pa1: a, pa2: c, pb1: b, pb2: d, scoreA: 21, scoreB: 12, playedAt: "2026-09-15" });
    const asOf = "2026-09-16T20:00:00+08:00";

    const weekly = computeWeeklyStats("2026-09-14", listPlayers(getDb()), listMatchesByDate(getDb()), {
      rating: "glicko2",
      asOf,
      db: getDb(),
    });
    const season = seasonStats("2026-07-01", asOf);

    const byPlayer = (rows: { playerId: number; matches: number; wins: number; losses: number }[]) =>
      rows
        .map((r) => ({ playerId: r.playerId, matches: r.matches, wins: r.wins, losses: r.losses }))
        .sort((x, y) => x.playerId - y.playerId);
    expect(byPlayer(season.attendance)).toEqual(byPlayer(weekly.attendance));
    expect(byPlayer(season.winKing)).toEqual(byPlayer(weekly.winKing));
    expect(season.fun.closest).toEqual(weekly.fun.closest);
    expect(season.fun.blowout).toEqual(weekly.fun.blowout);
    expect(season.fun.streakKing).toEqual(weekly.fun.streakKing);
    expect(season.fun.upset).toEqual(weekly.fun.upset);
  });

  it("最佳组合门槛 ≥5 场，连胜王跨周累计", () => {
    const [a, b, c, d] = dbPlayers;
    // 两周各 3 场，A/B 组合 6 场全胜。
    for (const date of ["2026-09-14", "2026-09-15", "2026-09-16"]) {
      addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 12, playedAt: date });
    }
    for (const date of ["2026-09-21", "2026-09-22", "2026-09-23"]) {
      addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 14, playedAt: date });
    }
    const asOf = "2026-09-28T12:00:00+08:00";

    const q3 = seasonStats("2026-07-01", asOf);
    expect(q3.matches).toBe(6);
    expect(q3.bestPair).not.toBeNull();
    expect(q3.bestPair!.total).toBe(6);
    expect(q3.bestPair!.wins).toBe(6);
    expect(q3.bestPair!.winRate).toBe(1);

    // A 跨两周 6 连胜（周报口径只算周内，赛季口径跨周不断）。
    expect(q3.fun.streakKing).not.toBeNull();
    expect(q3.fun.streakKing!.playerId).toBe(a);
    expect(q3.fun.streakKing!.streak).toBe(6);
  });

  it("参数导出：持久化配置与未初始化兜底都给默认赛季参数", () => {
    expect(loadSeasonRatingParams(getDb())).toEqual({
      seasonLower: 900,
      seasonUpper: 1100,
      seasonRetention: 0.75,
      seasonRdFloor: 90,
    });

    closeDb();
    const barePath = join(tmpdir(), `season-bare-${Date.now()}-${Math.random()}.db`);
    process.env.DATABASE_URL = barePath;
    try {
      expect(loadSeasonRatingParams(getDb())).toEqual({
        seasonLower: 900,
        seasonUpper: 1100,
        seasonRetention: 0.75,
        seasonRdFloor: 90,
      });
    } finally {
      closeDb();
      try {
        unlinkSync(barePath);
      } catch {
        // ignore
      }
    }
  });

  it("服务 unavailable：buildSeasonStats 抛 WeeklyRatingUnavailableError", () => {
    const [a, b, c, d] = dbPlayers;
    addMatch({ pa1: a, pa2: b, pb1: c, pb2: d, scoreA: 21, scoreB: 15, playedAt: "2026-09-14" });
    getDb().prepare(`DELETE FROM meta WHERE key = 'ratings.config.v1'`).run();

    expect(() =>
      buildSeasonStats("2026-07-01", { asOf: "2026-09-16T12:00:00+08:00", db: getDb() })
    ).toThrow(WeeklyRatingUnavailableError);
    expect(() =>
      buildSeasonPageData(null, { asOf: "2026-09-16T12:00:00+08:00", db: getDb() })
    ).toThrow(WeeklyRatingUnavailableError);
  });
});
