import { describe, it, expect, beforeEach, afterEach, vi, type MockInstance } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addMatch, addPlayer, recomputeAllRatings } from "@/lib/repo";
import {
  initializeRatingConfig,
  setActiveModel,
} from "@/lib/rating-config";
import { loadGlickoSnapshot } from "@/lib/rating-service";
import { loadPredictionView, loadRatingView } from "@/lib/rating-view";
import { predictElo, predictEloDeltas } from "@/lib/elo";
import { predictDoubles } from "@/lib/ratings/doubles";
import { estimateNextMatch } from "@/lib/ratings/segment";
import { shanghaiLocalDateFromInstant } from "@/lib/ratings/calendar";
import { readRatingConfig } from "@/lib/rating-config";
import { buildStatsData, loadLegacyStatsView } from "@/lib/stats";
import * as replayModule from "@/lib/ratings/replay";
import { ratingConfigVersion } from "@/lib/ratings/config";
import {
  applyRatingParam,
  isRatingModelParam,
} from "@/components/rating-mode-control";
import {
  formatStatusInstant,
  RATING_STATUS_EXPLAINERS,
  ratingStatusBanner,
  toRatingStatusInput,
} from "@/components/rating-status";
import { boundaryRefreshDelayMs } from "@/lib/use-rating-boundary-refresh";

const FIRST_SEASON_START = "2026-07-01";
const AS_OF_WEEK = "2026-10-06T20:00:00+08:00"; // 2026-10-05 起的一周

let dbPath: string;
let players: number[];
let replaySpy: MockInstance<typeof replayModule.replayRatings>;

function seedMatch(playedAt: string): number {
  return addMatch(
    {
      pa1: players[0],
      pa2: players[1],
      pb1: players[2],
      pb2: players[3],
      scoreA: 21,
      scoreB: 15,
      playedAt,
    },
    getDb()
  );
}

beforeEach(() => {
  closeDb();
  dbPath = join(tmpdir(), `rating-view-test-${Date.now()}-${Math.random()}.db`);
  process.env.DATABASE_URL = dbPath;
  players = [addPlayer("A"), addPlayer("B"), addPlayer("C"), addPlayer("D")];
  replaySpy = vi.spyOn(replayModule, "replayRatings");
});

afterEach(() => {
  replaySpy.mockRestore();
  closeDb();
  try {
    unlinkSync(dbPath);
  } catch {
    // ignore
  }
  delete process.env.DATABASE_URL;
});

describe("loadRatingView 模型选择", () => {
  it("无配置默认 legacy：view 为旧 stats 语义，version='legacy'", () => {
    const result = loadRatingView({ asOf: AS_OF_WEEK });
    expect(result.model).toBe("legacy");
    expect(result.version).toBe("legacy");
    expect(result.freshness).toBe("fresh");
    expect(result.nextBoundary).toBeNull();
    if (result.model !== "legacy") throw new Error("expected legacy");
    // 旧 stats 形状：players/matches/ratings(Map)/eloHistory/tsPlayers。
    expect(result.view.players.map((p) => p.name)).toEqual(["A", "B", "C", "D"]);
    expect(result.view.ratings).toBeInstanceOf(Map);
    expect(loadLegacyStatsView()).toEqual(buildStatsData());
  });

  it("非法 rating 参数回退 activeModel；activeModel=legacy 时默认走旧语义", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    // 默认 activeModel=legacy。
    expect(loadRatingView({ asOf: AS_OF_WEEK }).model).toBe("legacy");
    expect(loadRatingView({ rating: "bogus", asOf: AS_OF_WEEK }).model).toBe("legacy");
    expect(loadRatingView({ rating: "", asOf: AS_OF_WEEK }).model).toBe("legacy");
  });

  it("activeModel=glicko2 时非法参数回退到 glicko2；显式 legacy 仍走旧语义", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    setActiveModel("glicko2", getDb());
    expect(loadRatingView({ rating: "nonsense", asOf: AS_OF_WEEK }).model).toBe(
      "glicko2"
    );
    expect(loadRatingView({ asOf: AS_OF_WEEK }).model).toBe("glicko2");
    const explicitLegacy = loadRatingView({ rating: "legacy", asOf: AS_OF_WEEK });
    expect(explicitLegacy.model).toBe("legacy");
    expect(explicitLegacy.version).toBe("legacy");
  });
});

describe("loadRatingView glicko2 分支", () => {
  it("ready：fresh 投影，version=configVersion，nextBoundary 来自重放", () => {
    const record = initializeRatingConfig(
      { firstSeasonStart: FIRST_SEASON_START },
      getDb()
    );
    setActiveModel("glicko2", getDb());
    seedMatch("2026-10-05");

    const result = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    expect(result.model).toBe("glicko2");
    expect(result.freshness).toBe("fresh");
    expect(result.asOf).toBe(AS_OF_WEEK);
    expect(result.version).toBe(
      ratingConfigVersion(record.record.config)
    );
    expect(result.nextBoundary).toBeTypeOf("string");

    if (result.model !== "glicko2" || result.freshness !== "fresh") {
      throw new Error("expected glicko2 fresh");
    }
    const estimated = result.view.players.filter(
      (p) => p.status === "estimated"
    );
    expect(estimated).toHaveLength(4);
    expect(Object.keys(result.view.matchEstimatesById)).toHaveLength(1);
    expect(result.view.weekSegments.length).toBeGreaterThan(0);
  });

  it("stale：投影 lastGood，freshness='stale'，asOf 如实为快照旧时点", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    setActiveModel("glicko2", getDb());
    seedMatch("2026-10-05");
    const first = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    expect(first.freshness).toBe("fresh");

    // 改变输入 + 注入计算失败 → 服务只能给出旧成功快照。
    getDb().prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const stale = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    expect(stale.model).toBe("glicko2");
    expect(stale.freshness).toBe("stale");
    expect(stale.asOf).toBe(AS_OF_WEEK); // lastGood.asOf 即首次成功时点
    if (stale.model !== "glicko2" || stale.freshness !== "stale") {
      throw new Error("expected glicko2 stale");
    }
    // 投影来自 lastGood：该场逐场事件仍标 Estimated，人数与首次一致。
    expect(stale.view.players.filter((p) => p.status === "estimated")).toHaveLength(4);
    expect(Object.keys(stale.view.matchEstimatesById)).toHaveLength(1);
    expect(stale.view.matchEstimatesById["1"]?.kind).toBe("match_estimated");
  });

  it("unavailable：空结构、不伪造 1000 分", () => {
    // 未初始化配置 + 显式 glicko2：不回退 legacy。
    const result = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    expect(result.model).toBe("glicko2");
    expect(result.freshness).toBe("unavailable");
    expect(result.version).toBeNull();
    expect(result.nextBoundary).toBeNull();
    if (result.model !== "glicko2" || result.freshness !== "unavailable") {
      throw new Error("expected glicko2 unavailable");
    }
    expect(result.view.players).toEqual([]);
    expect(result.view.points).toEqual([]);
    expect(result.view.weekSegments).toEqual([]);
    expect(result.view.matchEstimatesById).toEqual({});
    expect(result.view.peakFinal).toEqual({});

    // 有配置但计算失败且无缓存 → 同样 unavailable。
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    setActiveModel("glicko2", getDb());
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const failed = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    expect(failed.freshness).toBe("unavailable");
  });

  it("跨周界不需要 DB 写入：asOf 越过 nextBoundary 后同一批人由 Estimated 变 Final", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    setActiveModel("glicko2", getDb());
    seedMatch("2026-10-05");

    const before = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    if (before.model !== "glicko2" || before.freshness !== "fresh") {
      throw new Error("expected glicko2 fresh");
    }
    expect(before.view.players.map((p) => p.status)).toEqual([
      "estimated",
      "estimated",
      "estimated",
      "estimated",
    ]);
    const boundary = Date.parse(before.nextBoundary);

    // 同一数据库、无新写入，仅把评分时点推过周界（读取时按 asOf 重新结算）。
    const after = loadRatingView({
      rating: "glicko2",
      asOf: new Date(boundary + 60_000).toISOString(),
    });
    if (after.model !== "glicko2" || after.freshness !== "fresh") {
      throw new Error("expected glicko2 fresh");
    }
    expect(after.currentSegmentId).not.toBe(before.currentSegmentId);
    expect(Date.parse(after.nextBoundary)).toBeGreaterThan(boundary);
    for (const player of after.view.players) {
      expect(player.status).toBe("final");
      expect(player.lastFinal).not.toBeNull();
    }
  });
});

describe("loadPredictionView", () => {
  it("legacy：preWinA 与每人赢输变化沿用 predictElo/predictEloDeltas", () => {
    // 无配置默认 legacy。
    const m1 = seedMatch("2026-10-05");
    expect(m1).toBeTypeOf("number");
    const result = loadPredictionView({
      pa1: players[0],
      pa2: players[1],
      pb1: players[2],
      pb2: players[3],
      asOf: AS_OF_WEEK,
    });
    expect(result.model).toBe("legacy");
    expect(result.freshness).toBe("fresh");
    if (result.model !== "legacy") throw new Error("expected legacy");

    const ratings = recomputeAllRatings(getDb());
    const eloRatings: Record<string, number> = Object.fromEntries(
      [...ratings].map(([id, r]) => [String(id), r.elo])
    );
    const s = players.map(String);
    expect(result.preWinA).toBe(predictElo(s[0], s[1], s[2], s[3], eloRatings).teamAWin);
    const deltas = predictEloDeltas(s[0], s[1], s[2], s[3], eloRatings);
    expect(result.players).toHaveLength(4);
    for (const p of result.players) {
      expect(p.win).toBe(deltas[String(p.playerId)].win);
      expect(p.loss).toBe(deltas[String(p.playerId)].loss);
    }
  });

  it("glicko2 ready：preWinA 用 predictDoubles，赢输模拟不改 replay.current", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    seedMatch("2026-10-05");

    const before = structuredClone(
      loadGlickoSnapshot(getDb(), AS_OF_WEEK)
    );
    if (before.state !== "ready") throw new Error("expected ready");

    const result = loadPredictionView({
      pa1: players[0],
      pa2: players[1],
      pb1: players[2],
      pb2: players[3],
      rating: "glicko2",
      asOf: AS_OF_WEEK,
    });
    expect(result.model).toBe("glicko2");
    expect(result.freshness).toBe("fresh");
    if (result.model !== "glicko2" || result.freshness !== "fresh") {
      throw new Error("expected glicko2 fresh");
    }
    expect(result.inputHash).toBe(before.inputHash);
    expect(result.segmentId).toBe(before.replay.currentSegment.id);

    const config = readRatingConfig(getDb())!.config;
    const expectedPreWinA = predictDoubles(
      [players[0], players[1]],
      [players[2], players[3]],
      before.replay.current,
      config
    );
    expect(result.preWinA).toBe(expectedPreWinA);

    // 赢/输变化 = estimateNextMatch 对 replay.current + currentSegment 各模拟一次。
    const playedAt = shanghaiLocalDateFromInstant(AS_OF_WEEK);
    const base = {
      id: 0,
      playedAt,
      createdAt: AS_OF_WEEK,
      teamA: [players[0], players[1]] as [number, number],
      teamB: [players[2], players[3]] as [number, number],
    };
    const win = estimateNextMatch(
      { ...base, scoreA: 21, scoreB: 0 },
      before.replay.current,
      before.replay.currentSegment,
      config
    );
    const loss = estimateNextMatch(
      { ...base, scoreA: 0, scoreB: 21 },
      before.replay.current,
      before.replay.currentSegment,
      config
    );
    for (const p of result.players) {
      const w = win.changes.find((c) => c.playerId === p.playerId)!;
      const l = loss.changes.find((c) => c.playerId === p.playerId)!;
      expect(p.win.delta).toBe(w.delta);
      expect(p.win.after).toEqual(w.after);
      expect(p.loss.delta).toBe(l.delta);
      expect(p.loss.after).toEqual(l.after);
      // 赢加输减（对 A 队而言），变化非零。
      expect(p.win.delta).not.toBe(0);
    }

    // 模拟不修改原工作状态：重放前后深度相等。
    const after = loadGlickoSnapshot(getDb(), AS_OF_WEEK);
    if (after.state !== "ready") throw new Error("expected ready");
    expect(after.replay.current).toEqual(before.replay.current);
  });

  it("glicko2 stale/unavailable：如实标记且不给胜率数字", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    // 无历史缓存 + 计算失败 → unavailable。
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const unavailable = loadPredictionView({
      pa1: players[0],
      pa2: players[1],
      pb1: players[2],
      pb2: players[3],
      rating: "glicko2",
      asOf: AS_OF_WEEK,
    });
    expect(unavailable.model).toBe("glicko2");
    expect(unavailable.freshness).toBe("unavailable");
    expect("preWinA" in unavailable).toBe(false);
    expect("players" in unavailable).toBe(false);

    // 先成功一次写入最近成功快照，再改输入 + 注入失败 → stale。
    replaySpy.mockRestore();
    seedMatch("2026-10-05");
    const first = loadPredictionView({
      pa1: players[0],
      pa2: players[1],
      pb1: players[2],
      pb2: players[3],
      rating: "glicko2",
      asOf: AS_OF_WEEK,
    });
    expect(first.freshness).toBe("fresh");
    getDb().prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    // mockRestore 后原 spy 已失效，需重建注入。
    replaySpy = vi.spyOn(replayModule, "replayRatings").mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const stale = loadPredictionView({
      pa1: players[0],
      pa2: players[1],
      pb1: players[2],
      pb2: players[3],
      rating: "glicko2",
      asOf: AS_OF_WEEK,
    });
    expect(stale.model).toBe("glicko2");
    expect(stale.freshness).toBe("stale");
    expect("preWinA" in stale).toBe(false);
    expect("players" in stale).toBe(false);
    if (stale.freshness === "stale") {
      expect(stale.lastGoodAsOf).toBe(AS_OF_WEEK);
    }
  });
});

describe("rating-status 状态映射", () => {
  it("legacy 结果不显示新版状态", () => {
    const result = loadRatingView({ rating: "legacy", asOf: AS_OF_WEEK });
    expect(toRatingStatusInput(result)).toEqual({ model: "legacy" });
    expect(ratingStatusBanner(toRatingStatusInput(result))).toBeNull();
  });

  it("fresh：正常状态显示模型版本与更新时点", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    setActiveModel("glicko2", getDb());
    seedMatch("2026-10-05");
    const result = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    const banner = ratingStatusBanner(toRatingStatusInput(result));
    if (banner === null) throw new Error("expected banner");
    expect(banner.tone).toBe("ok");
    expect(banner.title).toBe("新版评分运行中");
    expect(banner.meta).toContain(`更新于 ${formatStatusInstant(AS_OF_WEEK)}`);
    expect(banner.detail).toBeNull();
    expect(banner.showLegend).toBe(true);
  });

  it("stale：显示最后成功时点，不冒充当前时刻", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    setActiveModel("glicko2", getDb());
    seedMatch("2026-10-05");
    const first = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    expect(first.freshness).toBe("fresh");

    // 改变输入 + 注入计算失败；第二次读取使用更晚的 asOf。
    getDb().prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const laterAsOf = "2026-10-07T20:00:00+08:00";
    const stale = loadRatingView({ rating: "glicko2", asOf: laterAsOf });
    expect(stale.freshness).toBe("stale");
    // asOf 如实为最后成功时点，而非第二次请求的时点。
    expect(stale.asOf).toBe(AS_OF_WEEK);

    const banner = ratingStatusBanner(toRatingStatusInput(stale));
    if (banner === null) throw new Error("expected banner");
    expect(banner.tone).toBe("warn");
    expect(banner.title).toBe("新版评分暂未更新");
    expect(banner.detail).toContain(
      formatStatusInstant(AS_OF_WEEK)
    );
    expect(banner.detail).toContain("最后成功");
    expect(banner.meta).toContain("最后成功");
    expect(banner.showLegend).toBe(true);
  });

  it("unavailable（配置未初始化）：显示原因，不附状态说明", () => {
    const result = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    expect(result.freshness).toBe("unavailable");
    const banner = ratingStatusBanner(toRatingStatusInput(result));
    if (banner === null) throw new Error("expected banner");
    expect(banner.tone).toBe("error");
    expect(banner.title).toBe("新版评分暂不可用");
    expect(banner.detail).toContain("rating config not initialized");
    expect(banner.meta).toBeNull();
    expect(banner.showLegend).toBe(false);
  });

  it("Estimated/Final/未评级 说明各不相同且覆盖三种状态", () => {
    expect(RATING_STATUS_EXPLAINERS.map((e) => e.status)).toEqual([
      "estimated",
      "final",
      "unrated",
    ]);
    const descriptions = new Set(
      RATING_STATUS_EXPLAINERS.map((e) => e.description)
    );
    expect(descriptions.size).toBe(3);
  });

  it("formatStatusInstant 输出本地「M月D日 HH:MM」", () => {
    const iso = "2026-10-06T12:34:56Z";
    const d = new Date(iso);
    const expected = `${d.getMonth() + 1}月${d.getDate()}日 ${String(
      d.getHours()
    ).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    expect(formatStatusInstant(iso)).toBe(expected);
    expect(formatStatusInstant(iso)).toMatch(/^\d{1,2}月\d{1,2}日 \d{2}:\d{2}$/);
  });
});

describe("rating-mode-control 查询参数", () => {
  it("isRatingModelParam 只接受 glicko2/legacy，其余非法", () => {
    expect(isRatingModelParam("glicko2")).toBe(true);
    expect(isRatingModelParam("legacy")).toBe(true);
    for (const bad of ["Glicko2", "glicko", "bogus", "", null, undefined, 2]) {
      expect(isRatingModelParam(bad)).toBe(false);
    }
  });

  it("applyRatingParam 保留 week/pa1..pb2 等其他查询参数", () => {
    // weekly 的 week 与 predict 的预填阵容都保留，rating 追加在末尾。
    expect(
      applyRatingParam("week=2026-10-05&pa1=1&pa2=2&pb1=3&pb2=4", "glicko2")
    ).toBe("week=2026-10-05&pa1=1&pa2=2&pb1=3&pb2=4&rating=glicko2");
    // 已带 rating 时覆盖原值而不是追加。
    expect(applyRatingParam("rating=glicko2&week=2026-10-05", "legacy")).toBe(
      "rating=legacy&week=2026-10-05"
    );
    // 无其他参数时只保留 rating。
    expect(applyRatingParam("", "glicko2")).toBe("rating=glicko2");
  });
});

describe("use-rating-boundary-refresh 边界计算", () => {
  const NOW = Date.parse("2026-10-06T12:00:00Z");

  it("nextBoundary 为 null（legacy/unavailable）或非法 ISO 时不设定时", () => {
    expect(boundaryRefreshDelayMs(null, NOW)).toBeNull();
    expect(boundaryRefreshDelayMs("not-a-date", NOW)).toBeNull();
  });

  it("未到边界返回剩余毫秒；已到/已过边界回 0", () => {
    expect(boundaryRefreshDelayMs("2026-10-11T16:00:00Z", NOW)).toBe(
      Date.parse("2026-10-11T16:00:00Z") - NOW // 5 天 4 小时
    );
    expect(boundaryRefreshDelayMs("2026-10-06T12:00:00Z", NOW)).toBe(0);
    expect(boundaryRefreshDelayMs("2026-10-01T12:00:00Z", NOW)).toBe(0);
  });

  it("glicko2 fresh 结果的 nextBoundary 为 ISO 可定时；legacy 为 null", () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    setActiveModel("glicko2", getDb());
    seedMatch("2026-10-05");

    const fresh = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK });
    if (fresh.model !== "glicko2") throw new Error("expected glicko2");
    expect(typeof fresh.nextBoundary).toBe("string");
    const delay = boundaryRefreshDelayMs(fresh.nextBoundary, NOW);
    expect(delay).not.toBeNull();
    expect(delay).toBeGreaterThan(0);

    const legacy = loadRatingView({ rating: "legacy", asOf: AS_OF_WEEK });
    expect(legacy.nextBoundary).toBeNull();
    expect(boundaryRefreshDelayMs(legacy.nextBoundary, NOW)).toBeNull();
  });
});
