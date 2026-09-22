import { describe, it, expect, beforeEach, afterEach, vi, type MockInstance } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addMatch, addPlayer } from "@/lib/repo";
import {
  initializeRatingConfig,
  setActiveModel,
} from "@/lib/rating-config";
import { loadRatingView } from "@/lib/rating-view";
import { buildStatsData, loadLegacyStatsView } from "@/lib/stats";
import * as replayModule from "@/lib/ratings/replay";
import { ratingConfigVersion } from "@/lib/ratings/config";

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
});
