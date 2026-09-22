import type Database from "better-sqlite3";
import { getDb } from "./db";
import { readRatingConfig } from "./rating-config";
import {
  loadGlickoSnapshot,
  type RatingServiceResult,
} from "./rating-service";
import { projectRatingView } from "./ratings/projections";
import { estimateNextMatch } from "./ratings/segment";
import { predictDoubles } from "./ratings/doubles";
import { shanghaiLocalDateFromInstant } from "./ratings/calendar";
import {
  predictElo,
  predictEloDeltas,
} from "./elo";
import { recomputeAllRatings, listPlayers } from "./repo";
import type { RatingModel, RatingState } from "./ratings/types";
import type { RatingView } from "./ratings/view-types";
import { loadLegacyStatsView, type StatsData } from "./stats";

export interface RatingViewOptions {
  /** 显式模型："glicko2" | "legacy"；非法值或缺省回 activeModel（无配置默认 legacy）。 */
  rating?: string;
  /** 可注入的评分时点（ISO instant），默认当前时间；一次调用只捕获一次。 */
  asOf?: string;
  db?: Database.Database;
}

/**
 * 统一展示 DTO。三种分支共享 model/asOf/version/freshness/nextBoundary 骨架：
 * - glicko2 fresh：ready 快照的直接投影；
 * - glicko2 stale：最近成功快照（lastGood）的投影，asOf 如实为旧时点；
 * - glicko2 unavailable：空结构，不伪造 1000 分；
 * - legacy：沿用旧 stats 语义（loadLegacyStatsView，version="legacy"）。
 * inputHash/currentSegmentId 供缓存指纹等消费方使用，legacy/unavailable 为 null。
 */
export type LoadRatingViewResult =
  | {
      model: "glicko2";
      asOf: string;
      version: string;
      /** ready 时为当前输入指纹；stale 时为 null（旧快照的指纹不代表当前输入）。 */
      inputHash: string | null;
      currentSegmentId: string;
      freshness: "fresh" | "stale";
      nextBoundary: string;
      view: RatingView;
    }
  | {
      model: "glicko2";
      asOf: string;
      version: null;
      inputHash: null;
      currentSegmentId: null;
      freshness: "unavailable";
      reason: string;
      nextBoundary: null;
      view: RatingView;
    }
  | {
      model: "legacy";
      asOf: string;
      version: "legacy";
      inputHash: null;
      currentSegmentId: null;
      freshness: "fresh";
      nextBoundary: null;
      view: StatsData;
    };

/** 无可用新版快照时的空结构：不伪造 1000 分、不补造历史线。 */
function emptyRatingView(): RatingView {
  return {
    players: [],
    points: [],
    weekSegments: [],
    matchEstimatesById: {},
    issues: [],
    peakFinal: {},
  };
}

/**
 * 按显式/默认模型构造统一展示 DTO。asOf 一次捕获贯穿本次评分；
 * 不自行重放，glicko2 分支完全经 loadGlickoSnapshot。
 */
export function loadRatingView(
  options: RatingViewOptions = {}
): LoadRatingViewResult {
  const conn = options.db ?? getDb();

  const requested = options.rating;
  let model: RatingModel;
  if (requested === "glicko2" || requested === "legacy") {
    model = requested;
  } else {
    // 非法参数或缺省回默认模型；配置损坏的抛错向上传播（不静默给旧数据）。
    model = readRatingConfig(conn)?.activeModel ?? "legacy";
  }

  const asOf = options.asOf ?? new Date().toISOString();

  if (model === "legacy") {
    return {
      model: "legacy",
      asOf,
      version: "legacy",
      inputHash: null,
      currentSegmentId: null,
      freshness: "fresh",
      nextBoundary: null,
      view: loadLegacyStatsView(conn),
    };
  }

  let result: RatingServiceResult;
  try {
    result = loadGlickoSnapshot(conn, asOf);
  } catch (error) {
    result = {
      state: "unavailable",
      model: "glicko2",
      reason: `rating service failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    };
  }

  if (result.state === "ready") {
    return {
      model: "glicko2",
      asOf,
      version: result.replay.configVersion,
      inputHash: result.inputHash,
      currentSegmentId: result.replay.currentSegment.id,
      freshness: "fresh",
      nextBoundary: result.replay.nextBoundary,
      view: projectRatingView(
        result.replay,
        listPlayers(conn).map((p) => ({ id: p.id, name: p.name }))
      ),
    };
  }

  if (result.state === "stale") {
    // stale 明确显示旧成功时点，不冒充当前。
    return {
      model: "glicko2",
      asOf: result.lastGood.asOf,
      version: result.lastGood.configVersion,
      inputHash: null,
      currentSegmentId: result.lastGood.currentSegment.id,
      freshness: "stale",
      nextBoundary: result.lastGood.nextBoundary,
      view: projectRatingView(
        result.lastGood,
        listPlayers(conn).map((p) => ({ id: p.id, name: p.name }))
      ),
    };
  }

  return {
    model: "glicko2",
    asOf,
    version: null,
    inputHash: null,
    currentSegmentId: null,
    freshness: "unavailable",
    reason: result.reason,
    nextBoundary: null,
    view: emptyRatingView(),
  };
}

/** 预测 DTO 输入：一场双打的四方整数球员 ID。 */
export interface PredictionViewOptions {
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  /** 显式模型："glicko2" | "legacy"；非法值或缺省回 activeModel（无配置默认 legacy）。 */
  rating?: string;
  /** 可注入的评分时点（ISO instant），默认当前时间；一次调用只捕获一次。 */
  asOf?: string;
  db?: Database.Database;
}

/** glicko2 分支单人的一种赛果模拟结果（引擎 PlayerChange 语义）。 */
export interface PredictionPlayerOutcome {
  playerId: number;
  before: RatingState;
  after: RatingState;
  delta: number;
}

/**
 * 显式模型化的预测 DTO：
 * - legacy：preWinA/每人赢输变化沿用 predictElo/predictEloDeltas 旧语义；
 * - glicko2 ready：preWinA 用 predictDoubles（replay.current 工作状态），
 *   赢/输两种赛果各用 estimateNextMatch 对 replay.current + currentSegment
 *   模拟一次，不人为换算 ELO K 值；引擎返回新对象，不修改 replay.current；
 * - stale/unavailable：如实标记 freshness，不给任何胜率/变化数字。
 */
export type LoadPredictionViewResult =
  | {
      model: "legacy";
      freshness: "fresh";
      asOf: string;
      version: "legacy";
      preWinA: number;
      players: Array<{ playerId: number; win: number; loss: number }>;
    }
  | {
      model: "glicko2";
      freshness: "fresh";
      asOf: string;
      version: string;
      inputHash: string;
      segmentId: string;
      preWinA: number;
      players: Array<{
        playerId: number;
        win: PredictionPlayerOutcome;
        loss: PredictionPlayerOutcome;
      }>;
    }
  | {
      model: "glicko2";
      freshness: "stale";
      asOf: string;
      version: string;
      lastGoodAsOf: string;
      reason: string;
    }
  | {
      model: "glicko2";
      freshness: "unavailable";
      asOf: string;
      version: null;
      reason: string;
    };

/** 预测用的合成比赛 ID：只用于单场模拟事件，不写入任何账本。 */
const PREDICTION_SYNTHETIC_MATCH_ID = 0;

export function loadPredictionView(
  options: PredictionViewOptions
): LoadPredictionViewResult {
  const conn = options.db ?? getDb();

  const requested = options.rating;
  let model: RatingModel;
  if (requested === "glicko2" || requested === "legacy") {
    model = requested;
  } else {
    model = readRatingConfig(conn)?.activeModel ?? "legacy";
  }

  const asOf = options.asOf ?? new Date().toISOString();
  const teamA = [options.pa1, options.pa2] as const;
  const teamB = [options.pb1, options.pb2] as const;
  const playerIds = [options.pa1, options.pa2, options.pb1, options.pb2];

  if (model === "legacy") {
    const ratings = recomputeAllRatings(conn);
    const eloRatings: Record<string, number> = Object.fromEntries(
      [...ratings].map(([id, r]) => [String(id), r.elo])
    );
    const stringIds = playerIds.map(String);
    const preWinA = predictElo(
      stringIds[0],
      stringIds[1],
      stringIds[2],
      stringIds[3],
      eloRatings
    ).teamAWin;
    const deltas = predictEloDeltas(
      stringIds[0],
      stringIds[1],
      stringIds[2],
      stringIds[3],
      eloRatings
    );
    return {
      model: "legacy",
      freshness: "fresh",
      asOf,
      version: "legacy",
      preWinA,
      players: playerIds.map((playerId) => ({
        playerId,
        win: deltas[String(playerId)].win,
        loss: deltas[String(playerId)].loss,
      })),
    };
  }

  let result: RatingServiceResult;
  try {
    result = loadGlickoSnapshot(conn, asOf);
  } catch (error) {
    result = {
      state: "unavailable",
      model: "glicko2",
      reason: `rating service failed: ${
        error instanceof Error ? error.message : String(error)
      }`,
    };
  }

  if (result.state === "stale") {
    // 旧成功快照可用于展示，但明确旧时点；stale 不用于新预测，不给数字。
    return {
      model: "glicko2",
      freshness: "stale",
      asOf: result.lastGood.asOf,
      version: result.lastGood.configVersion,
      lastGoodAsOf: result.lastGood.asOf,
      reason: result.reason,
    };
  }
  if (result.state === "unavailable") {
    return {
      model: "glicko2",
      freshness: "unavailable",
      asOf,
      version: null,
      reason: result.reason,
    };
  }

  // ready：配置必然可读（服务内部已读取成功）。
  const record = readRatingConfig(conn);
  if (record === null) {
    return {
      model: "glicko2",
      freshness: "unavailable",
      asOf,
      version: null,
      reason: "rating config disappeared after ready snapshot",
    };
  }
  const config = record.config;
  const states = result.replay.current;
  const preWinA = predictDoubles(teamA, teamB, states, config);

  // 赢/输两种赛果各模拟一次：只应用单场更新，不再加周过程方差；
  // estimateNextMatch 返回新状态对象，replay.current 不被修改。
  const playedAt = shanghaiLocalDateFromInstant(asOf);
  const baseMatch = {
    id: PREDICTION_SYNTHETIC_MATCH_ID,
    playedAt,
    createdAt: asOf,
    teamA,
    teamB,
  };
  const segment = result.replay.currentSegment;
  const winEstimate = estimateNextMatch(
    { ...baseMatch, scoreA: 21, scoreB: 0 },
    states,
    segment,
    config
  );
  const lossEstimate = estimateNextMatch(
    { ...baseMatch, scoreA: 0, scoreB: 21 },
    states,
    segment,
    config
  );

  const pick = (
    estimate: typeof winEstimate,
    playerId: number
  ): PredictionPlayerOutcome => {
    const change = estimate.changes.find((c) => c.playerId === playerId);
    if (change === undefined) {
      throw new RangeError(`prediction estimate missing player ${playerId}`);
    }
    return {
      playerId: change.playerId,
      before: { ...change.before },
      after: { ...change.after },
      delta: change.delta,
    };
  };

  return {
    model: "glicko2",
    freshness: "fresh",
    asOf,
    version: result.replay.configVersion,
    inputHash: result.inputHash,
    segmentId: segment.id,
    preWinA,
    players: playerIds.map((playerId) => ({
      playerId,
      win: pick(winEstimate, playerId),
      loss: pick(lossEstimate, playerId),
    })),
  };
}
