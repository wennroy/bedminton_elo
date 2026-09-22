import type Database from "better-sqlite3";
import { getDb } from "./db";
import { readRatingConfig } from "./rating-config";
import {
  loadGlickoSnapshot,
  type RatingServiceResult,
} from "./rating-service";
import { projectRatingView } from "./ratings/projections";
import type { RatingModel } from "./ratings/types";
import type { RatingView } from "./ratings/view-types";
import { listPlayers } from "./repo";
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
 */
export type LoadRatingViewResult =
  | {
      model: "glicko2";
      asOf: string;
      version: string;
      freshness: "fresh" | "stale";
      nextBoundary: string;
      view: RatingView;
    }
  | {
      model: "glicko2";
      asOf: string;
      version: null;
      freshness: "unavailable";
      nextBoundary: null;
      view: RatingView;
    }
  | {
      model: "legacy";
      asOf: string;
      version: "legacy";
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
      freshness: "fresh",
      nextBoundary: null,
      view: loadLegacyStatsView(),
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
    freshness: "unavailable",
    nextBoundary: null,
    view: emptyRatingView(),
  };
}
