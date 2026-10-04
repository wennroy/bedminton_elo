import type Database from "better-sqlite3";
import {
  listMatchesByDate,
  listPlayers,
  type MatchWithNames,
} from "@/lib/repo";
import { readRatingConfig } from "@/lib/rating-config";
import { DEFAULT_RATING_PARAMS } from "@/lib/ratings/config";
import {
  formatTrendSeasonLabel,
  listTrendSeasons,
} from "@/lib/ratings/chart-data";
import {
  nextQuarterStart,
  shanghaiLocalDateFromInstant,
} from "@/lib/ratings/calendar";
import { loadRatingView, type LoadRatingViewResult } from "@/lib/rating-view";
import type { RatingView } from "@/lib/ratings/view-types";
import {
  WeeklyRatingUnavailableError,
  type BestPair,
  type FunMatch,
  type UpsetMatch,
  type WeeklyFun,
  type WeeklyPlayerStat,
} from "@/lib/weekly";

export { WeeklyRatingUnavailableError };
export type { BestPair, FunMatch, UpsetMatch };

/** 季内搭档入围门槛：赛季口径严于周报（周报为 3 场）。 */
const SEASON_PAIR_MIN_MATCHES = 5;

/** 赛季排名 + 涨跌行：只列季内参赛球员，按期末展示值竞赛排名。 */
export interface SeasonRatingRow {
  playerId: number;
  name: string;
  /** 竞争排名（1,2,2,4），并列同名次；与 RatingViewPlayer.rank 同口径。 */
  rank: number | null;
  /** 期初展示值：季首重置后值 / 首周 Final；本季新加入为 null（不虚构期初）。 */
  startR: number | null;
  /** 期末展示值：已结束季取该季最后一个周 Final，期中季取当前 displayRating。 */
  endR: number;
  /** endR - startR；新人为 null。 */
  change: number | null;
  /** 全局首个评分事件落在本季（首事件为逐场预估），无季初口径。 */
  isNewcomer: boolean;
  /** 季内参赛场次。 */
  matchesPlayed: number;
}

/**
 * 赛季统计：版块形状镜像 WeeklyStats（出勤/战绩王/最佳组合/趣闻复用其
 * 行类型），排名与涨跌为期初→期末口径。数据源纯 RatingView 投影 +
 * 比赛事实表，不自行重放。
 */
export interface SeasonStats {
  seasonId: string;
  /** 展示标签（"2026年Q4"，复用 formatTrendSeasonLabel）。 */
  label: string;
  /** 季起（= seasonId）。 */
  start: string;
  /** 季止（下一季首前一日）。 */
  end: string;
  /** 评分时点（ISO instant）。 */
  asOf: string;
  /** asOf 的上海日期（供「截至 M月d日」展示，纯字符串派生无时区风险）。 */
  asOfLocalDate: string;
  /** 期中态：asOf 所在区段属于本季（最后一周未结算）。 */
  inProgress: boolean;
  version: string;
  inputHash: string | null;
  freshness: "fresh" | "stale";
  /** 季内比赛场数（空季为 0）。 */
  matches: number;
  attendance: WeeklyPlayerStat[];
  winKing: WeeklyPlayerStat[];
  rating: SeasonRatingRow[];
  bestPair: BestPair | null;
  fun: WeeklyFun;
}

/** glicko2 ready/stale 统一展示结果（与 weekly.ts 内部类型同口径）。 */
type ReadyOrStaleRatingView = Extract<
  LoadRatingViewResult,
  { model: "glicko2"; freshness: "fresh" | "stale" }
>;

/** 简介卡展示所需的当前生效赛季参数：持久化配置优先，未初始化用默认常量。 */
export interface SeasonRatingParams {
  seasonLower: number;
  seasonUpper: number;
  seasonRetention: number;
  seasonRdFloor: number;
}

export function loadSeasonRatingParams(db?: Database.Database): SeasonRatingParams {
  const config = readRatingConfig(db)?.config ?? DEFAULT_RATING_PARAMS;
  return {
    seasonLower: config.seasonLower,
    seasonUpper: config.seasonUpper,
    seasonRetention: config.seasonRetention,
    seasonRdFloor: config.seasonRdFloor,
  };
}

/** 赛季列表：points 中 season 非空去重倒序（升序来自 listTrendSeasons）。 */
export function listSeasonIds(view: RatingView): string[] {
  return listTrendSeasons(view.points).reverse();
}

/** 下一季首前一日（纯 UTC 数学，不依赖主机时区）。 */
function quarterEndLocal(seasonId: string): string {
  const next = nextQuarterStart(seasonId);
  const [year, month, day] = next.split("-").map(Number);
  const end = new Date(Date.UTC(year, month - 1, day) - 86400000);
  return [
    end.getUTCFullYear(),
    String(end.getUTCMonth() + 1).padStart(2, "0"),
    String(end.getUTCDate()).padStart(2, "0"),
  ].join("-");
}

/**
 * 纯函数核心：从投影 + 赛季 ID + 比赛事实计算赛季统计。
 * 季内比赛 = 该季各周区段（含跨季短段）matches 里的 matchId 对应事实，
 * 与投影归属天然一致，且不混入 asOf 之后的补录。
 */
export function computeSeasonStats(
  result: ReadyOrStaleRatingView,
  seasonId: string,
  players: ReadonlyArray<{ id: number; name: string }>,
  allMatches: readonly MatchWithNames[]
): SeasonStats {
  const view = result.view;
  const nameMap = new Map(players.map((p) => [p.id, p.name]));

  const seasonSegments = view.weekSegments.filter(
    (segment) => segment.seasonId === seasonId
  );
  const seasonMatchIds = new Set<number>();
  for (const segment of seasonSegments) {
    for (const estimate of segment.matches) seasonMatchIds.add(estimate.matchId);
  }
  const seasonMatches = allMatches.filter((m) => seasonMatchIds.has(m.id));

  const currentSeasonId =
    view.weekSegments.find((s) => s.segmentId === result.currentSegmentId)
      ?.seasonId ?? null;
  const inProgress = currentSeasonId !== null && seasonId === currentSeasonId;

  // 冷门：赛前 Estimated 概率（matchEstimates 按 matchId 取），引擎未计分
  // 的记录（坏数据/未来/未知球员）不参与——与周报 glicko2 分支同口径。
  const winnerProbByMatchId = new Map<number, number>();
  for (const m of seasonMatches) {
    const estimate = view.matchEstimatesById[String(m.id)];
    if (estimate === undefined) continue;
    winnerProbByMatchId.set(
      m.id,
      m.scoreA > m.scoreB ? estimate.preWinA : 1 - estimate.preWinA
    );
  }

  const facts = computeSeasonFacts(seasonMatches, nameMap, winnerProbByMatchId);
  const rating = computeSeasonRatingRows(view, seasonId, inProgress, facts.attendance);

  return {
    seasonId,
    label: formatTrendSeasonLabel(seasonId),
    start: seasonId,
    end: quarterEndLocal(seasonId),
    asOf: result.asOf,
    asOfLocalDate: shanghaiLocalDateFromInstant(result.asOf),
    inProgress,
    version: result.version,
    inputHash: result.inputHash,
    freshness: result.freshness,
    matches: seasonMatches.length,
    attendance: facts.attendance,
    winKing: facts.winKing,
    rating,
    bestPair: facts.bestPair,
    fun: facts.fun,
  };
}

/**
 * 排名与涨跌：期初 = 本季首个 season_reset / weekly_final 事件点（重置后值）；
 * 期末 = 已结束季最后一个 weekly_final，期中季 = 当前 displayRating；
 * 全局首事件落在本季的球员标「本季新加入」，startR/change 给 null 不虚构期初。
 */
function computeSeasonRatingRows(
  view: RatingView,
  seasonId: string,
  inProgress: boolean,
  attendance: readonly WeeklyPlayerStat[]
): SeasonRatingRow[] {
  const seasonPoints = view.points.filter((p) => p.season === seasonId);

  const firstOrderByPlayer = new Map<number, number>();
  const startRByPlayer = new Map<number, number>();
  const endFinalByPlayer = new Map<number, number>();
  const lastPointRByPlayer = new Map<number, number>();
  for (const point of view.points) {
    const existing = firstOrderByPlayer.get(point.playerId);
    if (existing === undefined || point.order < existing) {
      firstOrderByPlayer.set(point.playerId, point.order);
    }
  }
  for (const point of seasonPoints) {
    if (point.kind !== "match_estimated" && !startRByPlayer.has(point.playerId)) {
      startRByPlayer.set(point.playerId, Math.round(point.r));
    }
    if (point.kind === "weekly_final") {
      endFinalByPlayer.set(point.playerId, Math.round(point.r));
    }
    lastPointRByPlayer.set(point.playerId, Math.round(point.r));
  }

  const matchesPlayedByPlayer = new Map<number, number>();
  for (const row of attendance) {
    matchesPlayedByPlayer.set(row.playerId, row.matches);
  }

  const rows: SeasonRatingRow[] = [];
  for (const [playerId, matchesPlayed] of matchesPlayedByPlayer) {
    const endR = inProgress
      ? view.players.find((p) => p.playerId === playerId)?.displayRating
      : (endFinalByPlayer.get(playerId) ?? lastPointRByPlayer.get(playerId));
    if (endR === undefined || endR === null) continue;

    // 新人 = 全局首个评分事件落在本季（首事件必为逐场预估）：期初不虚构。
    // 非新人必有季首口径：上季在册球员有 season_reset 点，首季前参赛者
    // （season 为 null 的点）退化为本季首个 weekly_final。
    const isNewcomer = seasonPoints.some(
      (p) => p.playerId === playerId && p.order === firstOrderByPlayer.get(playerId)
    );
    const startR = isNewcomer ? null : (startRByPlayer.get(playerId) ?? null);
    rows.push({
      playerId,
      name: view.players.find((p) => p.playerId === playerId)?.name ?? "?",
      rank: null,
      startR,
      endR,
      change: startR === null ? null : endR - startR,
      isNewcomer,
      matchesPlayed,
    });
  }

  // 竞争排名：期末展示值降序，并列同名次（与 RatingViewPlayer.rank 同口径）。
  rows.sort((a, b) => b.endR - a.endR || a.playerId - b.playerId);
  let index = 0;
  while (index < rows.length) {
    let end = index + 1;
    while (end < rows.length && rows[end].endR === rows[index].endR) end++;
    const rank = index + 1;
    for (let i = index; i < end; i++) rows[i].rank = rank;
    index = end;
  }
  return rows;
}

export interface BuildSeasonStatsOptions {
  /** 可注入的评分时点（ISO instant），默认当前时间；一次调用只捕获一次。 */
  asOf?: string;
  db?: Database.Database;
}

/**
 * 赛季统计入口（glicko2 专用）：内部 loadRatingView，镜像 buildWeeklyStats
 * 的用法；快照不可用抛 WeeklyRatingUnavailableError，由路由如实拒答。
 */
export function buildSeasonStats(
  seasonId: string,
  options: BuildSeasonStatsOptions = {}
): SeasonStats {
  const conn = options.db;
  const view = loadRatingView({ rating: "glicko2", asOf: options.asOf, db: conn });
  if (view.model !== "glicko2") {
    throw new Error("unreachable: loadRatingView returned a different model");
  }
  if (view.freshness === "unavailable") {
    throw new WeeklyRatingUnavailableError(view.reason);
  }
  return computeSeasonStats(view, seasonId, listPlayers(conn), listMatchesByDate(conn));
}

/** 赛季页数据：赛季列表 + 当前季 + 目标季统计（未知季为 null，由页面空态）。 */
export interface SeasonPageData {
  /** 倒序赛季列表（points 中 season 非空去重）。 */
  seasons: string[];
  /** asOf 所在区段的赛季（首赛季起点前为 null）。 */
  currentSeasonId: string | null;
  /** 请求赛季的统计；season 不在列表中（含赛季列表为空）时为 null。 */
  stats: SeasonStats | null;
}

/** 一次加载赛季页全部数据：视图只取一次，赛季统计从同一投影派生。 */
export function buildSeasonPageData(
  requestedSeason: string | null,
  options: BuildSeasonStatsOptions = {}
): SeasonPageData {
  const conn = options.db;
  const view = loadRatingView({ rating: "glicko2", asOf: options.asOf, db: conn });
  if (view.model !== "glicko2") {
    throw new Error("unreachable: loadRatingView returned a different model");
  }
  if (view.freshness === "unavailable") {
    throw new WeeklyRatingUnavailableError(view.reason);
  }

  const seasons = listSeasonIds(view.view);
  const currentSeasonId =
    view.view.weekSegments.find((s) => s.segmentId === view.currentSegmentId)
      ?.seasonId ?? null;
  const seasonId = requestedSeason ?? currentSeasonId;
  const stats =
    seasonId !== null && seasons.includes(seasonId)
      ? computeSeasonStats(view, seasonId, listPlayers(conn), listMatchesByDate(conn))
      : null;
  return { seasons, currentSeasonId, stats };
}

/**
 * 事实统计（出勤/胜场/最佳组合/最接近/碾压/连胜/冷门）：算法镜像
 * computeWeeklyFacts（两模型复用同一份逻辑的赛季版），差异仅为
 * 搭档门槛（季内 ≥5 场）与比赛集合跨周。冷门概率由调用方注入赛前
 * Estimated 概率，不接触周末 Final 之后的状态。
 */
function computeSeasonFacts(
  seasonMatches: readonly MatchWithNames[],
  nameMap: ReadonlyMap<number, string>,
  winnerProbByMatchId: ReadonlyMap<number, number>
): {
  attendance: WeeklyPlayerStat[];
  winKing: WeeklyPlayerStat[];
  bestPair: BestPair | null;
  fun: WeeklyFun;
} {
  const stats = new Map<
    number,
    { playerId: number; name: string; matches: number; wins: number; losses: number }
  >();
  const ensure = (id: number) => {
    if (!stats.has(id)) {
      stats.set(id, {
        playerId: id,
        name: nameMap.get(id) ?? "?",
        matches: 0,
        wins: 0,
        losses: 0,
      });
    }
    return stats.get(id)!;
  };

  const pairStats = new Map<
    string,
    { key: string; names: string[]; wins: number; total: number }
  >();

  for (const m of seasonMatches) {
    const aWon = m.scoreA > m.scoreB;

    for (const id of [m.pa1, m.pa2]) {
      const s = ensure(id);
      s.matches++;
      if (aWon) s.wins++;
      else s.losses++;
    }
    for (const id of [m.pb1, m.pb2]) {
      const s = ensure(id);
      s.matches++;
      if (!aWon) s.wins++;
      else s.losses++;
    }

    const teamA = [m.pa1, m.pa2].sort((a, b) => a - b);
    const teamB = [m.pb1, m.pb2].sort((a, b) => a - b);
    const keyA = teamA.join(",");
    const keyB = teamB.join(",");

    for (const [key, ids, won] of [
      [keyA, teamA, aWon],
      [keyB, teamB, !aWon],
    ] as const) {
      if (!pairStats.has(key)) {
        pairStats.set(key, {
          key,
          names: ids.map((id) => nameMap.get(id) ?? "?"),
          wins: 0,
          total: 0,
        });
      }
      const p = pairStats.get(key)!;
      p.total++;
      if (won) p.wins++;
    }
  }

  const allValues = Array.from(stats.values());
  const attendance = [...allValues]
    .filter((s) => s.matches > 0)
    .sort((a, b) => b.matches - a.matches || b.wins - a.wins);
  const winKing = [...allValues]
    .filter((s) => s.wins > 0)
    .sort((a, b) => b.wins - a.wins || b.matches - a.matches);

  let bestPair: BestPair | null = null;
  for (const p of pairStats.values()) {
    if (p.total < SEASON_PAIR_MIN_MATCHES) continue;
    const winRate = p.wins / p.total;
    if (!bestPair || winRate > bestPair.winRate) {
      bestPair = {
        playerA: p.names[0],
        playerB: p.names[1],
        wins: p.wins,
        total: p.total,
        winRate,
      };
    }
  }

  const fun = computeSeasonFun(seasonMatches, nameMap, winnerProbByMatchId);
  return { attendance, winKing, bestPair, fun };
}

function toFunMatch(m: MatchWithNames): FunMatch {
  return {
    date: m.playedAt,
    teamA: [m.pa1Name, m.pa2Name],
    teamB: [m.pb1Name, m.pb2Name],
    scoreA: m.scoreA,
    scoreB: m.scoreB,
  };
}

/**
 * 赛季趣闻：最接近/碾压口径与周报一致（同分差时更高胜方分/更低负方分，
 * 再同取最早一场）；连胜王按 played_at 序跨周累计最长连胜（负场清零，
 * ≥2 起评，同长取先到）；冷门取季内赛前胜率最低且 <50% 的一场。
 */
function computeSeasonFun(
  seasonMatches: readonly MatchWithNames[],
  nameMap: ReadonlyMap<number, string>,
  winnerProbByMatchId: ReadonlyMap<number, number>
): WeeklyFun {
  if (seasonMatches.length === 0) {
    return { closest: null, blowout: null, streakKing: null, upset: null };
  }

  let closestM = seasonMatches[0];
  let blowoutM = seasonMatches[0];
  for (const m of seasonMatches) {
    const diff = Math.abs(m.scoreA - m.scoreB);
    const cDiff = Math.abs(closestM.scoreA - closestM.scoreB);
    if (
      diff < cDiff ||
      (diff === cDiff &&
        Math.max(m.scoreA, m.scoreB) > Math.max(closestM.scoreA, closestM.scoreB))
    ) {
      closestM = m;
    }
    const bDiff = Math.abs(blowoutM.scoreA - blowoutM.scoreB);
    if (
      diff > bDiff ||
      (diff === bDiff &&
        Math.min(m.scoreA, m.scoreB) < Math.min(blowoutM.scoreA, blowoutM.scoreB))
    ) {
      blowoutM = m;
    }
  }

  // 连胜跨周不断：只在输球时清零，与周报周内算法同逻辑、比赛集合更大。
  const streaks = new Map<number, number>();
  let streakKing: WeeklyFun["streakKing"] = null;
  for (const m of seasonMatches) {
    const aWon = m.scoreA > m.scoreB;
    const winners = aWon ? [m.pa1, m.pa2] : [m.pb1, m.pb2];
    const losers = aWon ? [m.pb1, m.pb2] : [m.pa1, m.pa2];
    for (const id of losers) streaks.set(id, 0);
    for (const id of winners) {
      const s = (streaks.get(id) ?? 0) + 1;
      streaks.set(id, s);
      if (s >= 2 && (streakKing === null || s > streakKing.streak)) {
        streakKing = { playerId: id, name: nameMap.get(id) ?? "?", streak: s };
      }
    }
  }

  let upset: UpsetMatch | null = null;
  let bestProb = 0.5;
  for (const m of seasonMatches) {
    const winnerProb = winnerProbByMatchId.get(m.id);
    if (winnerProb === undefined) continue;
    if (winnerProb < bestProb) {
      bestProb = winnerProb;
      upset = { ...toFunMatch(m), winnerWinProb: winnerProb };
    }
  }

  return {
    closest: toFunMatch(closestM),
    blowout: toFunMatch(blowoutM),
    streakKing,
    upset,
  };
}
