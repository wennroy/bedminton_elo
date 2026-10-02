import { createHash } from "node:crypto";
import type Database from "better-sqlite3";
import {
  listPlayers,
  listMatchesByDate,
  type MatchWithNames,
} from "@/lib/repo";
import { recomputeElos, computeMatchWinProbs, INITIAL_RATING, type Match as EloMatch } from "@/lib/elo";
import { readRatingConfig } from "@/lib/rating-config";
import { loadRatingView, type LoadRatingViewResult } from "@/lib/rating-view";
import {
  nextWeekStart,
  shanghaiLocalDateFromInstant,
  shanghaiMidnightIso,
  weekStart as ratingWeekStart,
} from "@/lib/ratings/calendar";
import type { RatingModel, RatingState } from "@/lib/ratings/types";

export type { MatchWithNames };

export interface WeeklyPlayerStat {
  playerId: number;
  name: string;
  matches: number;
  wins: number;
  losses: number;
}

export interface EloChangeStat {
  playerId: number;
  name: string;
  eloStart: number;
  eloEnd: number;
  change: number;
}

export interface BestPair {
  playerA: string;
  playerB: string;
  wins: number;
  total: number;
  winRate: number;
}

export interface FunMatch {
  date: string;
  teamA: [string, string];
  teamB: [string, string];
  scoreA: number;
  scoreB: number;
}

export interface UpsetMatch extends FunMatch {
  winnerWinProb: number;
}

export interface StreakKing {
  playerId: number;
  name: string;
  streak: number;
}

export interface WeeklyFun {
  closest: FunMatch | null;
  blowout: FunMatch | null;
  streakKing: StreakKing | null;
  upset: UpsetMatch | null;
}

/** glicko2 周区段内单人的变化汇总（数据源为投影 weekSegments，不自行重放）。 */
export interface WeeklyRatingPlayerReport {
  playerId: number;
  name: string;
  matchesPlayed: number;
  /** 本周首次上场前预估（取整）。 */
  startR: number;
  /** 末次上场后预估（取整）。 */
  endEstimatedR: number;
  estimatedChange: number;
  /** 周结算校准 final - estimatedEnd（取整）；未结算段为 0。 */
  correction: number;
  /** 周 Final 展示值（取整）；未结算（Estimated）段为 null。 */
  finalR: number | null;
}

/** glicko2 周区段行：跨季周被拆成两段分列。 */
export interface WeeklyRatingSegmentReport {
  segmentId: string;
  weekStart: string;
  seasonId: string | null;
  start: string;
  end: string;
  h: number;
  /** 已结束区段为 final（含周校准），进行中区段为 estimated。 */
  status: "estimated" | "final";
  players: WeeklyRatingPlayerReport[];
  /** 周校准 correction（取整后）。 */
  correction: Record<string, number>;
}

/** 季界软重置：只挂在触发重置的那一段上。 */
export interface WeeklyRatingReset {
  segmentId: string;
  at: string;
  seasonId: string;
  changes: Array<{
    playerId: number;
    name: string;
    beforeR: number;
    afterR: number;
    delta: number;
  }>;
}

/**
 * glicko2 模型化周报：独立于 eloChanges（后者保持旧 ELO 语义），
 * 不塞入 glicko 数值。数据源是 loadRatingView/projectRatingView 的
 * weekSegments，不自行重放。
 */
export interface WeeklyRatingReport {
  model: "glicko2";
  version: string;
  inputHash: string | null;
  asOf: string;
  freshness: "fresh" | "stale";
  nextBoundary: string;
  /** asOf 所属区段 ID（边界状态，供内容指纹使用）。 */
  segmentId: string;
  segments: WeeklyRatingSegmentReport[];
  resets: WeeklyRatingReset[];
}

export interface WeeklyStats {
  weekStart: string;
  weekEnd: string;
  weekNumber: number;
  attendance: WeeklyPlayerStat[];
  winKing: WeeklyPlayerStat[];
  eloChanges: EloChangeStat[];
  bestPair: BestPair | null;
  fun: WeeklyFun;
  /**
   * glicko2 分支的模型化评分报告；legacy 分支不设此字段，
   * 序列化形状与旧格式完全一致（旧内容指纹不因发版失效）。
   */
  ratingReport?: WeeklyRatingReport;
}

/** glicko2 评分不可用的明确错误：OG 路由据此返回 409 而非伪造数据。 */
export class WeeklyRatingUnavailableError extends Error {
  readonly reason: string;

  constructor(reason: string) {
    super(`glicko2 rating unavailable: ${reason}`);
    this.name = "WeeklyRatingUnavailableError";
    this.reason = reason;
  }
}

/** buildWeeklyStats 的模型上下文：显式 rating 可选，缺省回 activeModel。 */
export interface BuildWeeklyStatsOptions {
  rating?: string;
  /** 可注入的评分时点（ISO instant），默认当前时间；一次调用只捕获一次。 */
  asOf?: string;
  db?: Database.Database;
}

function toEloMatch(m: MatchWithNames): EloMatch {
  return {
    date: m.playedAt,
    a1: String(m.pa1),
    a2: String(m.pa2),
    b1: String(m.pb1),
    b2: String(m.pb2),
    scoreA: m.scoreA,
    scoreB: m.scoreB,
  };
}

export function getWeekRange(dateStr: string): {
  weekStart: string;
  weekEnd: string;
  weekNumber: number;
} {
  const [year, month, day] = dateStr.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  const dayOfWeek = date.getDay();
  const mondayOffset = dayOfWeek === 0 ? -6 : 1 - dayOfWeek;
  const monday = new Date(date);
  monday.setDate(date.getDate() + mondayOffset);
  const sunday = new Date(monday);
  sunday.setDate(monday.getDate() + 6);

  const fmt = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
      d.getDate()
    ).padStart(2, "0")}`;

  const startOfYear = new Date(monday.getFullYear(), 0, 1);
  const diffMs = monday.getTime() - startOfYear.getTime();
  const weekNumber = Math.floor(diffMs / (7 * 24 * 60 * 60 * 1000)) + 1;

  return { weekStart: fmt(monday), weekEnd: fmt(sunday), weekNumber };
}

export function listWeekStarts(): string[] {
  const matches = listMatchesByDate();
  if (matches.length === 0) return [];
  const first = matches[0].playedAt;
  const last = matches[matches.length - 1].playedAt;
  const { weekStart: firstWeek } = getWeekRange(first);
  const { weekStart: lastWeek } = getWeekRange(last);

  const starts: string[] = [];
  let current = firstWeek;
  while (current <= lastWeek) {
    starts.push(current);
    const [y, m, d] = current.split("-").map(Number);
    const next = new Date(y, m - 1, d + 7);
    current = `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(
      2,
      "0"
    )}-${String(next.getDate()).padStart(2, "0")}`;
  }
  return starts;
}

/** 显式 "glicko2"|"legacy" 优先；非法值/缺省回 activeModel（无配置默认 legacy）。 */
function resolveWeeklyModel(
  requested: string | undefined,
  db?: Database.Database
): RatingModel {
  if (requested === "glicko2" || requested === "legacy") return requested;
  return readRatingConfig(db)?.activeModel ?? "legacy";
}

export function buildWeeklyStats(
  weekStart: string,
  options: BuildWeeklyStatsOptions = {}
): WeeklyStats {
  const conn = options.db;
  const players = listPlayers(conn);
  const allMatches = listMatchesByDate(conn);
  const model = resolveWeeklyModel(options.rating, conn);
  if (model === "glicko2") {
    return computeWeeklyStats(weekStart, players, allMatches, {
      ...options,
      rating: "glicko2",
    });
  }
  // legacy 裸调用：与基线行为完全一致（无 ratingReport 字段）。
  return computeWeeklyStats(weekStart, players, allMatches);
}

/**
 * 周报内容指纹上下文：覆盖模型、参数版本、输入指纹、freshness 与
 * 时钟/区段边界状态（asOf 的上海日期、所属区段、下一边界）。
 * 不含 asOf 原文毫秒值，避免无意义失效。
 */
export interface WeeklyDataVersionContext {
  model: "glicko2";
  version: string;
  inputHash: string | null;
  freshness: "fresh" | "stale";
  asOfDate: string;
  asOfSegmentId: string;
  nextBoundary: string;
}

/**
 * 从 stats.ratingReport 构造指纹上下文；legacy（无报告）返回 undefined。
 * inputHash 是全库输入指纹：与本周无关的撤回/改名/合并也会使该周 ETag
 * 失效——重放有路径依赖，跨周影响真实存在（下游当前分会变），故取保守
 * 失效：宁可多失效一次，不错过该更新的缓存（e2e 场景 8 锁定不变时仍命中）。
 */
export function weeklyDataVersionContext(
  stats: WeeklyStats
): WeeklyDataVersionContext | undefined {
  const report = stats.ratingReport;
  if (report === undefined || report === null) return undefined;
  return {
    model: "glicko2",
    version: report.version,
    inputHash: report.inputHash,
    freshness: report.freshness,
    asOfDate: shanghaiLocalDateFromInstant(report.asOf),
    asOfSegmentId: report.segmentId,
    nextBoundary: report.nextBoundary,
  };
}

// 周报分享图的内容指纹:WeeklyStats 是分享图画面的全部决定因素
// (比赛/比分/球员名/ELO 都在里面),DB 一变指纹就变。用作 OG 路由的 ETag。
// 传入 context（glicko2）时把模型/版本/输入指纹/freshness/边界状态一并
// 指纹：跨周/季界即使 DB 未变也必须失效；legacy 调用方不传 context 时
// 指纹与旧版逐位一致。
export function weeklyDataVersion(
  stats: WeeklyStats,
  context?: WeeklyDataVersionContext
): string {
  if (context === undefined) {
    return createHash("sha1").update(JSON.stringify(stats)).digest("hex").slice(0, 16);
  }
  // asOf 原文（含毫秒）不入指纹：归一到上下文里的上海日期桶，
  // 同一边界内不同请求时点指纹稳定；边界变化由 context 字段覆盖。
  const report = stats.ratingReport;
  const normalizedStats =
    report === undefined
      ? stats
      : { ...stats, ratingReport: { ...report, asOf: context.asOfDate } };
  return createHash("sha1")
    .update(JSON.stringify({ stats: normalizedStats, context }))
    .digest("hex")
    .slice(0, 16);
}

// 分享图的设计指纹:版式/配色变更时递增。ETag 只指纹数据时,数据未变的周
// 在换版式后仍会对旧缓存 304,客户端永远显示旧设计(v1.5.1 踩过的坑)。
// 放在 lib 是因为 route 文件只允许导出 HTTP 方法,build 期类型检查会拦。
// d3: glicko2 模式新增「评分变化」版块（段级 Estimated/Final + 重置），
// 同阵容行数上限与长姓名截断；legacy 版式不变。
// d4: 评分版块标题防挤压（版本串只取首段）、Final/当前改前缀序（修
// 「校准 -1」与 Final 值粘连误读）、跨季双段周行数收敛与间距压缩（修
// 内容溢出与绝对定位页脚重叠、趣闻被裁）；legacy 版式仍不变。
export const OG_DESIGN_VERSION = "d4";

/** 上海周一界的本地日期加减（纯 UTC 数学，不依赖主机 TZ）。 */
function addDaysLocal(localDate: string, days: number): string {
  const [year, month, day] = localDate.split("-").map(Number);
  const shifted = new Date(Date.UTC(year, month - 1, day) + days * 86400000);
  return [
    shifted.getUTCFullYear(),
    String(shifted.getUTCMonth() + 1).padStart(2, "0"),
    String(shifted.getUTCDate()).padStart(2, "0"),
  ].join("-");
}

/** 上海周一界的周序号：年内第几个周一（1 起），与旧口径在无 DST 时一致。 */
function shanghaiWeekNumber(weekMonday: string): number {
  const [year, month, day] = weekMonday.split("-").map(Number);
  return (
    Math.floor(
      (Date.UTC(year, month - 1, day) - Date.UTC(year, 0, 1)) /
        (7 * 86400000)
    ) + 1
  );
}

export function computeWeeklyStats(
  weekStart: string,
  players: { id: number; name: string }[],
  allMatches: MatchWithNames[],
  options?: BuildWeeklyStatsOptions
): WeeklyStats {
  if (options?.rating === "glicko2") {
    return computeWeeklyStatsGlicko2(weekStart, players, allMatches, options);
  }
  return computeWeeklyStatsLegacy(weekStart, players, allMatches);
}

/** legacy 分支：与基线逐位一致（主机 TZ 周界 + ELO 冷门 + 无 ratingReport）。 */
function computeWeeklyStatsLegacy(
  weekStart: string,
  players: { id: number; name: string }[],
  allMatches: MatchWithNames[]
): WeeklyStats {
  const { weekEnd, weekNumber } = getWeekRange(weekStart);
  const nameMap = new Map(players.map((p) => [p.id, p.name]));

  const weekMatches = allMatches.filter(
    (m) => m.playedAt >= weekStart && m.playedAt <= weekEnd
  );

  const winnerProbByMatchId = legacyWinnerProbabilities(
    allMatches,
    weekStart,
    weekEnd
  );
  const facts = computeWeeklyFacts(weekMatches, nameMap, winnerProbByMatchId);
  const eloChanges = computeEloChanges(weekStart, weekEnd, nameMap, allMatches);

  return {
    weekStart,
    weekEnd,
    weekNumber,
    attendance: facts.attendance,
    winKing: facts.winKing,
    eloChanges,
    bestPair: facts.bestPair,
    fun: facts.fun,
  };
}

/**
 * glicko2 分支：周界以 ratings/calendar 为准（上海周一界），能查看最后
 * 一场之后的空周（投影的当前区段）与季界周（跨季两段 + 重置分列）；
 * 冷门从 matchId 对应的赛前 Estimated 概率求出，不接触周末 Final 之后的状态。
 */
function computeWeeklyStatsGlicko2(
  weekStart: string,
  players: { id: number; name: string }[],
  allMatches: MatchWithNames[],
  options: BuildWeeklyStatsOptions
): WeeklyStats {
  const asOf = options.asOf ?? new Date().toISOString();
  const view = loadRatingView({ rating: "glicko2", asOf, db: options.db });
  if (view.model !== "glicko2") {
    throw new Error("unreachable: loadRatingView returned a different model");
  }
  if (view.freshness === "unavailable") {
    // 不伪造胜率、不静默退回 legacy：明确向上抛错，由路由返回非 2xx。
    throw new WeeklyRatingUnavailableError(view.reason);
  }

  const monday = ratingWeekStart(weekStart);
  const weekEnd = addDaysLocal(monday, 6);
  const weekNumber = shanghaiWeekNumber(monday);
  const nameMap = new Map(players.map((p) => [p.id, p.name]));
  const weekMatches = allMatches.filter(
    (m) => m.playedAt >= monday && m.playedAt <= weekEnd
  );

  // 冷门：赛前 Estimated 概率（matchEstimates 按 matchId 取），引擎未计分
  // 的记录（坏数据/未来/未知球员）不参与。
  const winnerProbByMatchId = new Map<number, number>();
  for (const m of weekMatches) {
    const estimate = view.view.matchEstimatesById[String(m.id)];
    if (estimate === undefined) continue;
    winnerProbByMatchId.set(
      m.id,
      m.scoreA > m.scoreB ? estimate.preWinA : 1 - estimate.preWinA
    );
  }

  const facts = computeWeeklyFacts(weekMatches, nameMap, winnerProbByMatchId);
  const eloChanges = computeEloChanges(monday, weekEnd, nameMap, allMatches);
  const report = buildWeeklyRatingReport(
    view,
    nameMap,
    shanghaiMidnightIso(monday),
    shanghaiMidnightIso(nextWeekStart(monday))
  );

  return {
    weekStart: monday,
    weekEnd,
    weekNumber,
    attendance: facts.attendance,
    winKing: facts.winKing,
    eloChanges,
    bestPair: facts.bestPair,
    fun: facts.fun,
    ratingReport: report,
  };
}

type ReadyOrStaleRatingView = Extract<
  LoadRatingViewResult,
  { model: "glicko2"; freshness: "fresh" | "stale" }
>;

/** 从投影 weekSegments 构造模型化报告：只消费投影，不自行重放。 */
function buildWeeklyRatingReport(
  view: ReadyOrStaleRatingView,
  nameMap: Map<number, string>,
  weekStartInstant: string,
  weekEndExclusiveInstant: string
): WeeklyRatingReport {
  const segments: WeeklyRatingSegmentReport[] = [];
  const resets: WeeklyRatingReset[] = [];

  for (const segment of view.view.weekSegments) {
    // 只取与本周 [weekStart, weekEndExclusive) 有交集的区段。
    if (
      !(segment.start < weekEndExclusiveInstant && segment.end > weekStartInstant)
    ) {
      continue;
    }
    const status: "estimated" | "final" =
      segment.segmentId === view.currentSegmentId ? "estimated" : "final";

    const firstBefore = new Map<number, RatingState>();
    const lastAfter = new Map<number, RatingState>();
    const matchCount = new Map<number, number>();
    for (const event of segment.matches) {
      for (const change of event.changes) {
        if (!firstBefore.has(change.playerId)) {
          firstBefore.set(change.playerId, change.before);
        }
        lastAfter.set(change.playerId, change.after);
        matchCount.set(change.playerId, (matchCount.get(change.playerId) ?? 0) + 1);
      }
    }

    const playerReports: WeeklyRatingPlayerReport[] = [];
    for (const playerId of [...firstBefore.keys()].sort((a, b) => a - b)) {
      const before = firstBefore.get(playerId)!;
      const after = lastAfter.get(playerId)!;
      const correctionRaw = segment.correction[String(playerId)] ?? 0;
      playerReports.push({
        playerId,
        name: nameMap.get(playerId) ?? "?",
        matchesPlayed: matchCount.get(playerId) ?? 0,
        startR: Math.round(before.r),
        endEstimatedR: Math.round(after.r),
        estimatedChange: Math.round(after.r) - Math.round(before.r),
        correction: Math.round(correctionRaw),
        finalR:
          status === "final" ? Math.round(after.r + correctionRaw) : null,
      });
    }

    const correction: Record<string, number> = {};
    for (const [playerId, value] of Object.entries(segment.correction)) {
      correction[playerId] = Math.round(value);
    }

    segments.push({
      segmentId: segment.segmentId,
      weekStart: segment.weekStart,
      seasonId: segment.seasonId,
      start: segment.start,
      end: segment.end,
      h: segment.h,
      status,
      players: playerReports,
      correction,
    });

    if (segment.reset !== null) {
      resets.push({
        segmentId: segment.segmentId,
        at: segment.reset.at,
        seasonId: segment.reset.seasonId,
        changes: segment.reset.changes.map((change) => ({
          playerId: change.playerId,
          name: nameMap.get(change.playerId) ?? "?",
          beforeR: Math.round(change.before.r),
          afterR: Math.round(change.after.r),
          delta: Math.round(change.delta),
        })),
      });
    }
  }

  return {
    model: "glicko2",
    version: view.version,
    inputHash: view.inputHash,
    asOf: view.asOf,
    freshness: view.freshness,
    nextBoundary: view.nextBoundary,
    segmentId: view.currentSegmentId,
    segments,
    resets,
  };
}

/** legacy 冷门概率：ELO 重放，逐场胜方赛前胜率（与基线同口径）。 */
function legacyWinnerProbabilities(
  allMatches: MatchWithNames[],
  weekStart: string,
  weekEnd: string
): Map<number, number> {
  const probs = computeMatchWinProbs(allMatches.map(toEloMatch));
  const map = new Map<number, number>();
  for (let i = 0; i < allMatches.length; i++) {
    const m = allMatches[i];
    if (m.playedAt < weekStart || m.playedAt > weekEnd) continue;
    map.set(m.id, m.scoreA > m.scoreB ? probs[i] : 1 - probs[i]);
  }
  return map;
}

/**
 * 事实统计（出勤/胜场/最佳组合/最接近/碾压/连胜/冷门）：
 * 两模型复用同一份逻辑，冷门概率由调用方按模型注入。
 */
function computeWeeklyFacts(
  weekMatches: MatchWithNames[],
  nameMap: Map<number, string>,
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

  for (const m of weekMatches) {
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
    if (p.total < 3) continue;
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

  const fun = computeWeeklyFun(weekMatches, nameMap, winnerProbByMatchId);
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

function computeWeeklyFun(
  weekMatches: MatchWithNames[],
  nameMap: Map<number, string>,
  winnerProbByMatchId: ReadonlyMap<number, number>
): WeeklyFun {
  if (weekMatches.length === 0) {
    return { closest: null, blowout: null, streakKing: null, upset: null };
  }

  // closest: smallest diff; tie -> higher winner score; tie -> earliest.
  // blowout: largest diff; tie -> lower loser score; tie -> earliest.
  // weekMatches are in listMatchesByDate order, so "keep current" means earliest.
  let closestM = weekMatches[0];
  let blowoutM = weekMatches[0];
  for (const m of weekMatches) {
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

  // streak king: longest in-week win streak per player, in chronological
  // order; strict > keeps the first player to reach the max. < 2 -> null.
  const streaks = new Map<number, number>();
  let streakKing: StreakKing | null = null;
  for (const m of weekMatches) {
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

  // upset: lowest pre-match winner win prob among week matches;
  // >= 50% is not an upset. Strict < keeps the earliest on ties.
  // glicko2 分支的上游注入赛前 Estimated 概率，legacy 注入 ELO 概率。
  let upset: UpsetMatch | null = null;
  let bestProb = 0.5;
  for (const m of weekMatches) {
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

function computeEloChanges(
  weekStart: string,
  weekEnd: string,
  nameMap: Map<number, string>,
  allMatches: MatchWithNames[]
): EloChangeStat[] {
  const { snapshots } = recomputeElos(allMatches.map(toEloMatch));

  const lastSnapshotBefore = new Map<string, number>();
  const snapshotOnOrBeforeEnd = new Map<string, number>();

  for (const s of snapshots) {
    if (s.date < weekStart) {
      lastSnapshotBefore.set(s.playerId, s.elo);
    }
    if (s.date <= weekEnd) {
      snapshotOnOrBeforeEnd.set(s.playerId, s.elo);
    }
  }

  const playerIds = new Set<string>();
  for (const m of allMatches) {
    if (m.playedAt >= weekStart && m.playedAt <= weekEnd) {
      playerIds.add(String(m.pa1));
      playerIds.add(String(m.pa2));
      playerIds.add(String(m.pb1));
      playerIds.add(String(m.pb2));
    }
  }

  const changes: EloChangeStat[] = [];
  for (const id of playerIds) {
    const numId = Number(id);
    const eloStart = lastSnapshotBefore.get(id) ?? INITIAL_RATING;
    const eloEnd = snapshotOnOrBeforeEnd.get(id) ?? eloStart;
    changes.push({
      playerId: numId,
      name: nameMap.get(numId) ?? "?",
      eloStart: Math.round(eloStart),
      eloEnd: Math.round(eloEnd),
      change: Math.round(eloEnd - eloStart),
    });
  }

  return changes.sort((a, b) => b.change - a.change || a.playerId - b.playerId);
}
