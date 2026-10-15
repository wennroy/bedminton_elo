import { canonicalMatchKey, type ScheduledMatch } from "./types";

/**
 * balanced-v1 目标函数：候选方案评分、排序元组与独立全量重算。
 *
 * 每场候选 m：p_m = winProbability(a1,a2,b1,b2)（必须有限且在 [0,1]，
 * 否则整个生成失败）；d_m = 2*|p_m - 0.5| ∈ [0,1]。
 *
 * 分量（只随候选方案变化，已打历史只进 H 的分子、不进任何分母）：
 * - B = mean(d_m²)                          平均实力差
 * - P = Σ_pair C(newPartnerCount,2)/(2M)    本批搭档重复（每场 2 对搭档）
 * - O = Σ_pair C(newOpponentCount,2)/(4M)   本批个人对手重复（每人每场 2 个对手）
 * - E = Σ_match C(newExactMatchCount,2)/M   本批完整对阵重复（canonicalMatchKey）
 * - H = Σ_match playedExactCount*newExact/M 已打完整对阵复现
 * - J = wB*B + wP*P + wO*O + wE*E + wH*H
 *
 * 默认权重 wB=4 wP=2 wO=0.5 wE=2 wH=3（avoidPlayedRematches=false 时 wH=0，
 * checkbox 只开关 H，其余分量始终生效）。
 *
 * 排序元组（字典序，只做排序依据不是硬约束）：
 * (明显失衡场数, 总超阈值幅度, J, 规范化方案键)。
 * 明显失衡初值：一方预测胜率超过 70%，即 d_m > 0.4；
 * 总超阈值幅度 = Σ max(0, d_m - 0.4)。
 */

export interface ObjectiveWeights {
  wB: number;
  wP: number;
  wO: number;
  wE: number;
  wH: number;
}

export const DEFAULT_WEIGHTS: ObjectiveWeights = {
  wB: 4,
  wP: 2,
  wO: 0.5,
  wE: 2,
  wH: 3,
};

/** 由 avoidPlayedRematches 派生权重：checkbox 只开关 H 分量 */
export function weightsFor(input: {
  avoidPlayedRematches: boolean;
}): ObjectiveWeights {
  return { ...DEFAULT_WEIGHTS, wH: input.avoidPlayedRematches ? DEFAULT_WEIGHTS.wH : 0 };
}

/** 失衡阈值：d > IMBALANCE_THRESHOLD 即一方预测胜率超过 70% */
export const IMBALANCE_THRESHOLD = 0.4;

/** C(n,2)，n 为非负整数；输入保证整数，结果恒为精确整数 */
export function comb2(n: number): number {
  return (n * (n - 1)) / 2;
}

/** 两人无序对规范化键（队内/对手对共用） */
export function pairKey(x: string, y: string): string {
  return x < y ? JSON.stringify([x, y]) : JSON.stringify([y, x]);
}

export interface MatchEval {
  /** canonicalMatchKey 缓存键 */
  key: string;
  /** 原始概率（校验后在 [0,1]） */
  p: number;
  /** 2*|p-0.5| */
  d: number;
  d2: number;
  imbalanced: boolean;
  excess: number;
}

export type WinProbabilityFn = (
  a1: string,
  a2: string,
  b1: string,
  b2: string
) => number;

/**
 * 取概率并校验：必须有限且在 [0,1]，否则整次生成失败（带阵容信息的 Error）。
 */
export function evalProbability(
  a1: string,
  a2: string,
  b1: string,
  b2: string,
  winProbability: WinProbabilityFn
): { p: number; d: number; d2: number } {
  const p = winProbability(a1, a2, b1, b2);
  if (typeof p !== "number" || !Number.isFinite(p) || p < 0 || p > 1) {
    throw new Error(
      `winProbability 返回无效概率: match ${JSON.stringify([a1, a2, b1, b2])} -> ${String(p)}`
    );
  }
  const d = 2 * Math.abs(p - 0.5);
  return { p, d, d2: d * d };
}

/**
 * 单场评价：带懒缓存（按 canonicalMatchKey）。同一对阵的
 * d/d2/imbalanced/excess 只依赖 |p-0.5|，与队别朝向无关，可安全复用。
 */
export function evaluateMatch(
  match: ScheduledMatch,
  winProbability: WinProbabilityFn,
  cache?: Map<string, MatchEval>
): MatchEval {
  const key = canonicalMatchKey(match.a1, match.a2, match.b1, match.b2);
  const cached = cache?.get(key);
  if (cached) return cached;
  const { p, d, d2 } = evalProbability(
    match.a1,
    match.a2,
    match.b1,
    match.b2,
    winProbability
  );
  const eval_: MatchEval = {
    key,
    p,
    d,
    d2,
    imbalanced: d > IMBALANCE_THRESHOLD,
    excess: Math.max(0, d - IMBALANCE_THRESHOLD),
  };
  cache?.set(key, eval_);
  return eval_;
}

/** 一场的两对搭档键 */
export function partnerKeysOf(match: ScheduledMatch): [string, string] {
  return [
    pairKey(match.a1, match.a2),
    pairKey(match.b1, match.b2),
  ];
}

/** 一场的四对对手键（A 队每人对 B 队每人） */
export function opponentKeysOf(match: ScheduledMatch): [string, string, string, string] {
  return [
    pairKey(match.a1, match.b1),
    pairKey(match.a1, match.b2),
    pairKey(match.a2, match.b1),
    pairKey(match.a2, match.b2),
  ];
}

export interface ObjectiveBreakdown {
  B: number;
  P: number;
  O: number;
  E: number;
  H: number;
  J: number;
  imbalancedMatchCount: number;
  totalImbalanceExcess: number;
}

/**
 * 独立全量重算：与搜索内增量状态没有任何共享代码，供返回前校验和
 * 增量一致性测试做逐位对照。迭代顺序固定（场次序 + Map 插入序由
 * 调用方保证一致），结果确定。
 */
export function computeObjectiveBreakdown(
  schedule: ScheduledMatch[],
  opts: {
    winProbability: WinProbabilityFn;
    playedMatchCounts: Record<string, number>;
    weights: ObjectiveWeights;
  }
): ObjectiveBreakdown {
  const { winProbability, playedMatchCounts, weights } = opts;
  const M = schedule.length;
  const cache = new Map<string, MatchEval>();

  let sumD2 = 0;
  let imbalancedMatchCount = 0;
  let totalImbalanceExcess = 0;
  const partner = new Map<string, number>();
  const opponent = new Map<string, number>();
  const exact = new Map<string, number>();

  for (const match of schedule) {
    const e = evaluateMatch(match, winProbability, cache);
    sumD2 += e.d2;
    if (e.imbalanced) imbalancedMatchCount++;
    totalImbalanceExcess += e.excess;

    for (const pk of partnerKeysOf(match)) {
      partner.set(pk, (partner.get(pk) ?? 0) + 1);
    }
    for (const ok of opponentKeysOf(match)) {
      opponent.set(ok, (opponent.get(ok) ?? 0) + 1);
    }
    exact.set(e.key, (exact.get(e.key) ?? 0) + 1);
  }

  let partnerC2 = 0;
  for (const c of partner.values()) partnerC2 += comb2(c);
  let opponentC2 = 0;
  for (const c of opponent.values()) opponentC2 += comb2(c);
  let exactC2 = 0;
  let exactPlayed = 0;
  for (const [key, c] of exact) {
    exactC2 += comb2(c);
    const played = playedMatchCounts[key] ?? 0;
    exactPlayed += played * c;
  }

  return assembleBreakdown({
    M,
    sumD2,
    imbalancedMatchCount,
    totalImbalanceExcess,
    partnerC2,
    opponentC2,
    exactC2,
    exactPlayed,
    weights,
  });
}

/** 由聚合量组装 J（搜索增量状态与全量重算共用同一组装公式，保证逐位一致） */
export function assembleBreakdown(agg: {
  M: number;
  sumD2: number;
  imbalancedMatchCount: number;
  totalImbalanceExcess: number;
  partnerC2: number;
  opponentC2: number;
  exactC2: number;
  exactPlayed: number;
  weights: ObjectiveWeights;
}): ObjectiveBreakdown {
  const {
    M,
    sumD2,
    imbalancedMatchCount,
    totalImbalanceExcess,
    partnerC2,
    opponentC2,
    exactC2,
    exactPlayed,
    weights,
  } = agg;
  if (M <= 0) {
    throw new Error(`assembleBreakdown: 场数 M 必须为正，实际 ${M}`);
  }
  const B = sumD2 / M;
  const P = partnerC2 / (2 * M);
  const O = opponentC2 / (4 * M);
  const E = exactC2 / M;
  const H = exactPlayed / M;
  const J = weights.wB * B + weights.wP * P + weights.wO * O + weights.wE * E + weights.wH * H;
  return { B, P, O, E, H, J, imbalancedMatchCount, totalImbalanceExcess };
}

/** 方案排序元组：字典序比较，key 为最终平局决胜 */
export interface RankTuple {
  imbalancedMatchCount: number;
  totalImbalanceExcess: number;
  J: number;
  /** 规范化方案键（scheduleKey），只在前三项全等时参与比较 */
  key: string;
}

export function tupleFromBreakdown(
  b: ObjectiveBreakdown,
  key = ""
): RankTuple {
  return {
    imbalancedMatchCount: b.imbalancedMatchCount,
    totalImbalanceExcess: b.totalImbalanceExcess,
    J: b.J,
    key,
  };
}

/** 字典序比较：负 = a 优于 b；前三项全等比规范化方案键 */
export function compareRankTuples(a: RankTuple, b: RankTuple): number {
  if (a.imbalancedMatchCount !== b.imbalancedMatchCount) {
    return a.imbalancedMatchCount - b.imbalancedMatchCount;
  }
  if (a.totalImbalanceExcess !== b.totalImbalanceExcess) {
    return a.totalImbalanceExcess - b.totalImbalanceExcess;
  }
  if (a.J !== b.J) return a.J - b.J;
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0;
}

/** 规范化方案键：各场 canonicalMatchKey 按场次顺序 JSON 拼接 */
export function scheduleKey(schedule: ScheduledMatch[]): string {
  return JSON.stringify(
    schedule.map((m) => canonicalMatchKey(m.a1, m.a2, m.b1, m.b2))
  );
}
