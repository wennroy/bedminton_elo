import {
  ALGORITHM_VERSION,
  OBJECTIVE_VERSION,
  canonicalMatchKey,
  validateInput,
  type SchedulingInput,
  type SchedulingResult,
  type ScheduledMatch,
} from "./types";
import {
  buildFairFixtures,
  computeQuotas,
  transferQuota,
  validateFairness,
} from "./fairness";
import { createRng, deriveSeed, type PRNG } from "./random";
import {
  assembleBreakdown,
  computeObjectiveBreakdown,
  evaluateMatch,
  opponentKeysOf,
  partnerKeysOf,
  scheduleKey,
  weightsFor,
  type MatchEval,
  type ObjectiveBreakdown,
  type ObjectiveWeights,
  type WinProbabilityFn,
} from "./objective";

/**
 * 多起点受约束局部搜索（quota-local-v1）。
 *
 * - 固定 4 个起点：每起点用 fairness 构造合法公平解
 *   （deriveSeed(masterSeed, restartIndex) 派生种子），初始解纳入 best；
 * - 每起点固定 3000 次邻域提案（总 12000），提案类型：
 *   ①同场三种分队重选；②两场交换人员；③两场八位置多重集重新分组；
 *   ④同等公平下的单名额转移（transferQuota，F 不变）；
 * - 接受规则：排序元组改善或持平即接受 current；连续 200 次未改善时
 *   对 current 做 3 次合法随机扰动再继续；best 始终保留；
 * - 无效提案记为无效仍消耗预算，不无限重试；
 * - 懒概率缓存按 canonicalMatchKey；增量评价只动受影响的至多两场，
 *   objective.computeObjectiveBreakdown 是独立全量重算（供校验）；
 * - 返回 4 起点中最优完整合法解（不承诺全局软目标最优），返回前独立校验。
 */

const RESTARTS = 4;
const PROPOSALS_PER_RESTART = 3000;
/**
 * 固定搜索预算，随 algorithmVersion 版本化（变更预算必须升版本）。
 * 随 generation 快照落库，供复算审计。
 */
export const SEARCH_BUDGET = {
  restarts: RESTARTS,
  proposalsPerRestart: PROPOSALS_PER_RESTART,
} as const;
const STAGNATION_LIMIT = 200;
const PERTURB_COUNT = 3;
const PERTURB_MAX_ATTEMPTS = 40;
/** 类型③重新分组：单次提案内有限洗牌尝试，全部无效则该提案记无效 */
const REGROUP_ATTEMPTS = 4;

const POSITIONS = ["a1", "a2", "b1", "b2"] as const;
type Position = (typeof POSITIONS)[number];

export interface SearchState {
  schedule: ScheduledMatch[];
  quotas: Record<string, number>;
  /** 每场评价（与 schedule 对齐，提案只更新受影响位置） */
  evals: MatchEval[];
  /** 本批搭档计数（pairKey -> 次数） */
  partner: Map<string, number>;
  /** 本批个人对手计数 */
  opponent: Map<string, number>;
  /** 本批完整对阵计数（canonicalMatchKey -> 次数） */
  exact: Map<string, number>;
  /** Σ C(partner,2)，整数精确维护 */
  partnerC2: number;
  /** Σ C(opponent,2) */
  opponentC2: number;
  /** Σ_match C(newExact,2) */
  exactC2: number;
  /** Σ_match playedExact*newExact */
  exactPlayed: number;
  /** 懒概率缓存（canonicalMatchKey -> MatchEval），跨起点共享 */
  cache: Map<string, MatchEval>;
  playerIds: string[];
  appearances: Record<string, number>;
  matches: number;
  playedMatchCounts: Record<string, number>;
  weights: ObjectiveWeights;
  winProbability: WinProbabilityFn;
}

function addMatchCounts(st: SearchState, match: ScheduledMatch, ev: MatchEval): void {
  for (const k of partnerKeysOf(match)) {
    const c = st.partner.get(k) ?? 0;
    st.partnerC2 += c; // C(c+1,2)-C(c,2)=c
    st.partner.set(k, c + 1);
  }
  for (const k of opponentKeysOf(match)) {
    const c = st.opponent.get(k) ?? 0;
    st.opponentC2 += c;
    st.opponent.set(k, c + 1);
  }
  const c = st.exact.get(ev.key) ?? 0;
  st.exactC2 += c;
  st.exactPlayed += st.playedMatchCounts[ev.key] ?? 0;
  st.exact.set(ev.key, c + 1);
}

function removeMatchCounts(st: SearchState, match: ScheduledMatch, ev: MatchEval): void {
  for (const k of partnerKeysOf(match)) {
    const c = st.partner.get(k);
    if (c === undefined) {
      throw new Error(`内部错误：搭档计数缺失 ${k}`);
    }
    st.partnerC2 -= c - 1; // C(c,2)-C(c-1,2)=c-1
    if (c === 1) st.partner.delete(k);
    else st.partner.set(k, c - 1);
  }
  for (const k of opponentKeysOf(match)) {
    const c = st.opponent.get(k);
    if (c === undefined) {
      throw new Error(`内部错误：对手计数缺失 ${k}`);
    }
    st.opponentC2 -= c - 1;
    if (c === 1) st.opponent.delete(k);
    else st.opponent.set(k, c - 1);
  }
  const c = st.exact.get(ev.key);
  if (c === undefined) {
    throw new Error(`内部错误：完整对阵计数缺失 ${ev.key}`);
  }
  st.exactC2 -= c - 1;
  st.exactPlayed -= st.playedMatchCounts[ev.key] ?? 0;
  if (c === 1) st.exact.delete(ev.key);
  else st.exact.set(ev.key, c - 1);
}

function evalMatchCached(st: SearchState, match: ScheduledMatch): MatchEval {
  return evaluateMatch(match, st.winProbability, st.cache);
}

/**
 * 由完整方案构建搜索状态：按场次顺序逐场 add，整数聚合量与
 * 全量重算完全一致（comb2 输入为非负整数，结果精确）。
 */
export function createSearchState(
  input: SchedulingInput,
  schedule: ScheduledMatch[],
  quotas: Record<string, number>,
  weights?: ObjectiveWeights,
  cache?: Map<string, MatchEval>
): SearchState {
  const st: SearchState = {
    schedule: schedule.map((m) => ({ ...m })),
    quotas: { ...quotas },
    evals: [],
    partner: new Map(),
    opponent: new Map(),
    exact: new Map(),
    partnerC2: 0,
    opponentC2: 0,
    exactC2: 0,
    exactPlayed: 0,
    cache: cache ?? new Map(),
    playerIds: input.playerIds,
    appearances: input.effectiveAppearances,
    matches: input.matches,
    playedMatchCounts: input.playedMatchCounts,
    weights: weights ?? weightsFor(input),
    winProbability: input.winProbability,
  };
  for (let i = 0; i < st.schedule.length; i++) {
    const ev = evalMatchCached(st, st.schedule[i]);
    st.evals.push(ev);
    addMatchCounts(st, st.schedule[i], ev);
  }
  return st;
}

/**
 * 状态目标分解：sumD2/imbalanced/excess 按 evals 场次序求和，
 * 与 objective.computeObjectiveBreakdown 的迭代顺序逐位一致；
 * C2 类聚合量来自整数标量。两部分组装共用 assembleBreakdown。
 */
export function breakdownOfState(st: SearchState): ObjectiveBreakdown {
  let sumD2 = 0;
  let imbalancedMatchCount = 0;
  let totalImbalanceExcess = 0;
  for (const e of st.evals) {
    sumD2 += e.d2;
    if (e.imbalanced) imbalancedMatchCount++;
    totalImbalanceExcess += e.excess;
  }
  return assembleBreakdown({
    M: st.matches,
    sumD2,
    imbalancedMatchCount,
    totalImbalanceExcess,
    partnerC2: st.partnerC2,
    opponentC2: st.opponentC2,
    exactC2: st.exactC2,
    exactPlayed: st.exactPlayed,
    weights: st.weights,
  });
}

interface GeneratedMove {
  apply(st: SearchState): void;
  rollback(st: SearchState): void;
}

function fourOf(m: ScheduledMatch): [string, string, string, string] {
  return [m.a1, m.a2, m.b1, m.b2];
}

function matchFrom(four: string[], offset = 0): ScheduledMatch {
  return {
    a1: four[offset],
    a2: four[offset + 1],
    b1: four[offset + 2],
    b2: four[offset + 3],
  };
}

function isDistinctFour(m: ScheduledMatch): boolean {
  return new Set(fourOf(m)).size === 4;
}

function setPos(m: ScheduledMatch, pos: Position, value: string): ScheduledMatch {
  const copy = { ...m };
  copy[pos] = value;
  return copy;
}

/** 单场替换型提案的 apply/rollback（类型①与名额转移共用骨架） */
function replaceMatchMove(
  st: SearchState,
  index: number,
  newMatch: ScheduledMatch,
  newQuotas?: Record<string, number>
): GeneratedMove {
  const oldMatch = st.schedule[index];
  const oldEval = st.evals[index];
  const oldQuotas = st.quotas;
  return {
    apply(s) {
      removeMatchCounts(s, oldMatch, oldEval);
      s.schedule[index] = { ...newMatch };
      s.evals[index] = evalMatchCached(s, s.schedule[index]);
      addMatchCounts(s, s.schedule[index], s.evals[index]);
      if (newQuotas) s.quotas = { ...newQuotas };
    },
    rollback(s) {
      const cur = s.schedule[index];
      const curEv = s.evals[index];
      removeMatchCounts(s, cur, curEv);
      s.schedule[index] = oldMatch;
      s.evals[index] = oldEval;
      addMatchCounts(s, oldMatch, oldEval);
      if (newQuotas) s.quotas = oldQuotas;
    },
  };
}

/** 双场替换型提案的 apply/rollback（类型②③共用骨架） */
function replaceTwoMatchesMove(
  st: SearchState,
  i: number,
  newI: ScheduledMatch,
  j: number,
  newJ: ScheduledMatch
): GeneratedMove {
  const oldI = st.schedule[i];
  const oldEvI = st.evals[i];
  const oldJ = st.schedule[j];
  const oldEvJ = st.evals[j];
  return {
    apply(s) {
      removeMatchCounts(s, oldI, oldEvI);
      removeMatchCounts(s, oldJ, oldEvJ);
      s.schedule[i] = { ...newI };
      s.evals[i] = evalMatchCached(s, s.schedule[i]);
      s.schedule[j] = { ...newJ };
      s.evals[j] = evalMatchCached(s, s.schedule[j]);
      addMatchCounts(s, s.schedule[i], s.evals[i]);
      addMatchCounts(s, s.schedule[j], s.evals[j]);
    },
    rollback(s) {
      const curI = s.schedule[i];
      const curEvI = s.evals[i];
      const curJ = s.schedule[j];
      const curEvJ = s.evals[j];
      removeMatchCounts(s, curI, curEvI);
      removeMatchCounts(s, curJ, curEvJ);
      s.schedule[i] = oldI;
      s.evals[i] = oldEvI;
      s.schedule[j] = oldJ;
      s.evals[j] = oldEvJ;
      addMatchCounts(s, oldI, oldEvI);
      addMatchCounts(s, oldJ, oldEvJ);
    },
  };
}

function twoDistinctIndices(rng: PRNG, m: number): [number, number] | null {
  if (m < 2) return null;
  const i = Math.floor(rng() * m);
  const rest = Math.floor(rng() * (m - 1));
  const j = rest >= i ? rest + 1 : rest;
  return [i, j];
}

function shuffled<T>(arr: T[], rng: PRNG): T[] {
  const copy = [...arr];
  for (let k = copy.length - 1; k > 0; k--) {
    const j = Math.floor(rng() * (k + 1));
    [copy[k], copy[j]] = [copy[j], copy[k]];
  }
  return copy;
}

/**
 * 生成一个邻域提案；返回 null 表示提案无效（仍消耗预算）。
 * rng 消耗顺序即提案类型与全部随机决策，保证同种子逐比特可复现。
 */
export function generateMove(st: SearchState, rng: PRNG): GeneratedMove | null {
  const M = st.matches;
  const op = Math.floor(rng() * 4);

  // ① 同场三种分队重选
  if (op === 0) {
    const i = Math.floor(rng() * M);
    const four = fourOf(st.schedule[i]);
    const p = Math.floor(rng() * 3);
    // 4 人固定顺序的三种 2-2 划分
    const parts: number[][] = [
      [0, 1, 2, 3],
      [0, 2, 1, 3],
      [0, 3, 1, 2],
    ];
    const sel = parts[p];
    const newMatch: ScheduledMatch = {
      a1: four[sel[0]],
      a2: four[sel[1]],
      b1: four[sel[2]],
      b2: four[sel[3]],
    };
    return replaceMatchMove(st, i, newMatch);
  }

  // ② 两场交换人员
  if (op === 1) {
    const pair = twoDistinctIndices(rng, M);
    if (!pair) return null;
    const [i, j] = pair;
    const mi = st.schedule[i];
    const mj = st.schedule[j];
    const pi = Math.floor(rng() * 4);
    const pj = Math.floor(rng() * 4);
    const vi = fourOf(mi)[pi];
    const vj = fourOf(mj)[pj];
    if (vi === vj) return null; // 交换同一人：无变化，记无效
    const newI = setPos(mi, POSITIONS[pi], vj);
    const newJ = setPos(mj, POSITIONS[pj], vi);
    if (!isDistinctFour(newI) || !isDistinctFour(newJ)) return null;
    return replaceTwoMatchesMove(st, i, newI, j, newJ);
  }

  // ③ 两场八个位置保留人员多重集重新分组
  if (op === 2) {
    const pair = twoDistinctIndices(rng, M);
    if (!pair) return null;
    const [i, j] = pair;
    const eight = [...fourOf(st.schedule[i]), ...fourOf(st.schedule[j])];
    for (let attempt = 0; attempt < REGROUP_ATTEMPTS; attempt++) {
      const sh = shuffled(eight, rng);
      const newI = matchFrom(sh, 0);
      const newJ = matchFrom(sh, 4);
      if (isDistinctFour(newI) && isDistinctFour(newJ)) {
        return replaceTwoMatchesMove(st, i, newI, j, newJ);
      }
    }
    return null;
  }

  // ④ 同等公平下的单名额转移（F 不变，配额不超上限由 fairness 保证）
  const outcome = transferQuota({
    playerIds: st.playerIds,
    appearances: st.appearances,
    quotas: st.quotas,
    fixtures: st.schedule,
    matches: M,
    rng,
  });
  if (!outcome.transferred || outcome.matchIndex === undefined) return null;
  return replaceMatchMove(
    st,
    outcome.matchIndex,
    outcome.fixtures[outcome.matchIndex],
    outcome.quotas
  );
}

export type ProposalStatus = "improved" | "tied" | "rejected" | "invalid";

/**
 * 单步提案：排序元组改善或持平即接受 current，否则回滚。
 * 返回状态供调用方统计停滞计数与 best 更新。
 */
export function proposeSearchStep(
  st: SearchState,
  rng: PRNG
): ProposalStatus {
  const move = generateMove(st, rng);
  if (!move) return "invalid";

  const bOld = breakdownOfState(st);
  const oldSchedule = st.schedule.slice();
  move.apply(st);
  const bNew = breakdownOfState(st);

  let cmp: number;
  if (bNew.imbalancedMatchCount !== bOld.imbalancedMatchCount) {
    cmp = bNew.imbalancedMatchCount - bOld.imbalancedMatchCount;
  } else if (bNew.totalImbalanceExcess !== bOld.totalImbalanceExcess) {
    cmp = bNew.totalImbalanceExcess - bOld.totalImbalanceExcess;
  } else if (bNew.J !== bOld.J) {
    cmp = bNew.J - bOld.J;
  } else {
    const kOld = scheduleKey(oldSchedule);
    const kNew = scheduleKey(st.schedule);
    cmp = kOld < kNew ? -1 : kOld > kNew ? 1 : 0;
  }

  if (cmp <= 0) return cmp < 0 ? "improved" : "tied";
  move.rollback(st);
  return "rejected";
}

interface BestSnapshot {
  schedule: ScheduledMatch[];
  quotas: Record<string, number>;
  tuple: { imbalancedMatchCount: number; totalImbalanceExcess: number; J: number; key: string };
}

function snapshotState(st: SearchState): BestSnapshot {
  return {
    schedule: st.schedule.map((m) => ({ ...m })),
    quotas: { ...st.quotas },
    tuple: (() => {
      const b = breakdownOfState(st);
      return {
        imbalancedMatchCount: b.imbalancedMatchCount,
        totalImbalanceExcess: b.totalImbalanceExcess,
        J: b.J,
        key: scheduleKey(st.schedule),
      };
    })(),
  };
}

function compareSnapshots(a: BestSnapshot, b: BestSnapshot): number {
  if (a.tuple.imbalancedMatchCount !== b.tuple.imbalancedMatchCount) {
    return a.tuple.imbalancedMatchCount - b.tuple.imbalancedMatchCount;
  }
  if (a.tuple.totalImbalanceExcess !== b.tuple.totalImbalanceExcess) {
    return a.tuple.totalImbalanceExcess - b.tuple.totalImbalanceExcess;
  }
  if (a.tuple.J !== b.tuple.J) return a.tuple.J - b.tuple.J;
  return a.tuple.key < b.tuple.key ? -1 : a.tuple.key > b.tuple.key ? 1 : 0;
}

/** 停滞时对 current 做固定次数合法随机扰动（只看合法性，不看目标） */
function perturbCurrent(st: SearchState, rng: PRNG): void {
  for (let k = 0; k < PERTURB_COUNT; k++) {
    for (let attempt = 0; attempt < PERTURB_MAX_ATTEMPTS; attempt++) {
      const move = generateMove(st, rng);
      if (move) {
        move.apply(st);
        break;
      }
    }
  }
}

/**
 * 引擎入口：校验输入 → 4 起点 × 3000 提案局部搜索 → 返回最优完整合法解。
 * 同完整快照 + 同 seed 逐比特可复现（全部随机性来自派生种子 rng）。
 *
 * weights 缺省取 weightsFor(input)（checkbox 只开关 H）；旧入口适配层
 * 可传入 lambda 映射权重（wB=8*(1-lambda)、wP=4*lambda）。
 */
export function solveScheduling(
  input: SchedulingInput,
  weights?: ObjectiveWeights
): SchedulingResult {
  validateInput(input);
  const { playerIds, matches: M, effectiveAppearances, playedMatchCounts, seed } = input;
  const effectiveWeights = weights ?? weightsFor(input);

  // 懒概率缓存跨起点共享：键是 canonicalMatchKey，评价与朝向无关
  const sharedCache = new Map<string, MatchEval>();
  let globalBest: (BestSnapshot & { F: number }) | null = null;

  for (let restart = 0; restart < RESTARTS; restart++) {
    const rng = createRng(deriveSeed(seed, restart));
    const { quotas, F } = computeQuotas({
      playerIds,
      appearances: effectiveAppearances,
      matches: M,
      rng,
    });
    const fixtures = buildFairFixtures({
      playerIds,
      quotas,
      matches: M,
      rng,
    });

    const st = createSearchState(
      input,
      fixtures,
      quotas,
      effectiveWeights,
      sharedCache
    );

    let best = snapshotState(st);
    let stagnant = 0;

    for (let k = 0; k < PROPOSALS_PER_RESTART; k++) {
      const status = proposeSearchStep(st, rng);
      if (status === "improved") {
        stagnant = 0;
        const cand = snapshotState(st);
        if (compareSnapshots(cand, best) < 0) best = cand;
      } else {
        stagnant++;
      }
      if (stagnant >= STAGNATION_LIMIT) {
        perturbCurrent(st, rng);
        stagnant = 0;
      }
    }

    if (!globalBest || compareSnapshots(best, globalBest) < 0) {
      globalBest = { ...best, F };
    }
  }

  if (!globalBest) {
    throw new Error("算法错误：局部搜索未产生任何起点解");
  }

  // 返回前独立校验：场数=M、每场四人互异且来自名单、各人新增=配额、F=F*；
  // 概率有效性在 evaluateMatch 处已强校验（非法即整次生成失败）。
  try {
    validateFairness({
      playerIds,
      appearances: effectiveAppearances,
      matches: M,
      quotas: globalBest.quotas,
      fixtures: globalBest.schedule,
      expectedF: globalBest.F,
    });
  } catch (error) {
    throw new Error(
      `算法错误：搜索结果未通过独立校验: ${
        error instanceof Error ? error.message : String(error)
      }`
    );
  }

  // 独立全量重算目标（与搜索内增量状态无任何共享代码，逐位对照用）
  const breakdown = computeObjectiveBreakdown(globalBest.schedule, {
    winProbability: input.winProbability,
    playedMatchCounts: input.playedMatchCounts,
    weights: effectiveWeights,
  });

  let directRepeatCount = 0;
  for (const m of globalBest.schedule) {
    const key = canonicalMatchKey(m.a1, m.a2, m.b1, m.b2);
    if ((playedMatchCounts[key] ?? 0) > 0) directRepeatCount++;
  }

  return {
    schedule: globalBest.schedule,
    metrics: {
      J: breakdown.J,
      imbalancedMatchCount: breakdown.imbalancedMatchCount,
      totalImbalanceExcess: breakdown.totalImbalanceExcess,
      directRepeatCount,
    },
    algorithmVersion: ALGORITHM_VERSION,
    objectiveVersion: OBJECTIVE_VERSION,
    seed,
    quotas: globalBest.quotas,
    fairnessF: globalBest.F,
  };
}
