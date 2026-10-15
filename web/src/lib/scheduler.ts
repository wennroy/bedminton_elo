import { predictTeamOutcomeWin, type TrueSkillPlayer } from "./trueskill";
import {
  ALGORITHM_VERSION,
  OBJECTIVE_VERSION,
} from "./scheduling/types";
import { solveScheduling } from "./scheduling/search";
import type { ObjectiveWeights } from "./scheduling/objective";

export interface ScheduledMatch {
  a1: string;
  a2: string;
  b1: string;
  b2: string;
}

export interface SchedulerMetrics {
  alphaVar: number;
  bestLoss: number;
  meanCloseness: number;
  maxCloseness: number;
  entropy: number;
}

export interface SchedulerResult {
  schedule: ScheduledMatch[];
  metrics: SchedulerMetrics;
  /** 新引擎算法版本（quota-local-v1） */
  algorithmVersion: typeof ALGORITHM_VERSION;
  /** 目标函数版本（balanced-v1） */
  objectiveVersion: typeof OBJECTIVE_VERSION;
  /** 本批完整对阵直接重复次数（对空历史恒为 0） */
  directRepeatCount: number;
}

export interface SchedulerOptions {
  playerIds: string[];
  matches: number;
  players: TrueSkillPlayer[];
  seed: number;
  lambda: number;
  iters?: number;
  startTemp?: number;
  alphaDecay?: number;
  floor?: number;
  /**
   * 自定义「A 队胜率」回调：注入后优化目标完全由它驱动
   * （如 glicko2 用 predictDoubles、legacy 用 predictElo）。
   * 缺省回调用 players 走 predictTeamOutcomeWin（TrueSkill 口径）。
   */
  winProbability?: (match: ScheduledMatch) => number;
}

export type PRNG = () => number;

/**
 * 旧 PRNG 实现，对外导出口径不变（注意此处沿用旧的算术右移
 * `>>` 行为，与 scheduling/random.ts 的无符号 Mulberry32 不同；
 * 保留仅为兼容已有调用方，引擎内部不再使用它）。
 */
export function mulberry32(seed: number): PRNG {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >> 7), r | 61);
    return ((r ^ (r >> 14)) >> 0) / 4294967296;
  };
}

function chaosByEntropy(lst: string[]): number {
  const n = lst.length;
  if (n <= 1) return 1.0;
  const freqs = new Map<string, number>();
  for (const item of lst) {
    freqs.set(item, (freqs.get(item) ?? 0) + 1);
  }
  let h = 0;
  for (const cnt of freqs.values()) {
    const p = cnt / n;
    h -= p * Math.log(p);
  }
  return 1 - h / Math.log(n);
}

/**
 * 旧 lambda 到新权重的映射：8*(1-lambda)*B + 4*lambda*P，
 * 其余分量权重不变（wO=0.5、wE=2）；avoidPlayedRematches 不经过
 * 本入口（预览 API 无历史），wH=0。lambda=0.5 时即 4B+2P，与新默认一致。
 */
export function lambdaToWeights(lambda: number): ObjectiveWeights {
  return { wB: 8 * (1 - lambda), wP: 4 * lambda, wO: 0.5, wE: 2, wH: 0 };
}

export function optimizeSchedule(options: SchedulerOptions): SchedulerResult {
  const {
    playerIds,
    matches: m,
    players,
    seed,
    lambda,
    winProbability,
  } = options;

  // 缺省回调：TrueSkill 口径，与旧硬编码路径相同的表达式。
  const tsCallback: WinProbabilityFn | null = winProbability
    ? null
    : (a1, a2, b1, b2) =>
        predictTeamOutcomeWin(
          [a1, a2].map((id) => players[playerIds.indexOf(id)]),
          [b1, b2].map((id) => players[playerIds.indexOf(id)])
        );

  const callback: WinProbabilityFn = winProbability
    ? (a1, a2, b1, b2) => winProbability({ a1, a2, b1, b2 })
    : (tsCallback as WinProbabilityFn);

  // 旧入口没有出场历史：effectiveAppearances 全 0、无已打对阵、
  // 不勾选避免复现；playerIds 排序是引擎约定，此处内部归一。
  const sortedIds = [...playerIds].sort();
  const effectiveAppearances: Record<string, number> = {};
  for (const id of sortedIds) effectiveAppearances[id] = 0;

  const input = {
    playerIds: sortedIds,
    matches: m,
    effectiveAppearances,
    playedMatchCounts: {},
    avoidPlayedRematches: false,
    seed,
    algorithmVersion: ALGORITHM_VERSION,
    objectiveVersion: OBJECTIVE_VERSION,
    winProbability: callback,
  };

  const result = solveScheduling(input, lambdaToWeights(lambda));
  const schedule = result.schedule;

  // 旧诊断口径：alphaVar = 出场次数方差（对全体 playerIds，n 为分母）
  const counts = new Map<string, number>();
  for (const id of playerIds) counts.set(id, 0);
  const closenessList: number[] = [];
  const scheduleTeam: string[] = [];
  const matchTuples: string[] = [];
  for (const match of schedule) {
    const team1 = [match.a1, match.a2];
    const team2 = [match.b1, match.b2];
    scheduleTeam.push([...team1].sort().join(","));
    scheduleTeam.push([...team2].sort().join(","));
    matchTuples.push(
      `(${[...team1].sort().join(",")})vs(${[...team2].sort().join(",")})`
    );
    for (const id of [...team1, ...team2]) {
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    const p = callback(match.a1, match.a2, match.b1, match.b2);
    closenessList.push(Math.abs(p - 0.5));
  }
  const countValues = Array.from(counts.values());
  const meanCount = countValues.reduce((a, b) => a + b, 0) / playerIds.length;
  const alphaVar =
    countValues.reduce((sum, c) => sum + (c - meanCount) ** 2, 0) /
    playerIds.length;
  const entropy =
    chaosByEntropy(scheduleTeam) + chaosByEntropy(matchTuples);
  const meanCloseness =
    closenessList.reduce((a, b) => a + b, 0) / closenessList.length;
  const maxCloseness = Math.max(...closenessList);

  return {
    schedule,
    metrics: {
      alphaVar,
      bestLoss: result.metrics.J,
      meanCloseness,
      maxCloseness,
      entropy,
    },
    algorithmVersion: result.algorithmVersion,
    objectiveVersion: result.objectiveVersion,
    directRepeatCount: result.metrics.directRepeatCount,
  };
}

type WinProbabilityFn = (
  a1: string,
  a2: string,
  b1: string,
  b2: string
) => number;
