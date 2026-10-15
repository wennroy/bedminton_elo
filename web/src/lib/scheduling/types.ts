import type { ScheduledMatch } from "../scheduler";

export type { ScheduledMatch };

export const ALGORITHM_VERSION = "quota-local-v1" as const;
export const OBJECTIVE_VERSION = "balanced-v1" as const;

export type AlgorithmVersion = typeof ALGORITHM_VERSION;
export type ObjectiveVersion = typeof OBJECTIVE_VERSION;

/**
 * 引擎输入。纯 DTO，不读 DB、不读当前时间。
 *
 * playerIds 约定：调用方保证已排序且唯一；validateInput 只校验唯一性，
 * 不重新排序（排序是调用方的责任，引擎内部不依赖顺序做正确性判断）。
 */
export interface SchedulingInput {
  /** 参与本次排阵的选手 ID，唯一（调用方保证已排序），4..64 人 */
  playerIds: string[];
  /** 本批场次数 M，1..99 */
  matches: number;
  /** 每人 effectiveAppearances（已打场数 c_i），覆盖每个 playerId，非负安全整数 */
  effectiveAppearances: Record<string, number>;
  /** 已打完整对阵历史计数：canonicalMatchKey -> 次数 */
  playedMatchCounts: Record<string, number>;
  /** 是否避免与历史完整对阵直接重复（重复由 T2 搜索负责消化） */
  avoidPlayedRematches: boolean;
  /** 随机种子，0..2^32-1 整数 */
  seed: number;
  algorithmVersion: AlgorithmVersion;
  objectiveVersion: ObjectiveVersion;
  /** A 队胜率纯函数回调（评分体系由调用方注入） */
  winProbability: (a1: string, a2: string, b1: string, b2: string) => number;
}

/**
 * 目标指标。J 与各分解项由 T2 搜索填充（T1 只定义形状）。
 * imbalancedMatchCount / totalImbalanceExcess 的「阈值」口径由 T2 定。
 */
export interface SchedulingMetrics {
  /** balanced-v1 目标函数值（越小越好，T2 填充） */
  J: number;
  /** 明显失衡场数（T2 填充） */
  imbalancedMatchCount: number;
  /** 总超阈值幅度（T2 填充） */
  totalImbalanceExcess: number;
  /** 本批完整对阵直接重复次数（对历史 playedMatchCounts 计数，T2 填充） */
  directRepeatCount: number;
}

/** 引擎结果 */
export interface SchedulingResult {
  /** 场次列表，ScheduledMatch 形状（a1/a2/b1/b2 字符串 ID） */
  schedule: ScheduledMatch[];
  metrics: SchedulingMetrics;
  algorithmVersion: AlgorithmVersion;
  objectiveVersion: ObjectiveVersion;
  /** 实际使用的种子（与输入一致；留档便于复现） */
  seed: number;
  /** 公平配额分配 x_i（每人本批新增场数） */
  quotas: Record<string, number>;
  /** 公平目标值 F* = sum_i (c_i + x_i)^2 */
  fairnessF: number;
}

function compareIds(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** 队间字典序：两名成员已分别排序，逐位比较 */
function compareTeams(t1: string[], t2: string[]): number {
  for (let i = 0; i < t1.length; i++) {
    const c = compareIds(t1[i], t2[i]);
    if (c !== 0) return c;
  }
  return 0;
}

/**
 * 完整对阵规范化键：队内两人 ID 排序，两队按字典序排序，JSON.stringify。
 * 'AB vs CD'、'BA vs DC'、'CD vs AB' 同一键；'AC vs BD' 为另一键。
 */
export function canonicalMatchKey(
  a1: string,
  a2: string,
  b1: string,
  b2: string
): string {
  const teamA = [a1, a2].sort(compareIds);
  const teamB = [b1, b2].sort(compareIds);
  const first = compareTeams(teamA, teamB) <= 0 ? teamA : teamB;
  const second = first === teamA ? teamB : teamA;
  return JSON.stringify([first, second]);
}

const MIN_PLAYERS = 4;
const MAX_PLAYERS = 64;
const MIN_MATCHES = 1;
const MAX_MATCHES = 99;
const MAX_SEED = 0xffffffff;

function fail(msg: string): never {
  throw new Error(`SchedulingInput 校验失败: ${msg}`);
}

/**
 * 输入校验：失败抛带明确信息的 Error，不静默吞。
 * 约定：只校验 playerIds 唯一性（不排序）；effectiveAppearances 不允许
 * 出现 playerIds 之外的键（严格模式，尽早暴露调用方 bug）。
 */
export function validateInput(input: SchedulingInput): void {
  const { playerIds, matches, effectiveAppearances, playedMatchCounts, seed } =
    input;

  if (!Array.isArray(playerIds)) fail("playerIds 必须是数组");
  if (
    playerIds.length < MIN_PLAYERS ||
    playerIds.length > MAX_PLAYERS
  ) {
    fail(`playerIds 长度必须在 ${MIN_PLAYERS}..${MAX_PLAYERS}，实际 ${playerIds.length}`);
  }
  for (const id of playerIds) {
    if (typeof id !== "string" || id.length === 0) {
      fail("playerIds 必须是非空字符串");
    }
  }
  if (new Set(playerIds).size !== playerIds.length) {
    fail("playerIds 必须唯一（重复 ID 或排序约定被破坏）");
  }

  if (!Number.isSafeInteger(matches) || matches < MIN_MATCHES || matches > MAX_MATCHES) {
    fail(`matches 必须是 ${MIN_MATCHES}..${MAX_MATCHES} 的整数，实际 ${matches}`);
  }

  if (!Number.isSafeInteger(seed) || seed < 0 || seed > MAX_SEED) {
    fail(`seed 必须是 0..${MAX_SEED} 的整数，实际 ${seed}`);
  }

  if (input.algorithmVersion !== ALGORITHM_VERSION) {
    fail(`algorithmVersion 必须是 '${ALGORITHM_VERSION}'`);
  }
  if (input.objectiveVersion !== OBJECTIVE_VERSION) {
    fail(`objectiveVersion 必须是 '${OBJECTIVE_VERSION}'`);
  }
  if (typeof input.avoidPlayedRematches !== "boolean") {
    fail("avoidPlayedRematches 必须是 boolean");
  }
  if (typeof input.winProbability !== "function") {
    fail("winProbability 必须是函数");
  }

  if (
    effectiveAppearances === null ||
    typeof effectiveAppearances !== "object" ||
    Array.isArray(effectiveAppearances)
  ) {
    fail("effectiveAppearances 必须是 Record<string, number>");
  }
  for (const id of playerIds) {
    const c = effectiveAppearances[id];
    if (c === undefined) fail(`effectiveAppearances 缺少 playerId '${id}'`);
    if (!Number.isSafeInteger(c) || c < 0) {
      fail(`effectiveAppearances['${id}'] 必须是非负安全整数，实际 ${c}`);
    }
  }
  for (const key of Object.keys(effectiveAppearances)) {
    if (!playerIds.includes(key)) {
      fail(`effectiveAppearances 存在名单外键 '${key}'`);
    }
  }

  if (
    playedMatchCounts !== null &&
    typeof playedMatchCounts === "object" &&
    !Array.isArray(playedMatchCounts)
  ) {
    for (const [key, count] of Object.entries(playedMatchCounts)) {
      if (!Number.isSafeInteger(count) || count < 0) {
        fail(`playedMatchCounts['${key}'] 必须是非负安全整数，实际 ${count}`);
      }
    }
  } else {
    fail("playedMatchCounts 必须是 Record<string, number>");
  }
}
