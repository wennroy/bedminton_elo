/**
 * 排阵引擎/服务层基准与硬验收脚本（T6）。
 *
 * 用法:
 *   pnpm exec tsx scripts/benchmark-scheduler.ts [--out <path>]
 *
 * 覆盖:
 * - 引擎矩阵 N∈{4,5,7,8,10,12,16} × M∈{1,4,8,16} × seed 1..30（gradient/空历史），
 *   另含评分分布变体（同分/单强手/两极分化）、历史变体（一种已打/三种分队/
 *   续排累计）、边界 N=64/M=99、checkbox 可观察构造例；
 * - 硬验收断言（fail-fast，任一失败即写 JSON、打摘要并以非零退出）：
 *   ①每场四人互异且来自名单；②场数=M；③各人实际新增=配额；
 *   ④F=F*（computeQuotas 独立重算）；⑤返回指标与 computeObjectiveBreakdown
 *   全量重算逐位一致；⑥返回方案排序三元组不差于各起点初始公平解的最佳；
 *   checkbox 构造例勾选时避开已打对阵；变体用例做同种子逐比特确定性复跑；
 * - 质量报告：出场分布、搭档重复、>70% 失衡场数、本批/对已打完整对阵重复；
 * - 新旧对比：保留的旧 golden（pre-v2 SA）与新引擎同输入分列对比（不承诺全维改善）；
 * - 服务层：临时 SQLite 合成库（createDb + addPlayer + 合成比赛），
 *   createProjectService 完整生成路径（含评分快照捕获与落库）暖态 p50/p95/max，
 *   常用规模目标 p95≤3s；边界 64/99 同测；续排多轮累计历史正确性 +
 *   PRAGMA foreign_key_check 收尾。
 *
 * 绝不触碰真实 data 目录；临时库建在 os.tmpdir() 并尽力清理。
 */
import { randomUUID } from "node:crypto";
import { readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import Database from "better-sqlite3";
import { createDb } from "../src/lib/db";
import { addPlayer } from "../src/lib/repo";
import { recordMatch } from "../src/lib/match-recording";
import {
  createProjectService,
  markPlayedService,
  recordFixtureScoreService,
  regenerateProjectService,
} from "../src/lib/schedule-service";
import { getProject } from "../src/lib/schedule-projects";
import { optimizeSchedule } from "../src/lib/scheduler";
import { predictTeamOutcomeWin } from "../src/lib/trueskill";
import { solveScheduling, SEARCH_BUDGET } from "../src/lib/scheduling/search";
import {
  ALGORITHM_VERSION,
  OBJECTIVE_VERSION,
  canonicalMatchKey,
  type SchedulingInput,
  type SchedulingResult,
  type ScheduledMatch,
} from "../src/lib/scheduling/types";
import {
  buildFairFixtures,
  computeQuotas,
} from "../src/lib/scheduling/fairness";
import { createRng, deriveSeed } from "../src/lib/scheduling/random";
import {
  computeObjectiveBreakdown,
  weightsFor,
  type ObjectiveWeights,
  type WinProbabilityFn,
} from "../src/lib/scheduling/objective";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = join(
  scriptDir,
  "../test/fixtures/scheduler-benchmark-cases.json"
);

type Distribution = "equal" | "gradient" | "one-strong" | "polarized";
type HistoryKind = "empty" | "one-played" | "three-splits" | "accumulated";

interface EngineCaseDef {
  id: string;
  N: number;
  M: number;
  distribution: Distribution;
  history: HistoryKind;
  seeds: number[];
  /** 是否做同种子双跑逐比特确定性复验（矩阵用例关闭以控时长）。 */
  determinismCheck: boolean;
}

interface Fixture {
  engine: {
    seeds: number[];
    baseDistribution: Distribution;
    baseHistory: HistoryKind;
    matrixN: number[];
    matrixM: number[];
    variants: Array<Omit<EngineCaseDef, "seeds" | "determinismCheck">>;
    boundary: { id: string; N: number; M: number; distribution: Distribution; history: HistoryKind; seeds: number[] };
    checkbox: { id: string; N: number; M: number; distribution: Distribution; history: HistoryKind; seeds: number[] };
  };
  service: {
    commonScaleP95TargetMs: number;
    cases: Array<{ id: string; N: number; M: number; repetitions: number }>;
    regenCase: { N: number; M: number; rounds: number };
  };
  golden: { old: string; note: string };
}

// ---------------------------------------------------------------------------
// 断言基础设施（fail-fast）
// ---------------------------------------------------------------------------

class BenchmarkAssertionError extends Error {}

let assertionCount = 0;

function assertTrue(ref: string, name: string, cond: boolean, detail: string): void {
  assertionCount += 1;
  if (!cond) {
    throw new BenchmarkAssertionError(`[${ref}] 断言失败 ${name}: ${detail}`);
  }
}

function assertEqual<T>(ref: string, name: string, actual: T, expected: T): void {
  assertTrue(
    ref,
    name,
    Object.is(actual, expected),
    `实际 ${JSON.stringify(actual)}，期望 ${JSON.stringify(expected)}`
  );
}

// ---------------------------------------------------------------------------
// 合成评分与历史
// ---------------------------------------------------------------------------

function roster(n: number): string[] {
  return Array.from({ length: n }, (_, i) => `p${i + 1}`);
}

/** 确定性合成 ELO（legacy 口径同量级）。 */
export function elosFor(distribution: Distribution, n: number): number[] {
  switch (distribution) {
    case "equal":
      return Array.from({ length: n }, () => 1000);
    case "gradient":
      return Array.from({ length: n }, (_, i) =>
        n === 1 ? 1000 : Math.round(900 + (200 * i) / (n - 1))
      );
    case "one-strong":
      return Array.from({ length: n }, (_, i) => (i === 0 ? 1500 : 900));
    case "polarized":
      return Array.from({ length: n }, (_, i) => (i < n / 2 ? 700 : 1300));
  }
}

/** 队均 ELO 的 Logistic 胜率回调（与 legacy predictElo 同形）。 */
export function winProbabilityFromElos(
  elos: Record<string, number>
): WinProbabilityFn {
  return (a1, a2, b1, b2) => {
    const ea = (elos[a1] + elos[a2]) / 2;
    const eb = (elos[b1] + elos[b2]) / 2;
    return 1 / (1 + Math.pow(10, (eb - ea) / 400));
  };
}

interface HistoryState {
  playedMatchCounts: Record<string, number>;
  effectiveAppearances: Record<string, number>;
}

function addAppearances(
  eff: Record<string, number>,
  four: string[],
  times = 1
): void {
  for (const id of four) eff[id] += times;
}

/**
 * 构造历史状态。accumulated 模拟「续排多次」：用固定种子 101/202 先后跑两批
 * 并全部记为已打，历史键与 effectiveAppearances 均由这两批真实输出累计
 * （同配置恒定，不随用例 seed 变化）。
 */
export function buildHistoryState(
  kind: HistoryKind,
  playerIds: string[],
  M: number,
  winProbability: WinProbabilityFn
): HistoryState {
  const eff: Record<string, number> = {};
  for (const id of playerIds) eff[id] = 0;
  const playedMatchCounts: Record<string, number> = {};

  const bump = (key: string, four: string[], times = 1) => {
    playedMatchCounts[key] = (playedMatchCounts[key] ?? 0) + times;
    addAppearances(eff, four, times);
  };

  if (kind === "one-played") {
    const [a, b, c, d] = playerIds;
    bump(canonicalMatchKey(a, b, c, d), [a, b, c, d]);
    return { playedMatchCounts, effectiveAppearances: eff };
  }
  if (kind === "three-splits") {
    const [a, b, c, d] = playerIds;
    bump(canonicalMatchKey(a, b, c, d), [a, b, c, d]);
    bump(canonicalMatchKey(a, c, b, d), [a, b, c, d]);
    bump(canonicalMatchKey(a, d, b, c), [a, b, c, d]);
    return { playedMatchCounts, effectiveAppearances: eff };
  }
  if (kind === "accumulated") {
    const base = {
      playerIds,
      matches: M,
      effectiveAppearances: eff,
      playedMatchCounts,
      avoidPlayedRematches: false,
      algorithmVersion: ALGORITHM_VERSION,
      objectiveVersion: OBJECTIVE_VERSION,
      winProbability,
    };
    for (const batchSeed of [101, 202]) {
      const batch = solveScheduling({ ...base, seed: batchSeed });
      for (const m of batch.schedule) {
        bump(
          canonicalMatchKey(m.a1, m.a2, m.b1, m.b2),
          [m.a1, m.a2, m.b1, m.b2]
        );
      }
    }
    return { playedMatchCounts, effectiveAppearances: eff };
  }
  return { playedMatchCounts, effectiveAppearances: eff };
}

// ---------------------------------------------------------------------------
// 引擎硬验收断言
// ---------------------------------------------------------------------------

function lex3(
  a: [number, number, number],
  b: [number, number, number]
): number {
  if (a[0] !== b[0]) return a[0] - b[0];
  if (a[1] !== b[1]) return a[1] - b[1];
  return a[2] - b[2];
}

/** 断言 ①②③：场数、四人互异、名单内、各人新增=配额。 */
function assertScheduleShape(
  ref: string,
  playerIds: string[],
  M: number,
  quotas: Record<string, number>,
  schedule: ScheduledMatch[]
): void {
  assertEqual(ref, "match-count", schedule.length, M);
  const rosterSet = new Set(playerIds);
  const actual: Record<string, number> = {};
  for (const id of playerIds) actual[id] = 0;
  for (let i = 0; i < schedule.length; i++) {
    const m = schedule[i];
    const four = [m.a1, m.a2, m.b1, m.b2];
    assertTrue(ref, "distinct-four", new Set(four).size === 4, `第 ${i + 1} 场 ${JSON.stringify(m)}`);
    for (const id of four) {
      assertTrue(ref, "roster", rosterSet.has(id), `第 ${i + 1} 场名单外选手 ${id}`);
      actual[id] += 1;
    }
  }
  for (const id of playerIds) {
    assertEqual(ref, `quota[${id}]`, actual[id], quotas[id]);
  }
}

/** 断言 ④：F 与 computeQuotas 独立重算（独立 rng，只比 F 值）一致。 */
function assertFairnessF(
  ref: string,
  playerIds: string[],
  effectiveAppearances: Record<string, number>,
  M: number,
  result: SchedulingResult
): void {
  const { F } = computeQuotas({
    playerIds,
    appearances: effectiveAppearances,
    matches: M,
    rng: createRng(0x5eed),
  });
  assertEqual(ref, "F==F*", result.fairnessF, F);
}

/** 断言 ⑤：返回指标与 computeObjectiveBreakdown 独立全量重算逐位一致。 */
function assertObjectiveConsistency(
  ref: string,
  schedule: ScheduledMatch[],
  winProbability: WinProbabilityFn,
  playedMatchCounts: Record<string, number>,
  weights: ObjectiveWeights,
  result: SchedulingResult
): void {
  const b = computeObjectiveBreakdown(schedule, {
    winProbability,
    playedMatchCounts,
    weights,
  });
  assertEqual(ref, "J", result.metrics.J, b.J);
  assertEqual(ref, "imbalanced", result.metrics.imbalancedMatchCount, b.imbalancedMatchCount);
  assertEqual(ref, "excess", result.metrics.totalImbalanceExcess, b.totalImbalanceExcess);
}

/**
 * 断言 ⑥：返回方案排序三元组不差于各起点初始公平解的最佳。
 * 起点复现与引擎一致：每 restart 用 deriveSeed(seed, r) 建 rng，先 computeQuotas
 * 再 buildFairFixtures（同一 rng 顺序消费）。
 */
function assertNoWorseThanInitials(
  ref: string,
  input: Pick<
    SchedulingInput,
    "playerIds" | "matches" | "effectiveAppearances" | "seed"
  >,
  winProbability: WinProbabilityFn,
  playedMatchCounts: Record<string, number>,
  weights: ObjectiveWeights,
  result: SchedulingResult
): void {
  let best: [number, number, number] | null = null;
  for (let r = 0; r < SEARCH_BUDGET.restarts; r++) {
    const rng = createRng(deriveSeed(input.seed, r));
    const { quotas } = computeQuotas({
      playerIds: input.playerIds,
      appearances: input.effectiveAppearances,
      matches: input.matches,
      rng,
    });
    const fixtures = buildFairFixtures({
      playerIds: input.playerIds,
      quotas,
      matches: input.matches,
      rng,
    });
    const b = computeObjectiveBreakdown(fixtures, {
      winProbability,
      playedMatchCounts,
      weights,
    });
    const tuple: [number, number, number] = [
      b.imbalancedMatchCount,
      b.totalImbalanceExcess,
      b.J,
    ];
    if (best === null || lex3(tuple, best) < 0) best = tuple;
  }
  const returned: [number, number, number] = [
    result.metrics.imbalancedMatchCount,
    result.metrics.totalImbalanceExcess,
    result.metrics.J,
  ];
  assertTrue(
    ref,
    "no-worse-than-initials",
    best !== null && lex3(returned, best) <= 0,
    `返回 ${JSON.stringify(returned)} 差于最佳起点 ${JSON.stringify(best)}`
  );
}

function runEngineAssertions(
  ref: string,
  input: SchedulingInput,
  result: SchedulingResult
): void {
  const weights = weightsFor(input);
  assertScheduleShape(ref, input.playerIds, input.matches, result.quotas, result.schedule);
  assertFairnessF(ref, input.playerIds, input.effectiveAppearances, input.matches, result);
  assertObjectiveConsistency(
    ref,
    result.schedule,
    input.winProbability,
    input.playedMatchCounts,
    weights,
    result
  );
  assertNoWorseThanInitials(
    ref,
    input,
    input.winProbability,
    input.playedMatchCounts,
    weights,
    result
  );
}

// ---------------------------------------------------------------------------
// 质量指标
// ---------------------------------------------------------------------------

interface RunQuality {
  appearanceSpread: number;
  appearanceStd: number;
  partnerRepeatMax: number;
  distinctPartnerPairs: number;
  opponentRepeatMax: number;
  imbalanced70: number;
  meanD: number;
  directRepeat: number;
}

export function qualityOfRun(
  playerIds: string[],
  schedule: ScheduledMatch[],
  winProbability: WinProbabilityFn,
  playedMatchCounts: Record<string, number>
): RunQuality {
  const appearances: Record<string, number> = {};
  for (const id of playerIds) appearances[id] = 0;
  const partner = new Map<string, number>();
  const opponent = new Map<string, number>();
  let imbalanced70 = 0;
  let sumD = 0;
  let directRepeat = 0;

  for (const m of schedule) {
    const four = [m.a1, m.a2, m.b1, m.b2];
    for (const id of four) appearances[id] += 1;
    const pairs: Array<[string, string]> = [
      [m.a1, m.a2],
      [m.b1, m.b2],
    ];
    for (const [x, y] of pairs) {
      const k = x < y ? `${x}|${y}` : `${y}|${x}`;
      partner.set(k, (partner.get(k) ?? 0) + 1);
    }
    for (const x of [m.a1, m.a2]) {
      for (const y of [m.b1, m.b2]) {
        const k = x < y ? `${x}|${y}` : `${y}|${x}`;
        opponent.set(k, (opponent.get(k) ?? 0) + 1);
      }
    }
    const p = winProbability(m.a1, m.a2, m.b1, m.b2);
    const d = 2 * Math.abs(p - 0.5);
    sumD += d;
    if (d > 0.4) imbalanced70 += 1;
    const key = canonicalMatchKey(m.a1, m.a2, m.b1, m.b2);
    if ((playedMatchCounts[key] ?? 0) > 0) directRepeat += 1;
  }

  const values = playerIds.map((id) => appearances[id]);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return {
    appearanceSpread: Math.max(...values) - Math.min(...values),
    appearanceStd: Math.sqrt(variance),
    partnerRepeatMax: Math.max(0, ...partner.values()),
    distinctPartnerPairs: partner.size,
    opponentRepeatMax: Math.max(0, ...opponent.values()),
    imbalanced70,
    meanD: schedule.length ? sumD / schedule.length : 0,
    directRepeat,
  };
}

function meanQuality(runs: RunQuality[]): Record<string, number> {
  const keys = Object.keys(runs[0]) as Array<keyof RunQuality>;
  const out: Record<string, number> = {};
  for (const k of keys) {
    out[k] = runs.reduce((a, r) => a + r[k], 0) / runs.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// 统计
// ---------------------------------------------------------------------------

export function percentile(sortedAsc: number[], p: number): number {
  if (sortedAsc.length === 0) return NaN;
  const idx = Math.min(
    sortedAsc.length - 1,
    Math.max(0, Math.ceil((p / 100) * sortedAsc.length) - 1)
  );
  return sortedAsc[idx];
}

export function durationStats(durations: number[]): {
  mean: number;
  p50: number;
  p95: number;
  max: number;
} {
  const sorted = [...durations].sort((a, b) => a - b);
  const mean = durations.reduce((a, b) => a + b, 0) / durations.length;
  return {
    mean,
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    max: sorted[sorted.length - 1],
  };
}

// ---------------------------------------------------------------------------
// fixture 加载与运行时校验
// ---------------------------------------------------------------------------

function isDistribution(v: unknown): v is Distribution {
  return v === "equal" || v === "gradient" || v === "one-strong" || v === "polarized";
}
function isHistoryKind(v: unknown): v is HistoryKind {
  return v === "empty" || v === "one-played" || v === "three-splits" || v === "accumulated";
}

function requireIntField(obj: Record<string, unknown>, key: string, min: number, max: number): number {
  const v = obj[key];
  if (typeof v !== "number" || !Number.isSafeInteger(v) || v < min || v > max) {
    throw new RangeError(`fixture 字段 ${key} 必须是 ${min}..${max} 的整数，实际 ${JSON.stringify(v)}`);
  }
  return v;
}

function requireNumberArray(obj: Record<string, unknown>, key: string, min: number, max: number): number[] {
  const v = obj[key];
  if (!Array.isArray(v) || v.length === 0) {
    throw new RangeError(`fixture 字段 ${key} 必须是非空数组`);
  }
  return v.map((n) => requireIntField({ n }, "n", min, max));
}

function requireStringArray(obj: Record<string, unknown>, key: string): string[] {
  const v = obj[key];
  if (!Array.isArray(v) || v.length === 0 || v.some((s) => typeof s !== "string")) {
    throw new RangeError(`fixture 字段 ${key} 必须是非空字符串数组`);
  }
  return v as string[];
}

export function loadFixture(path: string): Fixture {
  const raw = JSON.parse(readFileSync(path, "utf-8")) as Record<string, unknown>;
  const engine = raw.engine as Record<string, unknown>;
  const service = raw.service as Record<string, unknown>;
  const golden = raw.golden as Record<string, unknown>;

  const seeds = (engine.seeds as number[]).map((s) =>
    requireIntField({ s }, "s", 0, 0xffffffff)
  );
  const matrixN = requireNumberArray(engine, "matrixN", 4, 64);
  const matrixM = requireNumberArray(engine, "matrixM", 1, 99);
  for (const n of matrixN) {
    if (!Number.isSafeInteger(n) || n < 4 || n > 64) throw new RangeError(`matrixN 越界: ${n}`);
  }
  for (const m of matrixM) {
    if (!Number.isSafeInteger(m) || m < 1 || m > 99) throw new RangeError(`matrixM 越界: ${m}`);
  }
  const baseDistribution = engine.baseDistribution;
  const baseHistory = engine.baseHistory;
  if (!isDistribution(baseDistribution) || !isHistoryKind(baseHistory)) {
    throw new RangeError("baseDistribution/baseHistory 非法");
  }

  const variants = (engine.variants as Array<Record<string, unknown>>).map((v) => ({
    id: String(v.id),
    N: requireIntField(v, "N", 4, 64),
    M: requireIntField(v, "M", 1, 99),
    distribution: (() => {
      if (!isDistribution(v.distribution)) throw new RangeError(`变体 ${v.id} distribution 非法`);
      return v.distribution;
    })(),
    history: (() => {
      if (!isHistoryKind(v.history)) throw new RangeError(`变体 ${v.id} history 非法`);
      return v.history;
    })(),
  }));

  const boundaryRaw = engine.boundary as Record<string, unknown>;
  const boundary = {
    id: String(boundaryRaw.id),
    N: requireIntField(boundaryRaw, "N", 4, 64),
    M: requireIntField(boundaryRaw, "M", 1, 99),
    distribution: (() => {
      if (!isDistribution(boundaryRaw.distribution)) throw new RangeError("boundary distribution 非法");
      return boundaryRaw.distribution;
    })(),
    history: (() => {
      if (!isHistoryKind(boundaryRaw.history)) throw new RangeError("boundary history 非法");
      return boundaryRaw.history;
    })(),
    seeds: (boundaryRaw.seeds as number[]).map((s) =>
      requireIntField({ s }, "s", 0, 0xffffffff)
    ),
  };

  const checkboxRaw = engine.checkbox as Record<string, unknown>;
  const checkbox = {
    id: String(checkboxRaw.id),
    N: requireIntField(checkboxRaw, "N", 4, 64),
    M: requireIntField(checkboxRaw, "M", 1, 99),
    distribution: (() => {
      if (!isDistribution(checkboxRaw.distribution)) throw new RangeError("checkbox distribution 非法");
      return checkboxRaw.distribution;
    })(),
    history: (() => {
      if (!isHistoryKind(checkboxRaw.history)) throw new RangeError("checkbox history 非法");
      return checkboxRaw.history;
    })(),
    seeds: (checkboxRaw.seeds as number[]).map((s) =>
      requireIntField({ s }, "s", 0, 0xffffffff)
    ),
  };

  const cases = (service.cases as Array<Record<string, unknown>>).map((c) => ({
    id: String(c.id),
    N: requireIntField(c, "N", 4, 64),
    M: requireIntField(c, "M", 1, 99),
    repetitions: requireIntField(c, "repetitions", 1, 200),
  }));
  const regenRaw = service.regenCase as Record<string, unknown>;

  return {
    engine: {
      seeds,
      baseDistribution,
      baseHistory,
      matrixN,
      matrixM,
      variants,
      boundary,
      checkbox,
    },
    service: {
      commonScaleP95TargetMs: requireIntField(service, "commonScaleP95TargetMs", 1, 600000),
      cases,
      regenCase: {
        N: requireIntField(regenRaw, "N", 4, 64),
        M: requireIntField(regenRaw, "M", 1, 99),
        rounds: requireIntField(regenRaw, "rounds", 1, 20),
      },
    },
    golden: { old: String(golden.old), note: String(golden.note) },
  };
}

// ---------------------------------------------------------------------------
// 引擎用例执行
// ---------------------------------------------------------------------------

interface CaseReport {
  id: string;
  N: number;
  M: number;
  distribution: Distribution;
  history: HistoryKind;
  avoidPlayedRematches: boolean;
  seeds: number;
  engineDurationMs: { mean: number; p50: number; p95: number; max: number };
  quality: Record<string, number>;
  assertionsPassed: number;
  determinismVerified: boolean;
}

function runEngineCase(
  def: EngineCaseDef,
  avoidPlayedRematches: boolean
): CaseReport {
  const playerIds = roster(def.N);
  const elos = elosFor(def.distribution, def.N);
  const winProbability = winProbabilityFromElos(
    Object.fromEntries(playerIds.map((id, i) => [id, elos[i]]))
  );
  const { playedMatchCounts, effectiveAppearances } = buildHistoryState(
    def.history,
    playerIds,
    def.M,
    winProbability
  );

  const durations: number[] = [];
  const qualities: RunQuality[] = [];
  const before = assertionCount;

  for (const seed of def.seeds) {
    const input: SchedulingInput = {
      playerIds,
      matches: def.M,
      effectiveAppearances,
      playedMatchCounts,
      avoidPlayedRematches,
      seed,
      algorithmVersion: ALGORITHM_VERSION,
      objectiveVersion: OBJECTIVE_VERSION,
      winProbability,
    };
    const t0 = performance.now();
    const result = solveScheduling(input);
    durations.push(performance.now() - t0);

    const ref = `${def.id}/seed${seed}${avoidPlayedRematches ? "" : "/allow-repeat"}`;
    runEngineAssertions(ref, input, result);
    if (def.determinismCheck) {
      const rerun = solveScheduling(input);
      assertTrue(
        ref,
        "determinism",
        JSON.stringify(rerun) === JSON.stringify(result),
        "同输入同种子两次求解输出不一致"
      );
    }
    qualities.push(
      qualityOfRun(playerIds, result.schedule, winProbability, playedMatchCounts)
    );
  }

  return {
    id: def.id,
    N: def.N,
    M: def.M,
    distribution: def.distribution,
    history: def.history,
    avoidPlayedRematches,
    seeds: def.seeds.length,
    engineDurationMs: durationStats(durations),
    quality: meanQuality(qualities),
    assertionsPassed: assertionCount - before,
    determinismVerified: def.determinismCheck,
  };
}

// ---------------------------------------------------------------------------
// 新旧 golden 对比（同一输入，分列报告，不承诺全维度改善）
// ---------------------------------------------------------------------------

interface GoldenComparison {
  input: { playerIds: number; matches: number; seed: number; lambda: number };
  note: string;
  rows: Array<{ metric: string; old: number; new: number }>;
}

function scheduleAlphaVar(playerIds: string[], schedule: ScheduledMatch[]): number {
  const counts = new Map<string, number>();
  for (const id of playerIds) counts.set(id, 0);
  for (const m of schedule) {
    for (const id of [m.a1, m.a2, m.b1, m.b2]) {
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
  }
  const values = [...counts.values()];
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return values.reduce((s, c) => s + (c - mean) ** 2, 0) / values.length;
}

function runGoldenComparison(fixture: Fixture): GoldenComparison {
  const goldenPath = join(scriptDir, "..", fixture.golden.old);
  const golden = JSON.parse(readFileSync(goldenPath, "utf-8")) as {
    input: {
      playerIds: string[];
      matches: number;
      players: Array<{ mu: number; sigma: number }>;
      seed: number;
      lambda: number;
    };
    output: {
      schedule: ScheduledMatch[];
      metrics: {
        alphaVar: number;
        bestLoss: number;
        meanCloseness: number;
        maxCloseness: number;
        entropy: number;
      };
    };
  };

  const input = golden.input;
  const tsWinProbability = (m: ScheduledMatch) =>
    predictTeamOutcomeWin(
      [m.a1, m.a2].map((id) => input.players[input.playerIds.indexOf(id)]),
      [m.b1, m.b2].map((id) => input.players[input.playerIds.indexOf(id)])
    );

  const oldQuality = qualityOfRun(
    input.playerIds,
    golden.output.schedule,
    (a1, a2, b1, b2) => tsWinProbability({ a1, a2, b1, b2 }),
    {}
  );
  const newResult = optimizeSchedule(input);
  const newQuality = qualityOfRun(
    input.playerIds,
    newResult.schedule,
    (a1, a2, b1, b2) => tsWinProbability({ a1, a2, b1, b2 }),
    {}
  );

  return {
    input: {
      playerIds: input.playerIds.length,
      matches: input.matches,
      seed: input.seed,
      lambda: input.lambda,
    },
    note: fixture.golden.note,
    rows: [
      { metric: "alphaVar(出场方差)", old: golden.output.metrics.alphaVar, new: newResult.metrics.alphaVar },
      { metric: "entropy(旧多样性口径)", old: golden.output.metrics.entropy, new: newResult.metrics.entropy },
      { metric: "meanCloseness(平均|p-0.5|)", old: golden.output.metrics.meanCloseness, new: newResult.metrics.meanCloseness },
      { metric: "maxCloseness(最大|p-0.5|)", old: golden.output.metrics.maxCloseness, new: newResult.metrics.maxCloseness },
      { metric: "bestLoss(目标值,口径不同)", old: golden.output.metrics.bestLoss, new: newResult.metrics.bestLoss },
      { metric: "partnerRepeatMax(搭档重复上限)", old: oldQuality.partnerRepeatMax, new: newQuality.partnerRepeatMax },
      { metric: "imbalanced70(>70%失衡场数)", old: oldQuality.imbalanced70, new: newQuality.imbalanced70 },
      {
        metric: "alphaVar-local(本地复算)",
        old: scheduleAlphaVar(input.playerIds, golden.output.schedule),
        new: scheduleAlphaVar(input.playerIds, newResult.schedule),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// 服务层基准（临时 SQLite，绝不碰真实 data 目录）
// ---------------------------------------------------------------------------

interface ServiceCaseReport {
  id: string;
  N: number;
  M: number;
  repetitions: number;
  durationMs: { mean: number; p50: number; p95: number; max: number };
  fixturesCreated: number;
  assertionsPassed: number;
}

function mkTempDb(tag: string): { db: Database.Database; path: string } {
  const path = join(
    tmpdir(),
    `sched-bench-${tag}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.db`
  );
  return { db: createDb(path), path };
}

function closeTempDb(db: Database.Database, path: string): void {
  try {
    db.close();
  } finally {
    for (const suffix of ["", "-wal", "-shm"]) {
      rmSync(path + suffix, { force: true });
    }
  }
}

function seedPlayers(conn: Database.Database, n: number): string[] {
  const ids: string[] = [];
  for (let i = 0; i < n; i++) {
    ids.push(String(addPlayer(`bench-${n}-${i}-${Math.random().toString(36).slice(2, 6)}`, conn)));
  }
  return ids.sort((a, b) => Number(a) - Number(b));
}

/** 合成比赛制造 legacy ELO 差异（确定性 rng，比分合法且不相等）。 */
function seedLegacyMatches(
  conn: Database.Database,
  ids: string[],
  count: number,
  seed = 20261015
): void {
  const rng = createRng(seed);
  for (let i = 0; i < count; i++) {
    const shuffled = [...ids];
    for (let k = shuffled.length - 1; k > 0; k--) {
      const j = Math.floor(rng() * (k + 1));
      [shuffled[k], shuffled[j]] = [shuffled[j], shuffled[k]];
    }
    const [a, b, c, d] = shuffled;
    recordMatch({
      pa1: Number(a),
      pa2: Number(b),
      pb1: Number(c),
      pb2: Number(d),
      scoreA: 21,
      scoreB: 5 + Math.floor(rng() * 16),
      playedAt: `2026-09-${String(1 + (i % 28)).padStart(2, "0")}`,
      enteredBy: Number(a),
      db: conn,
    });
  }
}

function runServiceCase(def: {
  id: string;
  N: number;
  M: number;
  repetitions: number;
}): ServiceCaseReport {
  const { db, path } = mkTempDb(def.id);
  try {
    const ids = seedPlayers(db, def.N);
    seedLegacyMatches(db, ids, 3 * def.N);

    // 暖场：一次完整生成（含评分快照捕获与落库），不计时。
    createProjectService({
      name: `warmup-${def.id}`,
      playerIds: ids,
      matches: def.M,
      seed: 9000,
      model: "legacy",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });

    const durations: number[] = [];
    let lastFixtures = 0;
    const before = assertionCount;
    for (let i = 0; i < def.repetitions; i++) {
      const t0 = performance.now();
      const result = createProjectService({
        name: `bench-${def.id}`,
        playerIds: ids,
        matches: def.M,
        seed: 9001 + i,
        model: "legacy",
        actorId: ids[0],
        requestId: randomUUID(),
        db,
      });
      durations.push(performance.now() - t0);
      lastFixtures = result.project.fixtures.length;
      if (i === def.repetitions - 1) {
        const ref = `${def.id}/rep${i}`;
        assertEqual(ref, "svc-fixture-count", lastFixtures, def.M);
        for (const f of result.project.fixtures) {
          const four = [f.pa1, f.pa2, f.pb1, f.pb2];
          assertTrue(ref, "svc-distinct-four", new Set(four).size === 4, `fixture ${f.id}`);
          assertTrue(ref, "svc-pending", f.status === "pending", `fixture ${f.id} 状态 ${f.status}`);
          const wr = f.generatedWinRate;
          assertTrue(
            ref,
            "svc-win-rate-range",
            wr !== null && wr >= 0 && wr <= 1,
            `fixture ${f.id} generatedWinRate=${String(wr)}`
          );
        }
      }
    }

    return {
      id: def.id,
      N: def.N,
      M: def.M,
      repetitions: def.repetitions,
      durationMs: durationStats(durations),
      fixturesCreated: lastFixtures,
      assertionsPassed: assertionCount - before,
    };
  } finally {
    closeTempDb(db, path);
  }
}

interface RegenReport {
  N: number;
  M: number;
  rounds: number;
  playedKeysPerRound: number[];
  effectiveAppearancesFinal: Record<string, number>;
  revisionBumps: number[];
  foreignKeyCheckViolations: number;
  assertionsPassed: number;
}

/** 续排多轮：每轮把当前未打场次标记已打（末轮换记分录一条），再续排，历史应累计。 */
function runRegenScenario(cfg: { N: number; M: number; rounds: number }): RegenReport {
  const { db, path } = mkTempDb("regen");
  try {
    const before = assertionCount;
    const ids = seedPlayers(db, cfg.N);
    seedLegacyMatches(db, ids, 2 * cfg.N);

    const created = createProjectService({
      name: "regen-scenario",
      playerIds: ids,
      matches: cfg.M,
      seed: 4242,
      model: "legacy",
      actorId: ids[0],
      requestId: randomUUID(),
      db,
    });
    const projectId = created.project.id;
    let revision = created.project.revision;

    const playedKeysPerRound: number[] = [];
    const revisionBumps: number[] = [];

    for (let round = 1; round <= cfg.rounds; round++) {
      const detail = getProject(projectId, db);
      const pending = detail.fixtures.filter((f) => f.status === "pending");
      // 前一半标记已打；每轮最后一场走记分录分（覆盖 attachScore 路径）。
      const markCount = Math.max(1, Math.floor(pending.length / 2));
      for (let i = 0; i < markCount; i++) {
        const f = pending[i];
        if (i === markCount - 1 && round === cfg.rounds) {
          recordFixtureScoreService({
            projectId,
            fixtureId: f.id,
            scoreA: 21,
            scoreB: 12,
            expectedFixtureRevision: f.revision,
            actorId: ids[0],
            playedAt: "2026-10-01",
            db,
          });
        } else {
          markPlayedService({
            projectId,
            fixtureId: f.id,
            expectedFixtureRevision: f.revision,
            actorId: ids[0],
            db,
          });
        }
      }

      const regen = regenerateProjectService({
        projectId,
        playerIds: ids,
        matches: cfg.M,
        seed: 5000 + round,
        expectedRevision: getProject(projectId, db).revision,
        actorId: ids[0],
        requestId: randomUUID(),
        db,
      });

      const input = regen.project.latestGeneration?.input as {
        playedMatchCounts: Record<string, number>;
        effectiveAppearances: Record<string, number>;
      };
      playedKeysPerRound.push(Object.keys(input.playedMatchCounts).length);
      revisionBumps.push(regen.project.revision - revision);
      revision = regen.project.revision;

      assertEqual(
        `regen-round${round}`,
        "regen-fixture-count",
        regen.fixtures.length,
        cfg.M
      );
      assertTrue(
        `regen-round${round}`,
        "regen-history-accumulates",
        Object.keys(input.playedMatchCounts).length > 0,
        "续排快照应携带累计已打对阵"
      );
    }

    const finalDetail = getProject(projectId, db);
    const finalInput = finalDetail.latestGeneration?.input as {
      effectiveAppearances: Record<string, number>;
    };
    const violations = (db.pragma("foreign_key_check") as unknown[]).length;
    assertEqual("regen-final", "foreign-key-check", violations, 0);

    return {
      N: cfg.N,
      M: cfg.M,
      rounds: cfg.rounds,
      playedKeysPerRound,
      effectiveAppearancesFinal: finalInput.effectiveAppearances,
      revisionBumps,
      foreignKeyCheckViolations: violations,
      assertionsPassed: assertionCount - before,
    };
  } finally {
    closeTempDb(db, path);
  }
}

// ---------------------------------------------------------------------------
// 报告与入口
// ---------------------------------------------------------------------------

interface BenchmarkReport {
  generatedAt: string;
  environment: {
    node: string;
    platform: string;
    arch: string;
    searchBudget: typeof SEARCH_BUDGET;
  };
  engine: {
    matrix: CaseReport[];
    variants: CaseReport[];
    boundary: CaseReport[];
    checkbox: {
      avoid: CaseReport;
      allow: CaseReport;
      observation: string;
    };
  };
  goldenComparison: GoldenComparison;
  service: {
    commonScaleP95TargetMs: number;
    cases: ServiceCaseReport[];
    worstCommonP95Ms: number;
    commonScaleP95Met: boolean;
    regen: RegenReport;
  };
  assertions: { total: number; failures: string[] };
}

function fmtNum(v: number, digits = 1): string {
  return Number.isFinite(v) ? v.toFixed(digits) : "NaN";
}

function printSummary(report: BenchmarkReport): void {
  const line = (s = "") => console.log(s);
  line("================ 排阵基准摘要 ================");
  line(`环境: node ${report.environment.node} | ${report.environment.platform}/${report.environment.arch}`);
  line(`搜索预算: ${report.environment.searchBudget.restarts} 起点 x ${report.environment.searchBudget.proposalsPerRestart} 提案`);
  line("");
  line("-- 引擎矩阵 (gradient/空历史/avoid=true, seed 1..30) --");
  line("case            N    M   p50ms  p95ms  maxms  spread  partnerMax  imbal70  directRep");
  for (const c of report.engine.matrix) {
    line(
      `${c.id.padEnd(14)} ${String(c.N).padStart(3)} ${String(c.M).padStart(4)} ` +
      `${fmtNum(c.engineDurationMs.p50).padStart(7)} ${fmtNum(c.engineDurationMs.p95).padStart(6)} ${fmtNum(c.engineDurationMs.max).padStart(6)} ` +
      `${fmtNum(c.quality.appearanceSpread).padStart(7)} ${fmtNum(c.quality.partnerRepeatMax, 2).padStart(11)} ` +
      `${fmtNum(c.quality.imbalanced70, 2).padStart(8)} ${fmtNum(c.quality.directRepeat, 2).padStart(10)}`
    );
  }
  line("");
  line("-- 变体 (seed 1..30, 含确定性复验) --");
  for (const c of [...report.engine.variants, ...report.engine.boundary.map((b) => ({ ...b }))]) {
    line(
      `${c.id.padEnd(28)} N=${c.N} M=${c.M} ${c.distribution}/${c.history} ` +
      `p95=${fmtNum(c.engineDurationMs.p95)}ms max=${fmtNum(c.engineDurationMs.max)}ms ` +
      `spread=${fmtNum(c.quality.appearanceSpread)} partnerMax=${fmtNum(c.quality.partnerRepeatMax, 2)} ` +
      `imbal70=${fmtNum(c.quality.imbalanced70, 2)} directRep=${fmtNum(c.quality.directRepeat, 2)} det=${c.determinismVerified}`
    );
  }
  line("");
  const cb = report.engine.checkbox;
  line(`-- checkbox 构造例 (${cb.avoid.id}, 同分对称评分 + 一种已打原阵容) --`);
  line(
    `勾选: 30/30 seed 避开已打对阵, directRep 均值=${fmtNum(cb.avoid.quality.directRepeat, 2)} (断言 0)`
  );
  line(
    `不勾选: directRep 均值=${fmtNum(cb.allow.quality.directRepeat, 2)} (不保证, 仅记录) -> ${cb.observation}`
  );
  line("");
  line("-- 新旧 golden 对比 (同一输入, 分列报告, 不承诺全维度改善) --");
  for (const row of report.goldenComparison.rows) {
    line(`  ${row.metric.padEnd(28)} old=${fmtNum(row.old, 4).padStart(10)}  new=${fmtNum(row.new, 4).padStart(10)}`);
  }
  line("");
  line("-- 服务层 createProjectService 暖态耗时 (legacy 模型, 临时库) --");
  line(`case            N    M  reps  p50ms  p95ms  maxms`);
  for (const c of report.service.cases) {
    line(
      `${c.id.padEnd(14)} ${String(c.N).padStart(3)} ${String(c.M).padStart(4)} ${String(c.repetitions).padStart(5)} ` +
      `${fmtNum(c.durationMs.p50).padStart(7)} ${fmtNum(c.durationMs.p95).padStart(6)} ${fmtNum(c.durationMs.max).padStart(7)}`
    );
  }
  const target = report.service.commonScaleP95TargetMs;
  line(
    `常用规模 p95 目标 <= ${target}ms: 最差 p95=${fmtNum(report.service.worstCommonP95Ms)}ms -> ` +
    (report.service.commonScaleP95Met ? "达标" : "未达标（如实报告, 未改预算/引擎）")
  );
  const regen = report.service.regen;
  line(
    `-- 续排 ${regen.rounds} 轮: playedKeys=${JSON.stringify(regen.playedKeysPerRound)} ` +
    `fkViolations=${regen.foreignKeyCheckViolations} revisionBumps=${JSON.stringify(regen.revisionBumps)}`
  );
  line("");
  const failed = report.assertions.failures.length;
  line(`断言: 共执行 ${report.assertions.total} 条, 失败 ${failed} 条`);
  for (const f of report.assertions.failures) line(`  FAIL ${f}`);
  line(failed === 0 ? "结果: PASS" : "结果: FAIL");
  line("=============================================");
}

function parseArgs(argv: string[]): { out?: string } {
  let out: string | undefined;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--out") {
      if (out !== undefined) throw new RangeError("duplicate flag: --out");
      const v = argv[i + 1];
      if (v === undefined || v.startsWith("--")) throw new RangeError("missing value for --out");
      out = v;
      i += 1;
    } else if (arg === "--help" || arg === "-h") {
      console.log("Usage: tsx scripts/benchmark-scheduler.ts [--out <path>]");
      process.exit(0);
    } else {
      throw new RangeError(`unknown flag: ${arg}`);
    }
  }
  return { out };
}

export function buildReport(fixture: Fixture): BenchmarkReport {
  const e = fixture.engine;

  const matrix: CaseReport[] = [];
  for (const N of e.matrixN) {
    for (const M of e.matrixM) {
      matrix.push(
        runEngineCase(
          {
            id: `N${N}-M${M}`,
            N,
            M,
            distribution: e.baseDistribution,
            history: e.baseHistory,
            seeds: e.seeds,
            determinismCheck: false,
          },
          true
        )
      );
    }
  }

  const variants: CaseReport[] = e.variants.map((v) =>
    runEngineCase({ ...v, seeds: e.seeds, determinismCheck: true }, true)
  );

  const boundary: CaseReport[] = [
    runEngineCase(
      {
        id: e.boundary.id,
        N: e.boundary.N,
        M: e.boundary.M,
        distribution: e.boundary.distribution,
        history: e.boundary.history,
        seeds: e.boundary.seeds,
        determinismCheck: true,
      },
      true
    ),
  ];

  const checkboxAvoid = runEngineCase(
    {
      id: e.checkbox.id,
      N: e.checkbox.N,
      M: e.checkbox.M,
      distribution: e.checkbox.distribution,
      history: e.checkbox.history,
      seeds: e.checkbox.seeds,
      determinismCheck: false,
    },
    true
  );
  const checkboxAllow = runEngineCase(
    {
      id: e.checkbox.id,
      N: e.checkbox.N,
      M: e.checkbox.M,
      distribution: e.checkbox.distribution,
      history: e.checkbox.history,
      seeds: e.checkbox.seeds,
      determinismCheck: false,
    },
    false
  );
  // 硬断言：勾选输出在所有 seed 上避开已打对阵（directRepeat 均值必须 0）。
  assertEqual(
    "checkbox",
    "avoid-direct-repeat",
    checkboxAvoid.quality.directRepeat,
    0
  );
  const allowRepeats = checkboxAllow.quality.directRepeat > 0;

  const goldenComparison = runGoldenComparison(fixture);

  const serviceCases: ServiceCaseReport[] = fixture.service.cases.map((c) =>
    runServiceCase(c)
  );
  const commonCases = serviceCases.filter((c) => c.N <= 16 && c.M <= 20);
  const worstCommonP95Ms = Math.max(...commonCases.map((c) => c.durationMs.p95));
  const regen = runRegenScenario(fixture.service.regenCase);

  return {
    generatedAt: new Date().toISOString(),
    environment: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      searchBudget: SEARCH_BUDGET,
    },
    engine: {
      matrix,
      variants,
      boundary,
      checkbox: {
        avoid: checkboxAvoid,
        allow: checkboxAllow,
        observation: allowRepeats
          ? "不勾选时观察到对已打对阵的复现（符合预期：不保证避开）"
          : "不勾选时本构造例恰好未复现（小样本，仍不保证）",
      },
    },
    goldenComparison,
    service: {
      commonScaleP95TargetMs: fixture.service.commonScaleP95TargetMs,
      cases: serviceCases,
      worstCommonP95Ms,
      commonScaleP95Met: worstCommonP95Ms <= fixture.service.commonScaleP95TargetMs,
      regen,
    },
    assertions: { total: assertionCount, failures: [] },
  };
}

function main(): void {
  const { out } = parseArgs(process.argv.slice(2));
  const fixture = loadFixture(FIXTURE_PATH);

  let report: BenchmarkReport | null = null;
  let failure: string | null = null;
  try {
    console.log(`加载用例矩阵: ${FIXTURE_PATH}`);
    console.log("开始基准（引擎矩阵 + 变体 + 边界 + checkbox + golden 对比 + 服务层）...");
    const t0 = Date.now();
    report = buildReport(fixture);
    console.log(`基准完成，用时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  }

  if (report === null) {
    report = {
      generatedAt: new Date().toISOString(),
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch,
        searchBudget: SEARCH_BUDGET,
      },
      engine: { matrix: [], variants: [], boundary: [], checkbox: {} as never },
      goldenComparison: {} as never,
      service: {} as never,
      assertions: { total: assertionCount, failures: failure ? [failure] : ["unknown"] },
    };
  }

  if (out !== undefined) {
    writeFileSync(out, JSON.stringify(report, null, 2));
    console.log(`JSON 报告已写入: ${out}`);
  }
  printSummary(report);

  if (failure !== null || report.assertions.failures.length > 0) {
    process.exitCode = 1;
  }
}

const invokedDirectly =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(process.argv[1]).href;

if (invokedDirectly) {
  main();
}
