import { describe, expect, it } from "vitest";
import {
  ALGORITHM_VERSION,
  OBJECTIVE_VERSION,
  canonicalMatchKey,
  type SchedulingInput,
  type ScheduledMatch,
} from "./types";
import { computeQuotas, buildFairFixtures, validateFairness } from "./fairness";
import { createRng, deriveSeed, type PRNG } from "./random";
import { computeObjectiveBreakdown, weightsFor } from "./objective";
import {
  breakdownOfState,
  createSearchState,
  proposeSearchStep,
  solveScheduling,
} from "./search";

/** 确定性伪概率：由 ID 字符编码导出 [0.25,0.75] 的不对称概率 */
function deterministicWinProbability(
  a1: string,
  a2: string,
  b1: string,
  b2: string
): number {
  const code = (s: string) =>
    [...s].reduce((acc, ch) => (acc * 31 + ch.charCodeAt(0)) >>> 0, 7);
  const sum = code(a1) + code(a2) + code(b1) + code(b2);
  return 0.25 + ((sum % 1000) / 1000) * 0.5;
}

function makeInput(overrides: Partial<SchedulingInput> = {}): SchedulingInput {
  const playerIds = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"];
  const effectiveAppearances: Record<string, number> = {};
  for (const id of playerIds) effectiveAppearances[id] = 0;
  return {
    playerIds,
    matches: 6,
    effectiveAppearances,
    playedMatchCounts: {},
    avoidPlayedRematches: false,
    seed: 12345,
    algorithmVersion: ALGORITHM_VERSION,
    objectiveVersion: OBJECTIVE_VERSION,
    winProbability: deterministicWinProbability,
    ...overrides,
  };
}

function appearancesOf(schedule: ScheduledMatch[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const m of schedule) {
    for (const id of [m.a1, m.a2, m.b1, m.b2]) {
      counts[id] = (counts[id] ?? 0) + 1;
    }
  }
  return counts;
}

describe("solveScheduling", () => {
  it("同完整快照 + 同 seed 两次调用逐位可复现", () => {
    const input = makeInput();
    const r1 = solveScheduling(input);
    const r2 = solveScheduling(makeInput());
    expect(r1).toEqual(r2);
  });

  it("无历史时 avoidPlayedRematches on/off 输出一致", () => {
    const on = solveScheduling(makeInput({ avoidPlayedRematches: true }));
    const off = solveScheduling(makeInput({ avoidPlayedRematches: false }));
    expect(on.schedule).toEqual(off.schedule);
    expect(on.metrics).toEqual(off.metrics);
  });

  it("对称评分下勾选后优先避开已打过的完整对阵", () => {
    const playedKey = canonicalMatchKey("p1", "p2", "p3", "p4");
    const played = { [playedKey]: 1 };
    const eff: Record<string, number> = { p1: 0, p2: 0, p3: 0, p4: 0 };
    const base = makeInput({
      playerIds: ["p1", "p2", "p3", "p4"],
      matches: 1,
      effectiveAppearances: eff,
      playedMatchCounts: played,
      winProbability: () => 0.5,
      seed: 99,
    });
    const on = solveScheduling({ ...base, avoidPlayedRematches: true });
    expect(on.schedule).toHaveLength(1);
    const onKey = canonicalMatchKey(
      on.schedule[0].a1, on.schedule[0].a2, on.schedule[0].b1, on.schedule[0].b2
    );
    expect(onKey).not.toBe(playedKey);

    // 不勾选：所有对阵 J 并列，tie-break 只按键排序，不保证避开已打对阵；
    // 同样确定性可复现。
    const off1 = solveScheduling({ ...base, avoidPlayedRematches: false });
    const off2 = solveScheduling({ ...base, avoidPlayedRematches: false });
    expect(off1.schedule).toEqual(off2.schedule);
    expect(off1.schedule).toHaveLength(1);
  });

  it("仅 4 人 M=20 仍出解（允许自然重复）", () => {
    const playerIds = ["a", "b", "c", "d"];
    const eff: Record<string, number> = { a: 0, b: 0, c: 0, d: 0 };
    const result = solveScheduling(
      makeInput({ playerIds, matches: 20, effectiveAppearances: eff, seed: 7 })
    );
    expect(result.schedule).toHaveLength(20);
    for (const m of result.schedule) {
      expect(new Set([m.a1, m.a2, m.b1, m.b2]).size).toBe(4);
    }
    // 每人恰好 20 场（配额拉平）
    expect(appearancesOf(result.schedule)).toEqual({ a: 20, b: 20, c: 20, d: 20 });
    expect(() =>
      validateFairness({
        playerIds,
        appearances: eff,
        matches: 20,
        quotas: result.quotas,
        fixtures: result.schedule,
        expectedF: result.fairnessF,
      })
    ).not.toThrow();
  });

  it("返回方案排序分不差于 4 个初始公平解中最佳者", () => {
    const input = makeInput();
    const result = solveScheduling(input);
    const weights = weightsFor(input);

    let bestInitial: { imb: number; exc: number; J: number } | null = null;
    for (let restart = 0; restart < 4; restart++) {
      const rng = createRng(deriveSeed(input.seed, restart));
      const { quotas } = computeQuotas({
        playerIds: input.playerIds,
        appearances: input.effectiveAppearances,
        matches: input.matches,
        rng,
      });
      const fixtures = buildFixturesFor(input, quotas, rng);
      const b = computeObjectiveBreakdown(fixtures, {
        winProbability: input.winProbability,
        playedMatchCounts: input.playedMatchCounts,
        weights,
      });
      const cand = { imb: b.imbalancedMatchCount, exc: b.totalImbalanceExcess, J: b.J };
      if (
        !bestInitial ||
        cand.imb < bestInitial.imb ||
        (cand.imb === bestInitial.imb &&
          (cand.exc < bestInitial.exc ||
            (cand.exc === bestInitial.exc && cand.J < bestInitial.J)))
      ) {
        bestInitial = cand;
      }
    }

    const got = {
      imb: result.metrics.imbalancedMatchCount,
      exc: result.metrics.totalImbalanceExcess,
      J: result.metrics.J,
    };
    expect(got.imb).toBeLessThanOrEqual(bestInitial!.imb);
    if (got.imb === bestInitial!.imb) {
      expect(got.exc).toBeLessThanOrEqual(bestInitial!.exc);
      if (got.exc === bestInitial!.exc) {
        expect(got.J).toBeLessThanOrEqual(bestInitial!.J);
      }
    }
  });

  it("F 始终等于 computeQuotas 的 F*，且各人新增 == 配额", () => {
    const input = makeInput({ matches: 5, seed: 4242 });
    const result = solveScheduling(input);
    // F* 与 rng 无关（凸可分离贪心最优值）；配额拆分在同代价并列时
    // 由 rng 决定，搜索中的等价名额转移也会合法改变拆分，故只校 F 与执行口径。
    const { F } = computeQuotas({
      playerIds: input.playerIds,
      appearances: input.effectiveAppearances,
      matches: input.matches,
      rng: createRng(1),
    });
    expect(result.fairnessF).toBe(F);
    // 各人实际新增 == 最终配额
    expect(appearancesOf(result.schedule)).toEqual(result.quotas);
    // 配额总额 == 4M
    expect(Object.values(result.quotas).reduce((a, b) => a + b, 0)).toBe(
      4 * input.matches
    );
    expect(() =>
      validateFairness({
        playerIds: input.playerIds,
        appearances: input.effectiveAppearances,
        matches: input.matches,
        quotas: result.quotas,
        fixtures: result.schedule,
        expectedF: F,
      })
    ).not.toThrow();
  });

  it("非法概率使整次生成失败", () => {
    const input = makeInput({
      winProbability: () => NaN,
    });
    expect(() => solveScheduling(input)).toThrowError(/无效概率/);
  });

  it("指标形状完整并带版本号", () => {
    const result = solveScheduling(makeInput());
    expect(result.algorithmVersion).toBe("quota-local-v1");
    expect(result.objectiveVersion).toBe("balanced-v1");
    expect(result.seed).toBe(12345);
    expect(result.metrics.J).toBeGreaterThanOrEqual(0);
    expect(result.metrics.directRepeatCount).toBe(0);
    expect(result.schedule).toHaveLength(6);
  });
});

// 与 search 内部完全相同的初始解构造（computeQuotas + buildFairFixtures），
// 用于独立复算 4 个初始公平解的排序分上界。
function buildFixturesFor(
  input: SchedulingInput,
  quotas: Record<string, number>,
  rng: PRNG
): ScheduledMatch[] {
  return buildFairFixtures({
    playerIds: input.playerIds,
    quotas,
    matches: input.matches,
    rng,
  });
}

describe("增量评价与独立全量重算", () => {
  it("随机提案序列上逐位一致（含已打历史与勾选）", () => {
    const input = makeInput({
      matches: 8,
      seed: 777,
      avoidPlayedRematches: true,
      playedMatchCounts: {
        [canonicalMatchKey("p1", "p2", "p3", "p4")]: 1,
        [canonicalMatchKey("p5", "p6", "p7", "p8")]: 2,
      },
    });
    const rng = createRng(deriveSeed(input.seed, 0));
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
    const st = createSearchState(input, fixtures, quotas);

    const weights = weightsFor(input);
    const full0 = computeObjectiveBreakdown(st.schedule, {
      winProbability: input.winProbability,
      playedMatchCounts: input.playedMatchCounts,
      weights,
    });
    expect(breakdownOfState(st)).toEqual(full0);

    const stepRng = createRng(987654321);
    for (let k = 0; k < 800; k++) {
      proposeSearchStep(st, stepRng);
      const incremental = breakdownOfState(st);
      const full = computeObjectiveBreakdown(st.schedule, {
        winProbability: input.winProbability,
        playedMatchCounts: input.playedMatchCounts,
        weights,
      });
      // 逐位一致（toEqual 对 number 用 Object.is 语义）
      expect(incremental).toEqual(full);
      // 状态始终合法（提案只产生四人互异的场）
      for (const m of st.schedule) {
        expect(new Set([m.a1, m.a2, m.b1, m.b2]).size).toBe(4);
      }
    }

    // 结束后整体公平性仍成立（名额转移保持 F）
    const F = computeQuotas({
      playerIds: input.playerIds,
      appearances: input.effectiveAppearances,
      matches: input.matches,
      rng: createRng(1),
    }).F;
    expect(() =>
      validateFairness({
        playerIds: input.playerIds,
        appearances: input.effectiveAppearances,
        matches: input.matches,
        quotas: st.quotas,
        fixtures: st.schedule,
        expectedF: F,
      })
    ).not.toThrow();
  });
});
