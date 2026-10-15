import { describe, expect, it } from "vitest";
import {
  DEFAULT_WEIGHTS,
  IMBALANCE_THRESHOLD,
  compareRankTuples,
  computeObjectiveBreakdown,
  evaluateMatch,
  evalProbability,
  pairKey,
  scheduleKey,
  tupleFromBreakdown,
  weightsFor,
  type ObjectiveWeights,
} from "./objective";
import { canonicalMatchKey, type ScheduledMatch } from "./types";

const HALF = () => 0.5;

describe("objective", () => {
  describe("概率校验", () => {
    it("拒绝非有限概率（整次生成失败，带阵容信息）", () => {
      for (const bad of [NaN, Infinity, -Infinity]) {
        expect(() =>
          evalProbability("A", "B", "C", "D", () => bad)
        ).toThrowError(/winProbability 返回无效概率/);
      }
      try {
        evalProbability("A", "B", "C", "D", () => NaN);
        expect.unreachable();
      } catch (e) {
        expect((e as Error).message).toContain('["A","B","C","D"]');
      }
    });

    it("拒绝 [0,1] 之外的概率", () => {
      expect(() => evalProbability("A", "B", "C", "D", () => 1.0000001)).toThrow();
      expect(() => evalProbability("A", "B", "C", "D", () => -1e-9)).toThrow();
      expect(evalProbability("A", "B", "C", "D", () => 0).d).toBe(1);
      expect(evalProbability("A", "B", "C", "D", () => 1).d).toBe(1);
      expect(evalProbability("A", "B", "C", "D", HALF).d).toBe(0);
    });
  });

  describe("checkbox 权重", () => {
    it("avoidPlayedRematches 只开关 H 分量，其余权重不变", () => {
      const on = weightsFor({ avoidPlayedRematches: true });
      const off = weightsFor({ avoidPlayedRematches: false });
      expect(on).toEqual({ ...DEFAULT_WEIGHTS, wH: 3 });
      expect(off).toEqual({ ...DEFAULT_WEIGHTS, wH: 0 });
      for (const k of ["wB", "wP", "wO", "wE"] as const) {
        expect(on[k]).toBe(off[k]);
      }
    });

    it("J 的 on/off 差值恰为 3*H（权重只经过 H 生效）", () => {
      const schedule: ScheduledMatch[] = [
        { a1: "A", a2: "B", b1: "C", b2: "D" },
        { a1: "A", a2: "E", b1: "B", b2: "F" },
      ];
      const played = {
        [canonicalMatchKey("A", "B", "C", "D")]: 2,
      };
      const base: ObjectiveWeights = { wB: 4, wP: 2, wO: 0.5, wE: 2, wH: 0 };
      const on: ObjectiveWeights = { ...base, wH: 3 };
      const bOff = computeObjectiveBreakdown(schedule, {
        winProbability: HALF,
        playedMatchCounts: played,
        weights: base,
      });
      const bOn = computeObjectiveBreakdown(schedule, {
        winProbability: HALF,
        playedMatchCounts: played,
        weights: on,
      });
      expect(bOn.J - bOff.J).toBe(3 * bOn.H);
      // H 口径：playedExact*newExact/M = (2*1)/2
      expect(bOn.H).toBe(1);
    });
  });

  describe("分量口径（手算样例）", () => {
    it("B/P/O/E/H 与 J 按合同公式计算", () => {
      // m1: A B vs C D；m2: A B vs E F；对称概率， played: m1 打过 1 次
      const m1: ScheduledMatch = { a1: "A", a2: "B", b1: "C", b2: "D" };
      const m2: ScheduledMatch = { a1: "A", a2: "B", b1: "E", b2: "F" };
      const played = { [canonicalMatchKey("A", "B", "C", "D")]: 1 };

      const b = computeObjectiveBreakdown([m1, m2], {
        winProbability: HALF,
        playedMatchCounts: played,
        weights: weightsFor({ avoidPlayedRematches: true }),
      });
      expect(b.B).toBe(0); // d=0 → d²=0
      expect(b.P).toBe(0.25); // 搭档 AB 出现 2 次：C(2,2)=1，1/(2M)=1/4
      expect(b.O).toBe(0); // 8 个对手对各 1 次
      expect(b.E).toBe(0); // 两场完整对阵互异
      expect(b.H).toBe(0.5); // played(1)*new(1)/M=1/2
      expect(b.J).toBe(2 * 0.25 + 3 * 0.5); // 0.5+1.5
      expect(b.imbalancedMatchCount).toBe(0);
      expect(b.totalImbalanceExcess).toBe(0);
    });

    it("E 统计本批完整对阵重复（同键两次）", () => {
      const m: ScheduledMatch = { a1: "A", a2: "B", b1: "C", b2: "D" };
      const flipped: ScheduledMatch = { a1: "D", a2: "C", b1: "B", b2: "A" };
      const b = computeObjectiveBreakdown([m, flipped], {
        winProbability: HALF,
        playedMatchCounts: {},
        weights: weightsFor({ avoidPlayedRematches: false }),
      });
      // AB、CD 各搭档 2 次：P = (C(2,2)+C(2,2))/(2M) = 2/4 = 0.5
      expect(b.P).toBe(0.5);
      // 同一完整对阵出现 2 次：E = C(2,2)/M = 1/2
      expect(b.E).toBe(0.5);
    });

    it("明显失衡：胜率 >70%（d>0.4）计数并累计超阈值幅度", () => {
      const schedule: ScheduledMatch[] = [
        { a1: "A", a2: "B", b1: "C", b2: "D" },
        { a1: "A", a2: "E", b1: "F", b2: "G" },
        { a1: "A", a2: "H", b1: "I", b2: "J" },
      ];
      // 用 b1 区分场次注入概率
      const pByB1: Record<string, number> = { C: 0.5, F: 0.71, I: 0.9 };
      const b = computeObjectiveBreakdown(schedule, {
        winProbability: (_a1, _a2, b1) => pByB1[b1],
        playedMatchCounts: {},
        weights: weightsFor({ avoidPlayedRematches: false }),
      });
      // d: 0, 0.42, 0.8；阈值 0.4
      expect(b.imbalancedMatchCount).toBe(2);
      expect(b.totalImbalanceExcess).toBeCloseTo(0.02 + 0.4, 10);
      expect(IMBALANCE_THRESHOLD).toBe(0.4);
    });
  });

  describe("排序元组", () => {
    it("字典序：失衡场数优先，其次超阈值幅度，再次 J", () => {
      const base = {
        imbalancedMatchCount: 0,
        totalImbalanceExcess: 0,
        J: 1,
        key: "",
      };
      expect(
        compareRankTuples(
          { ...base, imbalancedMatchCount: 0 },
          { ...base, imbalancedMatchCount: 1 }
        )
      ).toBeLessThan(0);
      expect(
        compareRankTuples(
          { ...base, totalImbalanceExcess: 0.1 },
          { ...base, totalImbalanceExcess: 0.2 }
        )
      ).toBeLessThan(0);
      expect(
        compareRankTuples({ ...base, J: 0.9 }, { ...base, J: 1 })
      ).toBeLessThan(0);
      expect(compareRankTuples(base, { ...base })).toBe(0);
    });

    it("前三项全等时比规范化方案键", () => {
      const a = tupleFromBreakdown(
        {
          B: 0, P: 0, O: 0, E: 0, H: 0,
          J: 1, imbalancedMatchCount: 0, totalImbalanceExcess: 0,
        },
        scheduleKey([{ a1: "A", a2: "B", b1: "C", b2: "D" }])
      );
      const b = tupleFromBreakdown(
        {
          B: 0, P: 0, O: 0, E: 0, H: 0,
          J: 1, imbalancedMatchCount: 0, totalImbalanceExcess: 0,
        },
        scheduleKey([{ a1: "A", a2: "C", b1: "B", b2: "D" }])
      );
      expect(compareRankTuples(a, b)).toBeLessThan(0);
      expect(compareRankTuples(a, { ...a })).toBe(0);
    });
  });

  describe("缓存键", () => {
    it("evaluateMatch 按 canonicalMatchKey 缓存，换朝向复用", () => {
      let calls = 0;
      const wp = () => {
        calls++;
        return 0.7;
      };
      const cache = new Map();
      const e1 = evaluateMatch({ a1: "A", a2: "B", b1: "C", b2: "D" }, wp, cache);
      const e2 = evaluateMatch({ a1: "D", a2: "C", b1: "B", b2: "A" }, wp, cache);
      expect(calls).toBe(1);
      // d 与朝向无关
      expect(e1.d).toBe(e2.d);
      expect(e1.key).toBe(e2.key);
    });

    it("pairKey 无序且与 ID 拼接无歧义", () => {
      expect(pairKey("a", "bc")).toBe(pairKey("bc", "a"));
      expect(pairKey("ab", "c")).not.toBe(pairKey("a", "bc"));
    });
  });
});
