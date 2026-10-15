import { describe, expect, it } from "vitest";
import {
  buildFairFixtures,
  computeQuotas,
  transferQuota,
  validateFairness,
} from "./fairness";
import { createRng } from "./random";

function ids(n: number): string[] {
  return Array.from({ length: n }, (_, i) => `p${i + 1}`);
}

function zeroAppearances(playerIds: string[]): Record<string, number> {
  const a: Record<string, number> = {};
  for (const id of playerIds) a[id] = 0;
  return a;
}

/** 统计 fixtures 里各人实际新增场数 */
function actualAdditions(
  playerIds: string[],
  fixtures: Array<{ a1: string; a2: string; b1: string; b2: string }>
): Record<string, number> {
  const a: Record<string, number> = {};
  for (const id of playerIds) a[id] = 0;
  for (const f of fixtures) {
    a[f.a1]++;
    a[f.a2]++;
    a[f.b1]++;
    a[f.b2]++;
  }
  return a;
}

function sumQuotas(quotas: Record<string, number>): number {
  return Object.values(quotas).reduce((x, y) => x + y, 0);
}

describe("computeQuotas", () => {
  it("4 人多场：每人每场都打，配额 = M", () => {
    const ps = ids(4);
    const { quotas, F } = computeQuotas({
      playerIds: ps,
      appearances: zeroAppearances(ps),
      matches: 5,
      rng: createRng(1),
    });
    for (const id of ps) expect(quotas[id]).toBe(5);
    expect(F).toBe(4 * 25);
  });

  it("4M < N（5 人 1 场）：4 人 1 次、1 人 0 次", () => {
    const ps = ids(5);
    const { quotas, F } = computeQuotas({
      playerIds: ps,
      appearances: zeroAppearances(ps),
      matches: 1,
      rng: createRng(2),
    });
    const values = Object.values(quotas).sort();
    expect(values).toEqual([0, 1, 1, 1, 1]);
    expect(sumQuotas(quotas)).toBe(4);
    expect(F).toBe(4);
  });

  it("5 人 / 7 人 / 10 人：配额和为 4M 且每人 <= M", () => {
    for (const [n, m] of [
      [5, 3],
      [7, 4],
      [10, 6],
    ] as const) {
      const ps = ids(n);
      const { quotas, F } = computeQuotas({
        playerIds: ps,
        appearances: zeroAppearances(ps),
        matches: m,
        rng: createRng(n * 100 + m),
      });
      expect(sumQuotas(quotas)).toBe(4 * m);
      for (const id of ps) {
        expect(quotas[id]).toBeGreaterThanOrEqual(0);
        expect(quotas[id]).toBeLessThanOrEqual(m);
      }
      // 同起点：配额只能是 floor/ceil 两档
      const lo = Math.floor((4 * m) / n);
      const hi = Math.ceil((4 * m) / n);
      for (const id of ps) {
        expect([lo, hi]).toContain(quotas[id]);
      }
      expect(Number.isInteger(F)).toBe(true);
    }
  });

  it("上限容量 64 人 99 场：合法、和 396、每人 <= 99", () => {
    const ps = ids(64);
    const { quotas, F } = computeQuotas({
      playerIds: ps,
      appearances: zeroAppearances(ps),
      matches: 99,
      rng: createRng(64),
    });
    expect(sumQuotas(quotas)).toBe(396);
    for (const id of ps) expect(quotas[id]).toBeLessThanOrEqual(99);
    expect(F).toBeGreaterThan(0);
  });

  it("10 人 8 场：8 人 3 次、2 人 4 次，F=104（下界）", () => {
    const ps = ids(10);
    const { quotas, F } = computeQuotas({
      playerIds: ps,
      appearances: zeroAppearances(ps),
      matches: 8,
      rng: createRng(8),
    });
    const values = Object.values(quotas).sort((a, b) => a - b);
    expect(values).toEqual([3, 3, 3, 3, 3, 3, 3, 3, 4, 4]);
    expect(F).toBe(8 * 9 + 2 * 16); // 104
  });

  it("c_i 不均：5 老成员 c=3 + 新人 c=3（基线等效）共 6 人 3 场，每人新增 2", () => {
    // 「新人从加入时公平」口径：模块只看 effective，c 全等即对称分配
    const ps = ids(6);
    const appearances = zeroAppearances(ps); // 全员 c=3 等效于全员 c=0 的对称结构
    for (const id of ps) appearances[id] = 3;
    for (const seed of [1, 7, 42, 999]) {
      const { quotas, F } = computeQuotas({
        playerIds: ps,
        appearances,
        matches: 3,
        rng: createRng(seed),
      });
      for (const id of ps) expect(quotas[id]).toBe(2);
      expect(F).toBe(6 * 25); // (3+2)^2 * 6 = 150
    }
  });

  it("c_i 不均且 4M 不被 N 整除：向低 c 倾斜", () => {
    const ps = ids(6);
    const appearances: Record<string, number> = {
      p1: 0,
      p2: 1,
      p3: 2,
      p4: 3,
      p5: 4,
      p6: 5,
    };
    const { quotas, F } = computeQuotas({
      playerIds: ps,
      appearances,
      matches: 3,
      rng: createRng(11),
    });
    expect(sumQuotas(quotas)).toBe(12);
    for (const id of ps) {
      expect(quotas[id]).toBeLessThanOrEqual(3);
      expect(quotas[id]).toBeGreaterThanOrEqual(0);
    }
    // 单调：c 小的人配额不低于 c 大的人（同 seed 下成立）
    for (let i = 0; i < ps.length - 1; i++) {
      expect(quotas[ps[i]]).toBeGreaterThanOrEqual(quotas[ps[i + 1]]);
    }
    expect(Number.isInteger(F)).toBe(true);
  });

  it("小规模枚举 oracle：N=4..6、M=1..3、c_i=0..2，F 等于暴力枚举最小值", () => {
    const bruteForceMin = (
      c: number[],
      M: number
    ): number => {
      const n = c.length;
      let best = Infinity;
      const x = new Array<number>(n).fill(0);
      const dfs = (i: number, remaining: number) => {
        if (i === n) {
          if (remaining === 0) {
            let F = 0;
            for (let j = 0; j < n; j++) F += (c[j] + x[j]) ** 2;
            if (F < best) best = F;
          }
          return;
        }
        const upper = Math.min(M, remaining);
        for (let v = 0; v <= upper; v++) {
          x[i] = v;
          dfs(i + 1, remaining - v);
        }
        x[i] = 0;
      };
      dfs(0, 4 * M);
      return best;
    };

    let cases = 0;
    for (let n = 4; n <= 6; n++) {
      const combos = Math.pow(3, n);
      for (let combo = 0; combo < combos; combo++) {
        // 把 combo 解码为 n 个 0..2 的 c_i（含全组合）
        const c: number[] = [];
        let t = combo;
        for (let i = 0; i < n; i++) {
          c.push(t % 3);
          t = Math.floor(t / 3);
        }
        const appearances: Record<string, number> = {};
        const ps = ids(n);
        ps.forEach((id, i) => (appearances[id] = c[i]));
        for (let M = 1; M <= 3; M++) {
          const { F } = computeQuotas({
            playerIds: ps,
            appearances,
            matches: M,
            rng: createRng(combo * 10 + M),
          });
          expect(F).toBe(bruteForceMin(c, M));
          cases++;
        }
      }
    }
    expect(cases).toBeGreaterThan(3000);
  });
});

describe("buildFairFixtures", () => {
  it("多种规模与 seed：场数正确、每场四人互异、各人新增等于配额", () => {
    for (const [n, m, seed] of [
      [4, 2, 1],
      [5, 3, 2],
      [7, 5, 3],
      [10, 8, 4],
      [6, 3, 5],
      [64, 99, 6],
    ] as const) {
      const ps = ids(n);
      const appearances = zeroAppearances(ps);
      const { quotas, F } = computeQuotas({
        playerIds: ps,
        appearances,
        matches: m,
        rng: createRng(seed),
      });
      const fixtures = buildFairFixtures({
        playerIds: ps,
        quotas,
        matches: m,
        rng: createRng(seed + 1000),
      });
      expect(fixtures).toHaveLength(m);
      const roster = new Set(ps);
      for (const f of fixtures) {
        const four = [f.a1, f.a2, f.b1, f.b2];
        expect(new Set(four).size).toBe(4);
        for (const id of four) expect(roster.has(id)).toBe(true);
      }
      const actual = actualAdditions(ps, fixtures);
      for (const id of ps) expect(actual[id]).toBe(quotas[id]);
      validateFairness({
        playerIds: ps,
        appearances,
        matches: m,
        quotas,
        fixtures,
        expectedF: F,
      });
    }
  });

  it("同 seed 完全可复现，不同 seed 通常不同", () => {
    const ps = ids(8);
    const appearances = zeroAppearances(ps);
    const mk = (seed: number) => {
      const { quotas, F } = computeQuotas({
        playerIds: ps,
        appearances,
        matches: 4,
        rng: createRng(seed),
      });
      return buildFairFixtures({
        playerIds: ps,
        quotas,
        matches: 4,
        rng: createRng(seed + 5000),
      });
    };
    const a1 = mk(42);
    const a2 = mk(42);
    expect(a1).toEqual(a2);

    // 8 人 4 场有 3/2 配额差与大量平手机会，换 seed 应得到不同编排
    const others = new Set<number>();
    for (let s = 0; s < 10; s++) {
      others.add(JSON.stringify(mk(s)).length + JSON.stringify(mk(s)).split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0));
    }
    const base = JSON.stringify(a1);
    const differs = Array.from({ length: 10 }, (_, s) => JSON.stringify(mk(s))).some(
      (j) => j !== base
    );
    expect(differs).toBe(true);
  });
});

describe("transferQuota", () => {
  const ps = ["A", "B", "C", "D", "E"];
  const appearances: Record<string, number> = {
    A: 1,
    B: 0,
    C: 1,
    D: 0,
    E: 0,
  };
  const quotas: Record<string, number> = {
    A: 2,
    B: 2,
    C: 2,
    D: 1,
    E: 1,
  };
  // effective: A=3, B=2, C=3, D=1, E=1；可行转移：A/C(3)->B(2)（无场可换）、B(2)->D/E(1)
  const fixtures = [
    { a1: "A", a2: "B", b1: "C", b2: "D" },
    { a1: "A", a2: "B", b1: "C", b2: "E" },
  ];

  it("执行等价转移：F 不变、配额/场次同步、每场仍四人互异", () => {
    const preF =
      Object.entries(quotas).reduce(
        (acc, [id, q]) => acc + (appearances[id] + q) ** 2,
        0
      );
    const outcome = transferQuota({
      playerIds: ps,
      appearances,
      quotas,
      fixtures,
      matches: 2,
      rng: createRng(3),
    });
    expect(outcome.transferred).toBe(true);
    expect(outcome.from).toBe("B"); // 只有 B(2) -> D/E(1) 有可用场
    expect(["D", "E"]).toContain(outcome.to);

    // 配额同步
    expect(outcome.quotas[outcome.from!]).toBe(quotas[outcome.from!] - 1);
    expect(outcome.quotas[outcome.to!]).toBe(quotas[outcome.to!] + 1);
    for (const id of ps) {
      if (id !== outcome.from && id !== outcome.to) {
        expect(outcome.quotas[id]).toBe(quotas[id]);
      }
    }
    // 不超上限
    expect(outcome.quotas[outcome.to!]).toBeLessThanOrEqual(2);

    // F 不变、实际新增与配额一致、场数不变
    const postF =
      Object.entries(outcome.quotas).reduce(
        (acc, [id, q]) => acc + (appearances[id] + q) ** 2,
        0
      );
    expect(postF).toBe(preF);
    expect(outcome.fixtures).toHaveLength(2);
    for (const f of outcome.fixtures) {
      expect(new Set([f.a1, f.a2, f.b1, f.b2]).size).toBe(4);
    }
    const actual = actualAdditions(ps, outcome.fixtures);
    for (const id of ps) expect(actual[id]).toBe(outcome.quotas[id]);

    validateFairness({
      playerIds: ps,
      appearances,
      matches: 2,
      quotas: outcome.quotas,
      fixtures: outcome.fixtures,
      expectedF: preF,
    });
  });

  it("无可行转移时 transferred=false 且内容等价", () => {
    // 全部 effective 相等：不存在 h -> h-1 对
    const ps4 = ["A", "B", "C", "D"];
    const app4 = { A: 0, B: 0, C: 0, D: 0 };
    const q4 = { A: 1, B: 1, C: 1, D: 1 };
    const fx4 = [{ a1: "A", a2: "B", b1: "C", b2: "D" }];
    const outcome = transferQuota({
      playerIds: ps4,
      appearances: app4,
      quotas: q4,
      fixtures: fx4,
      matches: 1,
      rng: createRng(9),
    });
    expect(outcome.transferred).toBe(false);
    expect(outcome.quotas).toEqual(q4);
    expect(outcome.fixtures).toEqual(fx4);
  });

  it("转移后再次构造/校验依旧闭环（多次转移保持 F）", () => {
    let cur = { quotas, fixtures };
    for (let step = 0; step < 3; step++) {
      const outcome = transferQuota({
        playerIds: ps,
        appearances,
        quotas: cur.quotas,
        fixtures: cur.fixtures,
        matches: 2,
        rng: createRng(100 + step),
      });
      if (!outcome.transferred) break;
      const F =
        Object.entries(outcome.quotas).reduce(
          (acc, [id, q]) => acc + (appearances[id] + q) ** 2,
          0
        );
      validateFairness({
        playerIds: ps,
        appearances,
        matches: 2,
        quotas: outcome.quotas,
        fixtures: outcome.fixtures,
        expectedF: F,
      });
      cur = { quotas: outcome.quotas, fixtures: outcome.fixtures };
    }
  });
});

describe("validateFairness", () => {
  const ps = ["A", "B", "C", "D"];
  const appearances = { A: 0, B: 0, C: 0, D: 0 };
  const quotas = { A: 1, B: 1, C: 1, D: 1 };
  const fixtures = [{ a1: "A", a2: "B", b1: "C", b2: "D" }];

  it("合法输入通过", () => {
    expect(() =>
      validateFairness({
        playerIds: ps,
        appearances,
        matches: 1,
        quotas,
        fixtures,
        expectedF: 4,
      })
    ).not.toThrow();
  });

  it("场数不对抛错", () => {
    expect(() =>
      validateFairness({
        playerIds: ps,
        appearances,
        matches: 2,
        quotas,
        fixtures,
        expectedF: 4,
      })
    ).toThrow(/场数/);
  });

  it("同场重复抛错", () => {
    expect(() =>
      validateFairness({
        playerIds: ps,
        appearances,
        matches: 1,
        quotas,
        fixtures: [{ a1: "A", a2: "A", b1: "C", b2: "D" }],
        expectedF: 4,
      })
    ).toThrow(/互异/);
  });

  it("名单外选手抛错", () => {
    expect(() =>
      validateFairness({
        playerIds: ps,
        appearances,
        matches: 1,
        quotas,
        fixtures: [{ a1: "A", a2: "B", b1: "C", b2: "X" }],
        expectedF: 4,
      })
    ).toThrow(/名单外/);
  });

  it("实际新增与配额不符抛错", () => {
    expect(() =>
      validateFairness({
        playerIds: ps,
        appearances,
        matches: 1,
        quotas: { A: 2, B: 1, C: 1, D: 0 },
        fixtures,
        expectedF: 6,
      })
    ).toThrow(/新增/);
  });

  it("F 与 F* 不符抛错", () => {
    expect(() =>
      validateFairness({
        playerIds: ps,
        appearances,
        matches: 1,
        quotas,
        fixtures,
        expectedF: 5,
      })
    ).toThrow(/F/);
  });
});
