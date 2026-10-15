import { describe, expect, it } from "vitest";
import { createRng, deriveSeed } from "./random";

// 旧 scheduler.ts 的缺陷实现（算术右移 >>）：采样到不了 [0.5,1)。
// 内联在测试里做直接回归对照，生产代码不再使用它。
function legacyMulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >> 7), r | 61);
    return ((r ^ (r >> 14)) >> 0) / 4294967296;
  };
}

describe("createRng（标准无符号 Mulberry32）", () => {
  it("matches authoritative reference vectors (seed=0)", () => {
    const rng = createRng(0);
    const expected = [
      0.26642920868471265, 0.0003297457005828619, 0.2232720274478197,
      0.1462021479383111, 0.46732782293111086,
    ];
    for (const want of expected) {
      expect(rng()).toBe(want);
    }
  });

  it("matches authoritative reference vectors (seed=42)", () => {
    const rng = createRng(42);
    const expected = [
      0.6011037519201636, 0.44829055899754167, 0.8524657934904099,
      0.6697340414393693, 0.17481389874592423,
    ];
    for (const want of expected) {
      expect(rng()).toBe(want);
    }
  });

  it("outputs stay in [0,1)", () => {
    for (const seed of [0, 1, 42, 1337, 0xffffffff]) {
      const rng = createRng(seed);
      for (let i = 0; i < 10000; i++) {
        const v = rng();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    }
  });

  it("produces the same sequence for the same seed", () => {
    const a = createRng(7);
    const b = createRng(7);
    for (let i = 0; i < 1000; i++) {
      expect(a()).toBe(b());
    }
  });

  it("produces different sequences for different seeds", () => {
    const a = createRng(1);
    const b = createRng(2);
    const seqA = Array.from({ length: 100 }, () => a());
    const seqB = Array.from({ length: 100 }, () => b());
    expect(seqA).not.toEqual(seqB);
  });

  it("covers both halves [0,0.5) and [0.5,1) — regression for the legacy signed-shift defect", () => {
    const rng = createRng(12345);
    let low = 0;
    let high = 0;
    for (let i = 0; i < 100000; i++) {
      const v = rng();
      if (v < 0.5) low++;
      else high++;
    }
    expect(low).toBeGreaterThan(0);
    expect(high).toBeGreaterThan(0);
    // 两半都应大致各占一半（宽松断言，避免偶发统计抖动）
    expect(high / (low + high)).toBeGreaterThan(0.4);
    expect(high / (low + high)).toBeLessThan(0.6);

    // 直接对照：旧缺陷实现 20 万采样到不了 [0.5,1)
    const legacy = legacyMulberry32(1);
    let legacyMax = -Infinity;
    for (let i = 0; i < 200000; i++) {
      legacyMax = Math.max(legacyMax, legacy());
    }
    expect(legacyMax).toBeLessThan(0.5);
  });

  it("rejects invalid seeds", () => {
    for (const bad of [-1, 1.5, 0x100000000, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => createRng(bad)).toThrow();
    }
  });
});

describe("deriveSeed", () => {
  it("is deterministic for the same inputs", () => {
    expect(deriveSeed(42, 0)).toBe(deriveSeed(42, 0));
    expect(deriveSeed(42, 1)).toBe(deriveSeed(42, 1));
  });

  it("stays in unsigned 32-bit range", () => {
    for (let i = 0; i < 1000; i++) {
      const s = deriveSeed(0xffffffff, i);
      expect(Number.isInteger(s)).toBe(true);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(0xffffffff);
    }
  });

  it("varies with restartIndex for a fixed master seed", () => {
    const seen = new Set<number>();
    for (let i = 0; i < 200; i++) {
      const s = deriveSeed(20261015, i);
      expect(seen.has(s)).toBe(false);
      seen.add(s);
    }
  });

  it("feeds createRng and produces distinct sequences per restart", () => {
    const r0 = createRng(deriveSeed(99, 0));
    const r1 = createRng(deriveSeed(99, 1));
    const seq0 = Array.from({ length: 50 }, () => r0());
    const seq1 = Array.from({ length: 50 }, () => r1());
    expect(seq0).not.toEqual(seq1);
  });

  it("rejects invalid inputs", () => {
    expect(() => deriveSeed(-1, 0)).toThrow();
    expect(() => deriveSeed(0, -1)).toThrow();
    expect(() => deriveSeed(0, 1.5)).toThrow();
  });
});
