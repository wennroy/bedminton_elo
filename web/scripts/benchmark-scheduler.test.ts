import { describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildHistoryState,
  elosFor,
  loadFixture,
  percentile,
  durationStats,
  winProbabilityFromElos,
} from "./benchmark-scheduler";
import { canonicalMatchKey } from "../src/lib/scheduling/types";

const fixturePath = join(
  dirname(fileURLToPath(import.meta.url)),
  "../test/fixtures/scheduler-benchmark-cases.json"
);

describe("benchmark-scheduler helpers", () => {
  it("percentile 使用最近秩口径", () => {
    const sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(sorted, 50)).toBe(5);
    expect(percentile(sorted, 95)).toBe(10);
    expect(percentile(sorted, 100)).toBe(10);
    expect(percentile([7], 95)).toBe(7);
    expect(percentile([], 95)).toBeNaN();
  });

  it("durationStats 聚合均值与分位", () => {
    const s = durationStats([10, 20, 30, 40]);
    expect(s.mean).toBe(25);
    expect(s.p50).toBe(20);
    expect(s.p95).toBe(40);
    expect(s.max).toBe(40);
  });

  it("elosFor 四种分布确定且量级合理", () => {
    expect(elosFor("equal", 6)).toEqual([1000, 1000, 1000, 1000, 1000, 1000]);
    const gradient = elosFor("gradient", 5);
    expect(gradient[0]).toBe(900);
    expect(gradient[4]).toBe(1100);
    expect(elosFor("one-strong", 4)).toEqual([1500, 900, 900, 900]);
    expect(elosFor("polarized", 4)).toEqual([700, 700, 1300, 1300]);
  });

  it("winProbabilityFromElos 对称且强弱悬殊时仍合法", () => {
    const even = winProbabilityFromElos({ a: 1000, b: 1000, c: 1000, d: 1000 });
    expect(even("a", "b", "c", "d")).toBe(0.5);
    const strong = winProbabilityFromElos({ a: 1500, b: 1500, c: 900, d: 900 });
    const p = strong("a", "b", "c", "d");
    expect(p).toBeGreaterThan(0.9);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("buildHistoryState: one-played 恰好一条历史且 effective +1", () => {
    const ids = ["p1", "p2", "p3", "p4", "p5"];
    const h = buildHistoryState("one-played", ids, 4, () => 0.5);
    expect(Object.keys(h.playedMatchCounts)).toEqual([
      canonicalMatchKey("p1", "p2", "p3", "p4"),
    ]);
    expect(h.effectiveAppearances).toEqual({ p1: 1, p2: 1, p3: 1, p4: 1, p5: 0 });
  });

  it("buildHistoryState: three-splits 覆盖同一四人的三种分队", () => {
    const ids = ["p1", "p2", "p3", "p4"];
    const h = buildHistoryState("three-splits", ids, 2, () => 0.5);
    expect(Object.keys(h.playedMatchCounts)).toHaveLength(3);
    for (const key of Object.keys(h.playedMatchCounts)) {
      expect(h.playedMatchCounts[key]).toBe(1);
    }
    expect(h.effectiveAppearances).toEqual({ p1: 3, p2: 3, p3: 3, p4: 3 });
  });

  it("buildHistoryState: accumulated 键与 effective 同配置恒定", () => {
    const ids = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"];
    const wp = () => 0.5;
    const a = buildHistoryState("accumulated", ids, 4, wp);
    const b = buildHistoryState("accumulated", ids, 4, wp);
    expect(a.playedMatchCounts).toEqual(b.playedMatchCounts);
    expect(a.effectiveAppearances).toEqual(b.effectiveAppearances);
    const totalPlayed = Object.values(a.playedMatchCounts).reduce((x, y) => x + y, 0);
    expect(totalPlayed).toBe(8); // 两批各 4 场
    const totalEff = Object.values(a.effectiveAppearances).reduce((x, y) => x + y, 0);
    expect(totalEff).toBe(32); // 每场四人
  });

  it("buildHistoryState: empty 全零", () => {
    const h = buildHistoryState("empty", ["p1", "p2", "p3", "p4"], 1, () => 0.5);
    expect(h.playedMatchCounts).toEqual({});
    expect(h.effectiveAppearances).toEqual({ p1: 0, p2: 0, p3: 0, p4: 0 });
  });

  it("loadFixture 接受当前矩阵文件", () => {
    const fixture = loadFixture(fixturePath);
    expect(fixture.engine.matrixN).toEqual([4, 5, 7, 8, 10, 12, 16]);
    expect(fixture.engine.matrixM).toEqual([1, 4, 8, 16]);
    expect(fixture.engine.seeds).toHaveLength(30);
    expect(fixture.engine.variants).toHaveLength(7);
    expect(fixture.service.commonScaleP95TargetMs).toBe(3000);
  });

  it("loadFixture 拒绝越界 seed 与未知分布", () => {
    const raw = JSON.parse(readFileSync(fixturePath, "utf-8")) as Record<string, unknown>;
    const dir = mkdtempSync(join(tmpdir(), "bench-fixture-test-"));
    const badSeedPath = join(dir, "bad-seed.json");
    writeFileSync(
      badSeedPath,
      JSON.stringify({
        ...raw,
        engine: { ...(raw.engine as object), seeds: [-1] },
      })
    );
    expect(() => loadFixture(badSeedPath)).toThrow(RangeError);

    const badDistPath = join(dir, "bad-dist.json");
    const engine = raw.engine as Record<string, unknown>;
    writeFileSync(
      badDistPath,
      JSON.stringify({
        ...raw,
        engine: {
          ...engine,
          variants: [{ id: "x", N: 10, M: 8, distribution: "chaos", history: "empty" }],
        },
      })
    );
    expect(() => loadFixture(badDistPath)).toThrow(RangeError);
    rmSync(dir, { recursive: true, force: true });
  });
});
