import { describe, expect, it } from "vitest";
import { pickAxisTicks, yearBoundaryIndices } from "./axis-ticks";
import type { LocalDate } from "./types";

describe("yearBoundaryIndices", () => {
  it("空序列与单元素没有边界", () => {
    expect(yearBoundaryIndices([])).toEqual(new Set());
    expect(yearBoundaryIndices(["2026-01-01"])).toEqual(new Set());
  });

  it("同年内不标边界", () => {
    expect(
      yearBoundaryIndices(["2026-08-11", "2026-09-01", "2026-12-31"])
    ).toEqual(new Set());
  });

  it("跨年标记新年份的第一个下标", () => {
    expect(
      yearBoundaryIndices(["2025-12-28", "2026-01-04", "2026-01-11"])
    ).toEqual(new Set([1]));
  });

  it("同日多事件只标首个,多次跨年各自标记", () => {
    const dates: LocalDate[] = [
      "2024-12-30",
      "2024-12-30",
      "2025-01-02",
      "2025-01-02",
      "2025-12-29",
      "2026-01-05",
    ];
    expect(yearBoundaryIndices(dates)).toEqual(new Set([2, 5]));
  });
});

describe("pickAxisTicks", () => {
  // 2025-12-01 起连续 40 天:下标 31 = 2026-01-01(跨年边界)
  const dates = Array.from({ length: 40 }, (_, i) => {
    const d = new Date(Date.UTC(2025, 11, 1 + i));
    return d.toISOString().slice(0, 10) as LocalDate;
  });
  const isBoundary = (d: string) => d === "2026-01-01";

  it("数量不超过上限时全部保留", () => {
    const few = dates.slice(0, 5);
    expect(pickAxisTicks(few, { width: 800, isBoundary })).toEqual(few);
  });

  it("按宽度均匀抽稀,且跨年边界始终保留", () => {
    // 40 个刻度、宽 200px(每 44px 一个)→ 最多 4 个常规刻度 + 1 个边界
    const picked = pickAxisTicks(dates, { width: 200, isBoundary });
    expect(picked.length).toBeLessThanOrEqual(4 + 1);
    expect(picked).toContain("2026-01-01");
    expect(picked[0]).toBe("2025-12-01");
    // 顺序保持递增
    expect([...picked].sort()).toEqual(picked);
  });

  it("边界恰好落在常规刻度位时不重复", () => {
    // 8 个值、步长 2:边界在下标 2(常规刻度位)
    const values = ["a", "b", "2026", "c", "d", "e", "f", "g"];
    const picked = pickAxisTicks(values, {
      width: 4 * 44,
      isBoundary: (v) => v === "2026",
    });
    expect(picked.filter((v) => v === "2026")).toHaveLength(1);
  });

  it("宽度为 0 时退化为最少刻度但仍保留边界", () => {
    const picked = pickAxisTicks(dates, { width: 0, isBoundary });
    expect(picked.length).toBeGreaterThanOrEqual(2);
    expect(picked).toContain("2026-01-01");
  });

  it("无边界时纯均匀抽稀", () => {
    const picked = pickAxisTicks(dates, { width: 200, isBoundary: () => false });
    expect(picked).toHaveLength(4);
  });
});
