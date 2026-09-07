import { describe, it, expect } from "vitest";
import { quoteForWeek, WEEKLY_QUOTES } from "./quote";

// 2026-09-07 是周一；以下时间均为 UTC，注释里给出对应上海时间
describe("quoteForWeek", () => {
  it("上海时间周一 00:00 起属于新的一周", () => {
    // 上海 2026-09-07 00:00 = UTC 2026-09-06 16:00
    const q = quoteForWeek(new Date("2026-09-06T16:00:00Z"));
    expect(q.weekStart).toBe("2026-09-07");
    expect(q.dateLabel).toBe("2026.09.07 — 09.13");
  });

  it("上海时间周日 23:59 仍属于上一周", () => {
    // 上海 2026-09-06 23:59 = UTC 2026-09-06 15:59
    const q = quoteForWeek(new Date("2026-09-06T15:59:00Z"));
    expect(q.weekStart).toBe("2026-08-31");
    expect(q.dateLabel).toBe("2026.08.31 — 09.06");
  });

  it("UTC 周日白天（上海已是周一凌晨）归入新一周", () => {
    // 上海 2026-09-14 00:00（周一）= UTC 2026-09-13 16:00
    const q = quoteForWeek(new Date("2026-09-13T16:00:00Z"));
    expect(q.weekStart).toBe("2026-09-14");
    expect(q.dateLabel).toBe("2026.09.14 — 09.20");
  });

  it("UTC 周日深夜（上海仍是周日）不跨周", () => {
    // 上海 2026-09-13 23:59:59（周日）= UTC 2026-09-13 15:59:59
    const q = quoteForWeek(new Date("2026-09-13T15:59:59Z"));
    expect(q.weekStart).toBe("2026-09-07");
  });

  it("同一周任意时刻结果一致", () => {
    const base = quoteForWeek(new Date("2026-09-06T16:00:00Z")); // 上海周一 00:00
    for (const t of [
      "2026-09-08T03:00:00Z", // 上海周二 11:00
      "2026-09-11T12:00:00Z", // 上海周五 20:00
      "2026-09-13T15:59:59Z", // 上海周日 23:59:59
    ]) {
      expect(quoteForWeek(new Date(t))).toEqual(base);
    }
  });

  it("offsetWeeks 偏移整周，等价于直接查询那一周", () => {
    const base = new Date("2026-09-09T02:00:00Z"); // 上海周三 10:00
    expect(quoteForWeek(base, -1).weekStart).toBe("2026-08-31");
    expect(quoteForWeek(base, 1).weekStart).toBe("2026-09-14");
    expect(quoteForWeek(base, -1)).toEqual(
      quoteForWeek(new Date("2026-09-02T02:00:00Z"))
    );
    expect(quoteForWeek(base, 2)).toEqual(
      quoteForWeek(new Date("2026-09-23T02:00:00Z"))
    );
  });

  it("句子按池长循环", () => {
    const base = new Date("2026-09-09T02:00:00Z");
    const n = WEEKLY_QUOTES.length;
    expect(quoteForWeek(base, n).text).toBe(quoteForWeek(base).text);
    expect(quoteForWeek(base, -n).text).toBe(quoteForWeek(base).text);
    // 连续 n 周覆盖整个池，不重复
    const texts = new Set(
      Array.from({ length: n }, (_, i) => quoteForWeek(base, -i).text)
    );
    expect(texts.size).toBe(n);
  });
});
