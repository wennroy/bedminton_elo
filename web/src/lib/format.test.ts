// TZ 必须在任何 Date 运算之前设置,保证不同时区的机器上结果一致
process.env.TZ = "Asia/Shanghai";

import { describe, it, expect } from "vitest";
import { formatMatchMeta } from "./format";

describe("formatMatchMeta", () => {
  it("treats sqlite datetime('now') as UTC and renders local time", () => {
    // UTC 13:43 = 北京时间 21:43
    expect(formatMatchMeta("2026-08-30", "2026-08-30 13:43:05", "凯哥")).toBe(
      "8月30日 对局 · 凯哥 录入于 21:43"
    );
  });

  it("accepts ISO strings with Z", () => {
    expect(formatMatchMeta("2026-01-05", "2026-01-05T00:05:00Z", "张伟")).toBe(
      "1月5日 对局 · 张伟 录入于 08:05"
    );
  });

  it("keeps explicit offsets as-is", () => {
    expect(
      formatMatchMeta("2026-08-30", "2026-08-30T21:43:05+08:00", "李娜")
    ).toBe("8月30日 对局 · 李娜 录入于 21:43");
  });

  it("shows the played date, not the recorded date", () => {
    // 补录场景:9月1日录入 8月30日的比赛
    expect(formatMatchMeta("2026-08-30", "2026-09-01 13:00:00", "王强")).toBe(
      "8月30日 对局 · 王强 录入于 21:00"
    );
  });
});
