import { describe, it, expect } from "vitest";
import { RATING_REVALIDATE_PATHS } from "./rating-revalidation";

describe("RATING_REVALIDATE_PATHS 路径集合", () => {
  it("包含全部评分相关页面（含动态档案页）", () => {
    const entries = RATING_REVALIDATE_PATHS.map((entry) =>
      entry.type === undefined ? entry.path : `${entry.path} (${entry.type})`
    );
    for (const expected of [
      "/",
      "/trends",
      "/players",
      "/players/[id] (page)",
      "/weekly",
      "/predict",
      "/schedule",
      "/matches",
    ]) {
      expect(entries).toContain(expected);
    }
  });

  it("无重复路径（path + type 组合唯一）", () => {
    const keys = RATING_REVALIDATE_PATHS.map(
      (entry) => `${entry.type ?? ""}:${entry.path}`
    );
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("不含签到等与评分无关的路径", () => {
    const paths = RATING_REVALIDATE_PATHS.map((entry) => entry.path);
    for (const unrelated of ["/signup", "/api/matches", "/api/signups"]) {
      expect(paths).not.toContain(unrelated);
    }
  });

  it("动态档案页以 page 类型 revalidate（覆盖全部档案，而非逐个 id）", () => {
    const dynamic = RATING_REVALIDATE_PATHS.find(
      (entry) => entry.path === "/players/[id]"
    );
    expect(dynamic).toBeDefined();
    expect(dynamic?.type).toBe("page");
  });
});
