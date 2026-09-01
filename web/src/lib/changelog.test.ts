import { describe, it, expect } from "vitest";
import { parseChangelog } from "./changelog";

const SAMPLE = `# Changelog

本项目的所有重要更新都会记录在此文件中。

## [Unreleased]

### What's New

<!-- 发版前填写：用面向球友的自然语言描述这个版本的新功能和变化 -->
- 更新日志时间线页
- 个人主页趣味数据

### Fixed

- 这一小节不该被网页解析

## [1.1.0] - 2026-09-15

### What's New

- 本周战绩显示对局日期和记录时间

## [1.0.0] - 2026-08-31

### Added

- 只有其他小节的版本

### What's New

- 网站正式上线
`;

describe("parseChangelog", () => {
  it("parses versions, dates and What's New bullets in order", () => {
    const entries = parseChangelog(SAMPLE);
    expect(entries.map((e) => e.version)).toEqual([
      "Unreleased",
      "1.1.0",
      "1.0.0",
    ]);
    expect(entries[0].date).toBeNull();
    expect(entries[1].date).toBe("2026-09-15");
    expect(entries[2].date).toBe("2026-08-31");
  });

  it("only collects bullets under What's New, ignoring other subsections", () => {
    const entries = parseChangelog(SAMPLE);
    expect(entries[0].whatsNew).toEqual([
      "更新日志时间线页",
      "个人主页趣味数据",
    ]);
    expect(entries[2].whatsNew).toEqual(["网站正式上线"]);
  });

  it("skips HTML comments inside What's New", () => {
    const entries = parseChangelog(SAMPLE);
    expect(entries[0].whatsNew.join()).not.toContain("发版前填写");
  });

  it("returns an empty array for empty or heading-less markdown", () => {
    expect(parseChangelog("")).toEqual([]);
    expect(parseChangelog("# 只有标题\n\n- 普通列表\n")).toEqual([]);
  });

  it("handles a version without What's New subsection", () => {
    const entries = parseChangelog("## [0.2.0] - 2026-01-01\n\n### Fixed\n\n- 修复\n");
    expect(entries).toEqual([
      { version: "0.2.0", date: "2026-01-01", whatsNew: [] },
    ]);
  });
});
