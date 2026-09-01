import fs from "node:fs";
import path from "node:path";

export interface ChangelogEntry {
  /** "Unreleased" 或版本号如 "1.0.0" */
  version: string;
  /** 发布日期 YYYY-MM-DD；Unreleased 为 null */
  date: string | null;
  /** 该版本 What's New 小节的条目（网页只渲染这一部分） */
  whatsNew: string[];
}

const VERSION_HEADING = /^## \[(.+?)\](?:\s*-\s*(\d{4}-\d{2}-\d{2}))?\s*$/;
const SUB_HEADING = /^###\s+/;
const WHATS_NEW_HEADING = /^###\s+What's New\s*$/;
const BULLET = /^-\s+(.+?)\s*$/;

export function parseChangelog(markdown: string): ChangelogEntry[] {
  const entries: ChangelogEntry[] = [];
  let current: ChangelogEntry | null = null;
  let inWhatsNew = false;

  for (const line of markdown.split("\n")) {
    const heading = line.match(VERSION_HEADING);
    if (heading) {
      current = {
        version: heading[1],
        date: heading[2] ?? null,
        whatsNew: [],
      };
      entries.push(current);
      inWhatsNew = false;
      continue;
    }
    if (!current) continue;
    if (SUB_HEADING.test(line)) {
      inWhatsNew = WHATS_NEW_HEADING.test(line.trim());
      continue;
    }
    if (inWhatsNew) {
      const bullet = line.trim().match(BULLET);
      if (bullet) current.whatsNew.push(bullet[1]);
    }
  }
  return entries;
}

/**
 * 运行时读取 CHANGELOG.md:Docker 容器内 cwd=/app(volume 挂载命中),
 * 本地 dev cwd=web/(上一级即仓库根目录)。都找不到返回空数组。
 */
export function readChangelog(): ChangelogEntry[] {
  const candidates = [
    process.env.CHANGELOG_PATH,
    path.join(process.cwd(), "CHANGELOG.md"),
    path.join(process.cwd(), "..", "CHANGELOG.md"),
  ].filter((p): p is string => Boolean(p));
  for (const p of candidates) {
    try {
      return parseChangelog(fs.readFileSync(p, "utf8"));
    } catch {
      // 文件不存在或不可读,试下一个候选路径
    }
  }
  return [];
}
