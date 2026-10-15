import Database from "better-sqlite3";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { createDb } from "@/lib/db";
import { addPlayer } from "@/lib/repo";

/**
 * E2E 旅程测试的临时 SQLite 数据文件。
 * 与生产 ./badminton.db 完全隔离：每个测试文件拿到独立目录与独立 DB。
 */
export interface TempDb {
  dir: string;
  dbPath: string;
  /** 测试进程内直接打开同一文件做断言（WAL 支持多进程并发访问）。 */
  open(): Database.Database;
  close(): void;
  cleanup(): void;
}

export function makeTempDb(name: string): TempDb {
  const dir = mkdtempSync(join(tmpdir(), `bedminton-e2e-${name}-`));
  const dbPath = join(dir, "e2e.db");
  let conn: Database.Database | null = null;

  const seed = createDb(dbPath);
  try {
    for (let i = 1; i <= 12; i += 1) {
      addPlayer(`E2E-${String(i).padStart(2, "0")}`, seed);
    }
  } finally {
    seed.close();
  }

  return {
    dir,
    dbPath,
    open() {
      if (!conn) {
        conn = createDb(dbPath);
      }
      return conn;
    },
    close() {
      if (conn) {
        conn.close();
        conn = null;
      }
    },
    cleanup() {
      try {
        rmSync(dir, { recursive: true, force: true });
      } catch {
        // 清理失败不掩盖测试结果。
      }
    },
  };
}

/** 已种 12 名 E2E 球员，ID 1..12，名称 E2E-01..E2E-12。 */
export const PLAYER_IDS = Array.from({ length: 12 }, (_, i) => i + 1);
