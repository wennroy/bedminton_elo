import { describe, expect, it, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { RATING_CONFIG_META_KEY } from "../src/lib/rating-config";
import { parseArgs, run } from "./rating-config";

let dir: string;
let dbPath: string;
let stdout: string[];
let stderr: string[];

function io() {
  stdout = [];
  stderr = [];
  return {
    stdout: (message: string) => stdout.push(message),
    stderr: (message: string) => stderr.push(message),
  };
}

function output(): string {
  return stdout.join("");
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "rating-config-cli-"));
  dbPath = join(dir, "test.db");
  stdout = [];
  stderr = [];
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
  delete process.env.ADMIN_PASSWORD;
});

function metaCount(): number {
  const db = new Database(dbPath, { readonly: true, fileMustExist: true });
  try {
    const row = db
      .prepare(`SELECT COUNT(*) AS c FROM meta WHERE key = ?`)
      .get(RATING_CONFIG_META_KEY) as { c: number };
    return row.c;
  } finally {
    db.close();
  }
}

describe("rating-config CLI", () => {
  it("prints help without opening a database", () => {
    const result = run(["--help"], io());
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("Usage:");
    expect(output()).toContain("init");
    expect(output()).toContain("activate");
  });

  it("rejects unknown commands and flags, missing activate model, and bad numeric values", () => {
    expect(() => parseArgs(["--wat"])).toThrow(/first argument/);
    expect(() => parseArgs(["init", "--wat"])).toThrow(/unknown flag/);
    expect(() => parseArgs(["init", "--tau", "abc"])).toThrow(/finite number/);
    expect(() => parseArgs(["activate", "--db", dbPath])).toThrow(/requires --model/);
    expect(() => parseArgs(["activate", "--db", dbPath, "--model", "trueskill"])).toThrow(
      /glicko2 or legacy/
    );
    expect(() => parseArgs(["status", "--db", dbPath, "--tau", "0.4"])).toThrow(
      /only valid with the init command/
    );
  });

  it("init 写入默认 legacy 配置，status 显示字段", () => {
    const init = run(["init", "--db", dbPath], io());
    expect(init.exitCode).toBe(0);
    expect(output()).toContain("initialized rating config");
    expect(output()).toContain("activeModel=legacy");

    const status = run(["status", "--db", dbPath], io());
    expect(status.exitCode).toBe(0);
    expect(output()).toContain("rating config: initialized");
    expect(output()).toContain("paramsVersion: p1");
  });

  it("status 在未初始化时提示维持 Legacy 且不写库", () => {
    const result = run(["status", "--db", dbPath], io());
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("not initialized");
    expect(output()).toContain("Legacy");
  });

  it("init 支持显式 --first-season-start，非法日期拒绝且不写库", () => {
    const ok = run(["init", "--db", dbPath, "--first-season-start", "2026-07-01"], io());
    expect(ok.exitCode).toBe(0);
    expect(ok.record!.config.firstSeasonStart).toBe("2026-07-01");

    rmSync(dbPath);
    const bad = run(["init", "--db", dbPath, "--first-season-start", "2026-10-02"], io());
    expect(bad.exitCode).toBe(1);
    expect(bad.error).toMatch(/quarter start/);
  });

  it("重复 init 不覆写旧配置", () => {
    run(["init", "--db", dbPath, "--first-season-start", "2026-10-01"], io());
    const before = output();
    const again = run(["init", "--db", dbPath], io());
    expect(again.exitCode).toBe(0);
    expect(output()).toContain("no changes");
    expect(again.record!.config.firstSeasonStart).toBe("2026-10-01");
    expect(before).not.toContain("replaced");
  });

  it("同 paramsVersion 参数不同拒绝；新 paramsVersion 替换且保留 activeModel", () => {
    run(["init", "--db", dbPath], io());
    const conflict = run(["init", "--db", dbPath, "--tau", "0.4"], io());
    expect(conflict.exitCode).toBe(1);
    expect(conflict.error).toMatch(/already initialized with different parameters/);

    run(["activate", "--db", dbPath, "--model", "glicko2"], io());
    const bumped = run(
      ["init", "--db", dbPath, "--params-version", "p2", "--tau", "0.4"],
      io()
    );
    expect(bumped.exitCode).toBe(0);
    expect(output()).toContain("p1 -> p2");
    expect(bumped.record!.activeModel).toBe("glicko2");
    expect(bumped.record!.config.tau).toBe(0.4);
  });

  it("activate 切换模型；未初始化时报错", () => {
    const early = run(["activate", "--db", dbPath, "--model", "glicko2"], io());
    expect(early.exitCode).toBe(1);
    expect(early.error).toMatch(/not initialized/);

    run(["init", "--db", dbPath], io());
    const activated = run(["activate", "--db", dbPath, "--model", "glicko2"], io());
    expect(activated.exitCode).toBe(0);
    expect(output()).toContain("glicko2");
    const status = run(["status", "--db", dbPath], io());
    expect(output()).toContain("activeModel: glicko2");
  });

  it("init --dry-run 输出差异且数据库无写入", () => {
    const result = run(
      ["init", "--db", dbPath, "--params-version", "p2", "--tau", "0.4", "--dry-run"],
      io()
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("paramsVersion = p2");
    expect(output()).toContain("tau = 0.4");
    expect(output()).toContain("dry-run: no changes written");
    expect(metaCount()).toBe(0);
  });

  it("已有配置时 --dry-run 输出旧值到新值的差异且不写入", () => {
    run(["init", "--db", dbPath], io());
    const before = metaCount();
    const result = run(
      ["init", "--db", dbPath, "--params-version", "p2", "--tau", "0.4", "--dry-run"],
      io()
    );
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("paramsVersion: p1 -> p2");
    expect(output()).toContain("tau: 0.3 -> 0.4");
    expect(metaCount()).toBe(before);
    // dry-run 之后真实配置仍是 p1
    const status = run(["status", "--db", dbPath], io());
    expect(output()).toContain("paramsVersion: p1");
  });

  it("activate --dry-run 输出模型差异且不写入", () => {
    run(["init", "--db", dbPath], io());
    const result = run(["activate", "--db", dbPath, "--model", "glicko2", "--dry-run"], io());
    expect(result.exitCode).toBe(0);
    expect(output()).toContain("activeModel: legacy -> glicko2");
    const status = run(["status", "--db", dbPath], io());
    expect(output()).toContain("activeModel: legacy");
  });

  it("输出不包含无关环境变量值", () => {
    process.env.ADMIN_PASSWORD = "sentinel-admin-secret-123";
    run(["init", "--db", dbPath], io());
    run(["status", "--db", dbPath], io());
    run(["activate", "--db", dbPath, "--model", "glicko2"], io());
    expect(output()).not.toContain("sentinel-admin-secret-123");
    expect(stderr.join("")).not.toContain("sentinel-admin-secret-123");
  });
});
