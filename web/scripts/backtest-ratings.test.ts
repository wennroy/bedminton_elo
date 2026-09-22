import { existsSync, linkSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { describe, expect, it } from "vitest";
import { parseArgs, run } from "./backtest-ratings";

const asOf = "2026-10-05T00:00:00+08:00";
const seasonStart = "2026-10-01";

describe("backtest-ratings CLI", () => {
  it("prints help without requiring a source or output path", () => {
    const output: string[] = [];
    const result = run(["--help"], { stdout: (message) => output.push(message), stderr: () => undefined });

    expect(result.exitCode).toBe(0);
    expect(output.join("")).toContain("Usage:");
  });

  it("rejects unknown, duplicate, incomplete, and ambiguous flags", () => {
    expect(() => parseArgs(["--wat"])).toThrow(/unknown flag/);
    expect(() => parseArgs(["--fixture", "synthetic", "--fixture", "synthetic"])).toThrow(/duplicate flag/);
    expect(() => parseArgs(["--fixture", "synthetic", "--as-of", asOf])).toThrow(/first-season-start/);
    expect(() =>
      parseArgs([
        "--fixture",
        "synthetic",
        "--db",
        "data.sqlite",
        "--as-of",
        asOf,
        "--first-season-start",
        seasonStart,
        "--out",
        "report.json",
      ])
    ).toThrow(/exactly one source/);
    expect(() =>
      parseArgs(["--fixture", "synthetic", "--as-of", "2026-10-05T00:00:00", "--first-season-start", seasonStart, "--out", "report.json"])
    ).toThrow(/explicit time zone/);
  });

  it("writes a pretty, insufficient-evidence synthetic report", () => {
    const dir = mkdtempSync(join(tmpdir(), "rating-backtest-"));
    const out = join(dir, "report.json");
    try {
      const result = run([
        "--fixture",
        "synthetic",
        "--as-of",
        asOf,
        "--first-season-start",
        seasonStart,
        "--out",
        out,
      ]);

      expect(result.exitCode).toBe(0);
      const written = readFileSync(out, "utf8");
      expect(written.endsWith("\n")).toBe(true);
      expect(JSON.parse(written)).toMatchObject({
        reportVersion: "rating-backtest-v1",
        status: "insufficient_evidence",
        source: { kind: "synthetic", inputVersion: "synthetic-v1" },
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("opens SQLite input strictly read-only and emits no player names", () => {
    const dir = mkdtempSync(join(tmpdir(), "rating-backtest-db-"));
    const dbPath = join(dir, "ratings.sqlite");
    const out = join(dir, "report.json");
    try {
      const db = new Database(dbPath);
      db.exec(`
        CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE matches (
          id INTEGER PRIMARY KEY,
          pa1 INTEGER NOT NULL, pa2 INTEGER NOT NULL, pb1 INTEGER NOT NULL, pb2 INTEGER NOT NULL,
          score_a INTEGER NOT NULL, score_b INTEGER NOT NULL,
          played_at TEXT NOT NULL, created_at TEXT NOT NULL
        );
      `);
      db.prepare("INSERT INTO players (id, name) VALUES (?, ?)").run(1, "private-player-name");
      for (const id of [2, 3, 4]) db.prepare("INSERT INTO players (id, name) VALUES (?, ?)").run(id, `player-${id}`);
      db.prepare(
        "INSERT INTO matches (id, pa1, pa2, pb1, pb2, score_a, score_b, played_at, created_at) VALUES (1, 1, 2, 3, 4, 21, 18, '2026-10-02', '2026-10-02T12:00:00+08:00')"
      ).run();
      db.close();
      const before = readFileSync(dbPath);
      const beforeMtime = statSync(dbPath).mtimeMs;

      const result = run([
        "--db",
        dbPath,
        "--as-of",
        asOf,
        "--first-season-start",
        seasonStart,
        "--out",
        out,
      ]);

      expect(result.exitCode).toBe(0);
      expect(readFileSync(dbPath)).toEqual(before);
      expect(statSync(dbPath).mtimeMs).toBe(beforeMtime);
      expect(existsSync(`${dbPath}-wal`)).toBe(false);
      expect(existsSync(`${dbPath}-shm`)).toBe(false);
      expect(readFileSync(out, "utf8")).not.toContain("private-player-name");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects database output aliases before any write", () => {
    const dir = mkdtempSync(join(tmpdir(), "rating-backtest-output-alias-"));
    const dbPath = join(dir, "ratings.sqlite");
    const hardlinkPath = join(dir, "ratings-hardlink.sqlite");
    const symlinkPath = join(dir, "ratings-symlink.sqlite");
    const danglingSidecarLink = join(dir, "dangling-sidecar-output.json");
    try {
      const db = new Database(dbPath);
      db.exec("CREATE TABLE players (id INTEGER PRIMARY KEY); CREATE TABLE matches (id INTEGER PRIMARY KEY, pa1 INTEGER, pa2 INTEGER, pb1 INTEGER, pb2 INTEGER, score_a INTEGER, score_b INTEGER, played_at TEXT, created_at TEXT);");
      db.close();
      linkSync(dbPath, hardlinkPath);
      symlinkSync(dbPath, symlinkPath);
      symlinkSync(`${dbPath}-wal`, danglingSidecarLink);
      const before = readFileSync(dbPath);
      const beforeMtime = statSync(dbPath).mtimeMs;
      const shared = ["--db", dbPath, "--as-of", asOf, "--first-season-start", seasonStart];

      for (const out of [
        dbPath,
        hardlinkPath,
        symlinkPath,
        danglingSidecarLink,
        `${dbPath}-wal`,
        `${dbPath}-shm`,
        `${dbPath}-journal`,
      ]) {
        const result = run([...shared, "--out", out], { stdout: () => undefined, stderr: () => undefined });
        expect(result.exitCode, out).toBe(1);
        expect(readFileSync(dbPath)).toEqual(before);
        expect(statSync(dbPath).mtimeMs).toBe(beforeMtime);
      }
      expect(existsSync(`${dbPath}-wal`)).toBe(false);
      expect(existsSync(`${dbPath}-shm`)).toBe(false);
      expect(existsSync(`${dbPath}-journal`)).toBe(false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects an output hardlink to an existing database sidecar", () => {
    const dir = mkdtempSync(join(tmpdir(), "rating-backtest-sidecar-hardlink-"));
    const dbPath = join(dir, "ratings.sqlite");
    const walPath = `${dbPath}-wal`;
    const outputHardlink = join(dir, "sidecar-hardlink-output.json");
    try {
      const db = new Database(dbPath);
      db.exec("CREATE TABLE players (id INTEGER PRIMARY KEY); CREATE TABLE matches (id INTEGER PRIMARY KEY, pa1 INTEGER, pa2 INTEGER, pb1 INTEGER, pb2 INTEGER, score_a INTEGER, score_b INTEGER, played_at TEXT, created_at TEXT);");
      db.close();
      writeFileSync(walPath, "sidecar-content", "utf8");
      linkSync(walPath, outputHardlink);
      const sidecarBefore = readFileSync(walPath);

      const result = run(
        ["--db", dbPath, "--as-of", asOf, "--first-season-start", seasonStart, "--out", outputHardlink],
        { stdout: () => undefined, stderr: () => undefined }
      );

      expect(result).toMatchObject({ exitCode: 1, error: "--out must not refer to a database sidecar file" });
      expect(readFileSync(walPath)).toEqual(sidecarBefore);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
