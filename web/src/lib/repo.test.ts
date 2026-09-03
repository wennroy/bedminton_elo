import { describe, it, expect, beforeEach } from "vitest";
import Database from "better-sqlite3";
import { tmpdir } from "os";
import { join } from "path";
import {
  addPlayer,
  listPlayers,
  addMatch,
  listMatchesByDate,
  getMatch,
  deleteMatch,
  recomputeAllRatings,
  mergePlayers,
  renamePlayer,
} from "./repo";
import { createDb } from "./db";
import { migrateLegacy } from "../../scripts/migrate-legacy";
import { recomputeElos, INITIAL_RATING } from "./elo";
import { recomputeTrueSkills, TS_MU, TS_SIGMA } from "./trueskill";

function mkDb(): Database.Database {
  const path = join(tmpdir(), `repo-test-${Date.now()}-${Math.random()}.db`);
  return createDb(path);
}

function toEloMatches(matches: ReturnType<typeof listMatchesByDate>) {
  return matches.map((m) => ({
    date: m.playedAt,
    a1: String(m.pa1),
    a2: String(m.pa2),
    b1: String(m.pb1),
    b2: String(m.pb2),
    scoreA: m.scoreA,
    scoreB: m.scoreB,
  }));
}

describe("repo", () => {
  it("creates and lists players", () => {
    const db = mkDb();
    const id = addPlayer("Alice", db);
    expect(id).toBeGreaterThan(0);
    renamePlayer(id, "Alicia", db);
    expect(listPlayers(db)).toEqual([
      { id, name: "Alicia", createdAt: expect.any(String) },
    ]);
  });

  it("adds matches and validates rules", () => {
    const db = mkDb();
    const [a1, a2, b1, b2] = [
      addPlayer("A1", db),
      addPlayer("A2", db),
      addPlayer("B1", db),
      addPlayer("B2", db),
    ];
    expect(() =>
      addMatch({ pa1: a1, pa2: a2, pb1: a1, pb2: b2, scoreA: 21, scoreB: 18, playedAt: "2024-01-01" }, db)
    ).toThrow("distinct");
    expect(() =>
      addMatch({ pa1: a1, pa2: a2, pb1: b1, pb2: b2, scoreA: 21, scoreB: 21, playedAt: "2024-01-01" }, db)
    ).toThrow("equal");
    const id = addMatch({ pa1: a1, pa2: a2, pb1: b1, pb2: b2, scoreA: 21, scoreB: 18, playedAt: "2024-01-01" }, db);
    const all = listMatchesByDate(db);
    expect(all).toHaveLength(1);
    expect(getMatch(id, db)).toMatchObject({ pa1Name: "A1", pb2Name: "B2", scoreA: 21 });
  });

  it("recomputes ratings consistently with pure functions", () => {
    const db = mkDb();
    const [a1, a2, b1, b2] = [
      addPlayer("A1", db),
      addPlayer("A2", db),
      addPlayer("B1", db),
      addPlayer("B2", db),
    ];
    addMatch({ pa1: a1, pa2: a2, pb1: b1, pb2: b2, scoreA: 21, scoreB: 18, playedAt: "2024-01-01" }, db);
    addMatch({ pa1: b1, pa2: b2, pb1: a1, pb2: a2, scoreA: 21, scoreB: 19, playedAt: "2024-01-02" }, db);

    const matches = listMatchesByDate(db);
    const eloMatches = toEloMatches(matches);
    const { ratings: eloRatings } = recomputeElos(eloMatches);
    const { players: tsPlayers } = recomputeTrueSkills(eloMatches);

    const ratings = recomputeAllRatings(db);
    for (const id of [a1, a2, b1, b2]) {
      expect(ratings.get(id)).toEqual({
        elo: eloRatings[String(id)] ?? INITIAL_RATING,
        mu: tsPlayers[String(id)]?.mu ?? TS_MU,
        sigma: tsPlayers[String(id)]?.sigma ?? TS_SIGMA,
      });
    }
  });

  it("deletes a match", () => {
    const db = mkDb();
    const [a1, a2, b1, b2] = [
      addPlayer("A1", db),
      addPlayer("A2", db),
      addPlayer("B1", db),
      addPlayer("B2", db),
    ];
    const id = addMatch({ pa1: a1, pa2: a2, pb1: b1, pb2: b2, scoreA: 21, scoreB: 18, playedAt: "2024-01-01" }, db);
    deleteMatch(id, db);
    expect(listMatchesByDate(db)).toHaveLength(0);
  });

  it("merges players", () => {
    const db = mkDb();
    const [p1, p2, p3, p4, p5] = [
      addPlayer("P1", db),
      addPlayer("P2", db),
      addPlayer("P3", db),
      addPlayer("P4", db),
      addPlayer("P5", db),
    ];
    addMatch({ pa1: p1, pa2: p2, pb1: p3, pb2: p4, scoreA: 21, scoreB: 18, playedAt: "2024-01-01", enteredBy: p1 }, db);
    mergePlayers(p1, p5, db);
    expect(listPlayers(db).map((p) => p.name).sort()).toEqual(["P2", "P3", "P4", "P5"]);
    const m = listMatchesByDate(db)[0];
    expect(m.pa1).toBe(p5);
    expect(m.enteredBy).toBe(p5);
  });

  it("migrates legacy schema idempotently", () => {
    const db = mkDb();
    // Simulate the real legacy db: createDb() already ran schema.sql, but the
    // legacy tables (incl. a `players` table with a DIFFERENT shape) shadow it.
    db.exec(`DROP TABLE IF EXISTS matches`);
    // signups 引用 players(id)，必须先于 players drop，否则 FK mismatch
    db.exec(`DROP TABLE IF EXISTS signups`);
    db.exec(`DROP TABLE IF EXISTS players`);
    db.exec(`
      CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
      );
      CREATE TABLE players (
        user_id INTEGER PRIMARY KEY,
        elo REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
      );
      CREATE TABLE players_trueskill (user_id INTEGER PRIMARY KEY, mu REAL, sigma REAL);
      CREATE TABLE pending_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_a1 INTEGER, player_a2 INTEGER, player_b1 INTEGER, player_b2 INTEGER,
        score_a INTEGER, score_b INTEGER, submitted BOOLEAN DEFAULT FALSE
      );
      CREATE TABLE optimization_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        seed INTEGER, alpha_var REAL, best_loss REAL,
        mean_closeness REAL, max_closeness REAL, lambda_val REAL, entropy REAL
      );
      CREATE TABLE matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_type TEXT,
        player_a1 TEXT,
        player_a2 TEXT,
        player_b1 TEXT,
        player_b2 TEXT,
        score_a INTEGER,
        score_b INTEGER,
        date TEXT
      );
      INSERT INTO users (name) VALUES ('张伟'), ('李娜'), ('王强'), ('刘洋');
      INSERT INTO players (user_id, elo) VALUES (1, 1024.5), (2, 990.0), (3, 1010.2), (4, 975.3);
      INSERT INTO players_trueskill (user_id, mu, sigma) VALUES (1, 26.0, 7.5), (2, 24.0, 8.0);
      INSERT INTO matches (match_type, player_a1, player_a2, player_b1, player_b2, score_a, score_b, date) VALUES
        ('双打', '张伟', '李娜', '王强', '刘洋', 21, 18, '2024-01-01'),
        ('单打', '张伟', NULL, '王强', NULL, 21, 19, '2024-01-02');
    `);

    migrateLegacy(db);
    const migrated = listMatchesByDate(db);
    expect(migrated).toHaveLength(1);
    expect(migrated[0].pa1Name).toBe("张伟");
    expect(migrated[0].playedAt).toBe("2024-01-01");

    // New players table must have the new shape (id, name) — regression: the
    // legacy players(user_id, elo) table used to shadow it and crash the insert.
    const playerCols = db.prepare(`PRAGMA table_info(players)`).all() as Array<{ name: string }>;
    expect(playerCols.map((c) => c.name)).toContain("name");
    const players = db.prepare(`SELECT name FROM players ORDER BY name`).all() as Array<{ name: string }>;
    expect(players.map((p) => p.name)).toEqual(["刘洋", "张伟", "李娜", "王强"]);

    const legacy = db.prepare(`SELECT COUNT(*) AS c FROM matches_legacy`).get() as { c: number };
    expect(legacy.c).toBe(2);
    // All legacy tables preserved under renamed names.
    for (const t of ["users_legacy", "players_legacy", "players_trueskill_legacy", "pending_matches_legacy", "optimization_results_legacy"]) {
      expect(
        db.prepare(`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?`).get(t)
      ).toBeDefined();
    }

    migrateLegacy(db);
    expect(listMatchesByDate(db)).toHaveLength(1);
  });
});
