import type Database from "better-sqlite3";
import { createDb, getDatabaseUrl } from "../src/lib/db";

function hasTable(db: Database.Database, name: string): boolean {
  const row = db
    .prepare(
      `SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?`
    )
    .get(name);
  return row !== undefined;
}

function hasColumn(db: Database.Database, table: string, column: string): boolean {
  if (!hasTable(db, table)) return false;
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{
    name: string;
  }>;
  return rows.some((r) => r.name === column);
}

function isMigrated(db: Database.Database): boolean {
  if (!hasTable(db, "meta")) return false;
  const row = db
    .prepare(`SELECT value FROM meta WHERE key = 'legacy_migrated'`)
    .get() as { value: string } | undefined;
  return row?.value === "1";
}

function markMigrated(db: Database.Database): void {
  db.exec(`CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)`);
  db.prepare(
    `INSERT OR REPLACE INTO meta (key, value) VALUES ('legacy_migrated', '1')`
  ).run();
}

function createNewMatchesTable(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS matches (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      pa1 INTEGER NOT NULL,
      pa2 INTEGER NOT NULL,
      pb1 INTEGER NOT NULL,
      pb2 INTEGER NOT NULL,
      score_a INTEGER NOT NULL,
      score_b INTEGER NOT NULL,
      played_at TEXT NOT NULL,
      entered_by INTEGER,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      FOREIGN KEY (pa1) REFERENCES players(id),
      FOREIGN KEY (pa2) REFERENCES players(id),
      FOREIGN KEY (pb1) REFERENCES players(id),
      FOREIGN KEY (pb2) REFERENCES players(id),
      FOREIGN KEY (entered_by) REFERENCES players(id)
    );
  `);
}

export function migrateLegacy(db: Database.Database): void {
  if (isMigrated(db)) {
    console.log("Legacy migration already complete.");
    return;
  }

  const oldTable = "matches";
  const hasOldMatches =
    hasTable(db, oldTable) &&
    hasColumn(db, oldTable, "match_type") &&
    hasColumn(db, oldTable, "player_a1");

  if (!hasOldMatches) {
    console.log("No legacy matches table detected; nothing to migrate.");
    markMigrated(db);
    return;
  }

  db.exec(`
    CREATE TABLE IF NOT EXISTS players (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT UNIQUE NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  `);

  const nameToId = new Map<string, number>();
  if (hasTable(db, "users")) {
    const userRows = db
      .prepare(`SELECT id, name FROM users`)
      .all() as Array<{ id: number; name: string }>;
    for (const { name } of userRows) {
      const result = db
        .prepare(`INSERT OR IGNORE INTO players (name) VALUES (?)`)
        .run(name);
      const idRow = db
        .prepare(`SELECT id FROM players WHERE name = ?`)
        .get(name) as { id: number } | undefined;
      if (idRow) {
        nameToId.set(name, idRow.id);
      }
    }
  }

  const oldRows = db
    .prepare(
      `SELECT match_type, player_a1, player_a2, player_b1, player_b2, score_a, score_b, date FROM ${oldTable}`
    )
    .all() as Array<{
      match_type: string;
      player_a1: string;
      player_a2: string | null;
      player_b1: string;
      player_b2: string | null;
      score_a: number;
      score_b: number;
      date: string;
    }>;

  const skippedSingles: typeof oldRows = [];

  const tx = db.transaction(() => {
    db.exec(`ALTER TABLE ${oldTable} RENAME TO matches_legacy`);
    createNewMatchesTable(db);

    for (const row of oldRows) {
      if (row.match_type !== "双打") {
        skippedSingles.push(row);
        continue;
      }
      const pa1 = nameToId.get(row.player_a1);
      const pa2 = row.player_a2 ? nameToId.get(row.player_a2) : undefined;
      const pb1 = nameToId.get(row.player_b1);
      const pb2 = row.player_b2 ? nameToId.get(row.player_b2) : undefined;
      if (
        pa1 === undefined ||
        pa2 === undefined ||
        pb1 === undefined ||
        pb2 === undefined
      ) {
        console.warn("Skipping row with unknown player:", row);
        continue;
      }
      db.prepare(
        `INSERT INTO matches (pa1, pa2, pb1, pb2, score_a, score_b, played_at, entered_by, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)`
      ).run(
        pa1,
        pa2,
        pb1,
        pb2,
        row.score_a,
        row.score_b,
        row.date,
        `${row.date}T12:00:00`
      );
    }

    markMigrated(db);
  });

  tx();

  if (skippedSingles.length > 0) {
    console.log(
      `WARNING: skipped ${skippedSingles.length} singles matches during legacy migration.`
    );
  }
  console.log(
    `Legacy migration complete. Migrated ${oldRows.length - skippedSingles.length} doubles matches.`
  );
}

function main() {
  const db = createDb(getDatabaseUrl());
  try {
    migrateLegacy(db);
  } finally {
    db.close();
  }
}

if (
  import.meta.url.startsWith("file://") &&
  import.meta.url === `file://${process.argv[1]}`
) {
  main();
}
