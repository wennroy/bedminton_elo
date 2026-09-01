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

// All table names used by the legacy Streamlit app. They collide with the new
// schema (`players` exists in both with different shapes), so every one of them
// must be renamed out of the way BEFORE creating the new tables.
const LEGACY_TABLES = [
  "matches",
  "users",
  "players",
  "players_trueskill",
  "pending_matches",
  "optimization_results",
] as const;

export function migrateLegacy(db: Database.Database): void {
  if (isMigrated(db)) {
    console.log("Legacy migration already complete.");
    return;
  }

  const hasOldMatches =
    hasTable(db, "matches") &&
    hasColumn(db, "matches", "match_type") &&
    hasColumn(db, "matches", "player_a1");

  if (!hasOldMatches) {
    console.log("No legacy matches table detected; nothing to migrate.");
    markMigrated(db);
    return;
  }

  let migratedCount = 0;
  let skippedSingles = 0;

  const tx = db.transaction(() => {
    // 1. Rename every legacy table out of the way first — legacy `players`
    //    (user_id, elo) would otherwise shadow the new `players`(id, name).
    for (const t of LEGACY_TABLES) {
      if (hasTable(db, t)) {
        db.exec(`ALTER TABLE ${t} RENAME TO ${t}_legacy`);
      }
    }

    // 2. Create the new schema fresh.
    db.exec(`
      CREATE TABLE players (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
    `);
    createNewMatchesTable(db);

    // 3. Seed players from users_legacy, plus any names appearing in
    //    matches_legacy but missing from users (insurance against dropped rows).
    const insertPlayer = db.prepare(
      `INSERT OR IGNORE INTO players (name) VALUES (?)`
    );
    if (hasTable(db, "users_legacy")) {
      const userRows = db
        .prepare(`SELECT name FROM users_legacy`)
        .all() as Array<{ name: string }>;
      for (const { name } of userRows) {
        insertPlayer.run(name);
      }
    }

    const oldRows = db
      .prepare(
        `SELECT match_type, player_a1, player_a2, player_b1, player_b2, score_a, score_b, date FROM matches_legacy`
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

    for (const row of oldRows) {
      for (const n of [row.player_a1, row.player_a2, row.player_b1, row.player_b2]) {
        if (n) insertPlayer.run(n);
      }
    }

    const nameToId = new Map<string, number>();
    const playerRows = db
      .prepare(`SELECT id, name FROM players`)
      .all() as Array<{ id: number; name: string }>;
    for (const { id, name } of playerRows) {
      nameToId.set(name, id);
    }

    // 4. Copy doubles matches into the new table.
    const insertMatch = db.prepare(
      `INSERT INTO matches (pa1, pa2, pb1, pb2, score_a, score_b, played_at, entered_by, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)`
    );
    for (const row of oldRows) {
      if (row.match_type !== "双打") {
        skippedSingles += 1;
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
        throw new Error(
          `Legacy match references unknown player, aborting: ${JSON.stringify(row)}`
        );
      }
      insertMatch.run(
        pa1,
        pa2,
        pb1,
        pb2,
        row.score_a,
        row.score_b,
        row.date,
        `${row.date}T12:00:00`
      );
      migratedCount += 1;
    }

    markMigrated(db);
  });

  tx();

  if (skippedSingles > 0) {
    console.log(
      `WARNING: skipped ${skippedSingles} singles matches during legacy migration.`
    );
  }
  console.log(
    `Legacy migration complete. Migrated ${migratedCount} doubles matches.`
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
