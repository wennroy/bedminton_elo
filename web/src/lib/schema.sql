CREATE TABLE IF NOT EXISTS players (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

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

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_date TEXT NOT NULL,
  player_id INTEGER NOT NULL,
  party_size INTEGER NOT NULL DEFAULT 1 CHECK (party_size IN (1, 2)),
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (session_date, player_id),
  FOREIGN KEY (player_id) REFERENCES players(id)
);
