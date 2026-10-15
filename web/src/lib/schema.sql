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

CREATE TABLE IF NOT EXISTS schedule_projects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  model TEXT NOT NULL,
  revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
  next_fixture_no INTEGER NOT NULL CHECK (next_fixture_no >= 1),
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  create_request_id TEXT UNIQUE,
  create_payload_hash TEXT,
  -- 引用 players 主键（INTEGER PRIMARY KEY 即 rowid）。写表名不写明列，
  -- 语义等价于 players(id)，但 players 被旧库迁移脚本换形状 shadow 时
  -- 仍可按 rowid 解析，避免 SQLite 编译期 foreign key mismatch
  -- （repo.test.ts 的 legacy 模拟会 drop 并以 user_id 形状重建 players）。
  FOREIGN KEY (created_by) REFERENCES players
);

CREATE TABLE IF NOT EXISTS schedule_project_players (
  project_id INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  active INTEGER NOT NULL CHECK (active IN (0, 1)),
  fairness_base INTEGER NOT NULL DEFAULT 0 CHECK (fairness_base >= 0),
  played_at_join INTEGER NOT NULL DEFAULT 0 CHECK (played_at_join >= 0),
  joined_at TEXT,
  left_at TEXT,
  PRIMARY KEY (project_id, player_id),
  FOREIGN KEY (project_id) REFERENCES schedule_projects(id),
  FOREIGN KEY (player_id) REFERENCES players
);

CREATE TABLE IF NOT EXISTS schedule_generations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER NOT NULL,
  request_id TEXT NOT NULL,
  payload_hash TEXT,
  source_revision INTEGER NOT NULL CHECK (source_revision >= 0),
  snapshot_schema_version INTEGER NOT NULL,
  input_json TEXT NOT NULL,
  result_metrics_json TEXT,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (project_id, request_id),
  UNIQUE (id, project_id),
  FOREIGN KEY (project_id) REFERENCES schedule_projects(id),
  FOREIGN KEY (created_by) REFERENCES players
);

CREATE TABLE IF NOT EXISTS schedule_fixtures (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  project_id INTEGER NOT NULL,
  generation_id INTEGER NOT NULL,
  display_no INTEGER NOT NULL,
  pa1 TEXT NOT NULL,
  pa2 TEXT NOT NULL,
  pb1 TEXT NOT NULL,
  pb2 TEXT NOT NULL,
  generated_win_rate REAL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'played_unscored', 'recorded', 'superseded')),
  revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
  played_at TEXT,
  marked_at TEXT,
  marked_by TEXT,
  match_id INTEGER UNIQUE,
  UNIQUE (project_id, display_no),
  CHECK (pa1 <> pa2),
  CHECK (pa1 <> pb1),
  CHECK (pa1 <> pb2),
  CHECK (pa2 <> pb1),
  CHECK (pa2 <> pb2),
  CHECK (pb1 <> pb2),
  CHECK (
    (status = 'recorded' AND match_id IS NOT NULL)
    OR (status <> 'recorded' AND match_id IS NULL)
  ),
  CHECK (
    (status IN ('played_unscored', 'recorded') AND played_at IS NOT NULL)
    OR (status IN ('pending', 'superseded') AND played_at IS NULL)
  ),
  FOREIGN KEY (project_id) REFERENCES schedule_projects(id),
  FOREIGN KEY (generation_id, project_id) REFERENCES schedule_generations (id, project_id),
  FOREIGN KEY (match_id) REFERENCES matches(id),
  FOREIGN KEY (marked_by) REFERENCES players
);

CREATE INDEX IF NOT EXISTS idx_schedule_fixtures_project_status
  ON schedule_fixtures (project_id, status);

CREATE INDEX IF NOT EXISTS idx_schedule_fixtures_match_id
  ON schedule_fixtures (match_id);

CREATE INDEX IF NOT EXISTS idx_schedule_generations_project
  ON schedule_generations (project_id);

CREATE INDEX IF NOT EXISTS idx_schedule_project_players_project
  ON schedule_project_players (project_id, active);
