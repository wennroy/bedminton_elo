import { describe, it, expect } from "vitest";
import Database from "better-sqlite3";
import { readFileSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createDb } from "./db";
import { addPlayer, addMatch } from "./repo";
import {
  createProject,
  regenerate,
  listProjects,
  getProject,
  markPlayed,
  attachScore,
  detachScore,
  ScheduleProjectError,
  isScheduleProjectError,
  SNAPSHOT_SCHEMA_VERSION,
} from "./schedule-projects";
import type { SchedulingResult } from "./scheduling/types";

const __dirname = dirname(fileURLToPath(import.meta.url));

/** T3 改动前的旧 schema，用于验证旧库升级幂等。 */
const OLD_SCHEMA = `
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
`;

function mkDb(): Database.Database {
  const path = join(
    tmpdir(),
    `schedule-projects-test-${Date.now()}-${Math.random()}.db`
  );
  return createDb(path);
}

function seedPlayers(db: Database.Database, n: number): string[] {
  const ids: string[] = [];
  for (let i = 1; i <= n; i++) {
    ids.push(String(addPlayer(`SP-P${i}-${Math.random().toString(36).slice(2, 8)}`, db)));
  }
  return ids;
}

function engineResult(
  schedule: Array<[string, string, string, string]>
): SchedulingResult {
  return {
    schedule: schedule.map(([a1, a2, b1, b2]) => ({ a1, a2, b1, b2 })),
    metrics: {
      J: 0,
      imbalancedMatchCount: 0,
      totalImbalanceExcess: 0,
      directRepeatCount: 0,
    },
    algorithmVersion: "quota-local-v1",
    objectiveVersion: "balanced-v1",
    seed: 42,
    quotas: {},
    fairnessF: 0,
  };
}

function createWith(
  db: Database.Database,
  playerIds: string[],
  schedule: Array<[string, string, string, string]>,
  overrides: Partial<Parameters<typeof createProject>[0]> = {}
) {
  return createProject(
    {
      name: "周五场",
      model: "glicko2",
      createdBy: playerIds[0],
      requestId: `req-${Math.random().toString(36).slice(2, 10)}`,
      payloadHash: "hash-1",
      playerIds,
      engineInput: { playerIds, matches: schedule.length, seed: 1 },
      ...overrides,
    },
    engineResult(schedule),
    db
  );
}

function expectErrorCode(fn: () => unknown, code: ScheduleProjectError["code"]) {
  try {
    fn();
  } catch (err) {
    expect(isScheduleProjectError(err, code)).toBe(true);
    expect((err as ScheduleProjectError).code).toBe(code);
    return err as ScheduleProjectError;
  }
  throw new Error(`期望抛出 ${code}，但没有抛错`);
}

function raw<T>(db: Database.Database, sql: string, ...params: unknown[]): T {
  return db.prepare(sql).get(...(params as never[])) as T;
}

function rawAll<T>(db: Database.Database, sql: string, ...params: unknown[]): T[] {
  return db.prepare(sql).all(...(params as never[])) as T[];
}

function count(db: Database.Database, table: string): number {
  return raw<{ c: number }>(db, `SELECT COUNT(*) AS c FROM ${table}`).c;
}

// ---------------------------------------------------------------------------

describe("schema 初始化与外键", () => {
  it("空库初始化两次幂等，PRAGMA foreign_keys 生效，foreign_key_check 无行", () => {
    const path = join(
      tmpdir(),
      `schedule-projects-empty-${Date.now()}-${Math.random()}.db`
    );
    const db1 = createDb(path);
    db1.close();
    const db = createDb(path);
    expect(db.pragma("foreign_keys", { simple: true })).toBe(1);
    expect(db.pragma("foreign_key_check")).toEqual([]);
    // 手动再执行一次 schema（等价二次启动），不报错。
    const schema = readFileSync(join(__dirname, "schema.sql"), "utf-8");
    db.exec(schema);
    db.exec(schema);
    expect(db.pragma("foreign_key_check")).toEqual([]);
    db.close();
  });

  it("旧 schema 库升级：旧数据保留，新表补齐，重复执行幂等", () => {
    const path = join(
      tmpdir(),
      `schedule-projects-old-${Date.now()}-${Math.random()}.db`
    );
    const legacy = new Database(path);
    legacy.exec(OLD_SCHEMA);
    const p1 = Number(
      legacy.prepare(`INSERT INTO players (name) VALUES ('旧球员')`).run()
        .lastInsertRowid
    );
    legacy
      .prepare(
        `INSERT INTO matches (pa1, pa2, pb1, pb2, score_a, score_b, played_at)
         VALUES (?, ?, ?, ?, 21, 18, '2024-01-01')`
      )
      .run(p1, p1, p1, p1);
    legacy.close();

    const db = createDb(path);
    expect(db.pragma("foreign_keys", { simple: true })).toBe(1);
    expect(count(db, "players")).toBe(1);
    expect(count(db, "matches")).toBe(1);
    for (const t of [
      "schedule_projects",
      "schedule_project_players",
      "schedule_generations",
      "schedule_fixtures",
    ]) {
      const row = raw<{ n: number }>(
        db,
        `SELECT COUNT(*) AS n FROM sqlite_master WHERE type='table' AND name=?`,
        t
      );
      expect(row.n).toBe(1);
    }
    const schema = readFileSync(join(__dirname, "schema.sql"), "utf-8");
    db.exec(schema);
    db.exec(schema);
    expect(db.pragma("foreign_key_check")).toEqual([]);
    db.close();
  });
});

describe("createProject", () => {
  it("全字段落库：项目、成员、generation、fixtures", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
      [players[0], players[2], players[1], players[3]],
    ]);

    const project = raw<Record<string, unknown>>(
      db,
      `SELECT * FROM schedule_projects WHERE id = ?`,
      created.id
    );
    expect(project).toMatchObject({
      id: created.id,
      name: "周五场",
      model: "glicko2",
      revision: 0,
      next_fixture_no: 3,
      create_request_id: expect.any(String),
      create_payload_hash: "hash-1",
    });

    const members = rawAll<{ player_id: number; active: number; fairness_base: number; played_at_join: number; joined_at: string | null }>(
      db,
      `SELECT player_id, active, fairness_base, played_at_join, joined_at
       FROM schedule_project_players WHERE project_id = ? ORDER BY player_id`,
      created.id
    );
    expect(members).toHaveLength(4);
    for (const [i, m] of members.entries()) {
      expect(m).toMatchObject({
        player_id: Number(players[i]),
        active: 1,
        fairness_base: 0,
        played_at_join: 0,
      });
      expect(m.joined_at).toEqual(expect.any(String));
    }

    const gen = raw<{
      source_revision: number;
      snapshot_schema_version: number;
      input_json: string;
      result_metrics_json: string;
      created_by: string;
    }>(
      db,
      `SELECT source_revision, snapshot_schema_version, input_json, result_metrics_json, created_by
       FROM schedule_generations WHERE project_id = ?`,
      created.id
    );
    expect(gen.source_revision).toBe(0);
    expect(gen.snapshot_schema_version).toBe(SNAPSHOT_SCHEMA_VERSION);
    expect(JSON.parse(gen.input_json)).toEqual({
      playerIds: players,
      matches: 2,
      seed: 1,
    });
    expect(JSON.parse(gen.result_metrics_json)).toEqual({
      J: 0,
      imbalancedMatchCount: 0,
      totalImbalanceExcess: 0,
      directRepeatCount: 0,
    });
    expect(gen.created_by).toBe(players[0]);

    const fixtures = rawAll<{ display_no: number; status: string; pa1: string }>(
      db,
      `SELECT display_no, status, pa1 FROM schedule_fixtures
       WHERE project_id = ? ORDER BY display_no`,
      created.id
    );
    expect(fixtures.map((f) => f.display_no)).toEqual([1, 2]);
    expect(fixtures.every((f) => f.status === "pending")).toBe(true);
  });

  it("同 requestId 同 payloadHash 重放返回已建项目，不重复插入", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const first = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    const second = createProject(
      {
        name: "改名也应被忽略",
        model: "glicko2",
        createdBy: players[0],
        requestId: raw<{ create_request_id: string }>(
          db,
          `SELECT create_request_id FROM schedule_projects WHERE id = ?`,
          first.id
        ).create_request_id,
        payloadHash: "hash-1",
        playerIds: players,
      },
      engineResult([[players[0], players[1], players[2], players[3]]]),
      db
    );
    expect(second).toEqual({ id: first.id, revision: 0, replayed: true });
    expect(count(db, "schedule_projects")).toBe(1);
    expect(count(db, "schedule_project_players")).toBe(4);
    expect(count(db, "schedule_generations")).toBe(1);
    expect(count(db, "schedule_fixtures")).toBe(1);
  });

  it("同 requestId 不同 payload → idempotency_conflict", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    createProject(
      {
        name: "p",
        model: "m",
        createdBy: players[0],
        requestId: "req-dup",
        payloadHash: "hash-A",
        playerIds: players,
      },
      undefined,
      db
    );
    expectErrorCode(
      () =>
        createProject(
          {
            name: "p",
            model: "m",
            createdBy: players[0],
            requestId: "req-dup",
            payloadHash: "hash-B",
            playerIds: players,
          },
          undefined,
          db
        ),
      "idempotency_conflict"
    );
  });

  it("事务中途失败 → 全库零写", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    // 阵容非法（a1==a2）触发 CHECK 约束，整个事务必须回滚。
    expect(() =>
      createProject(
        {
          name: "坏项目",
          model: "m",
          createdBy: players[0],
          requestId: "req-bad",
          payloadHash: "h",
          playerIds: players,
        },
        engineResult([[players[0], players[0], players[2], players[3]]]),
        db
      )
    ).toThrow();
    expect(count(db, "schedule_projects")).toBe(0);
    expect(count(db, "schedule_project_players")).toBe(0);
    expect(count(db, "schedule_generations")).toBe(0);
    expect(count(db, "schedule_fixtures")).toBe(0);
  });

  it("目录外创建者/成员球员被 FK 或存在性校验拒绝，零写", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    expectErrorCode(
      () =>
        createProject(
          {
            name: "p",
            model: "m",
            createdBy: "99999",
            requestId: "req-x1",
            payloadHash: "h",
            playerIds: players,
          },
          undefined,
          db
        ),
      "player_not_found"
    );
    expectErrorCode(
      () =>
        createProject(
          {
            name: "p",
            model: "m",
            createdBy: players[0],
            requestId: "req-x2",
            payloadHash: "h",
            playerIds: [...players, "99998"],
          },
          undefined,
          db
        ),
      "player_not_found"
    );
    expect(count(db, "schedule_projects")).toBe(0);
  });

  it("参数严格校验：非法整数/球员 ID 形状/名单重复", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const base = {
      name: "p",
      model: "m",
      createdBy: players[0],
      requestId: "req-v",
      payloadHash: "h",
      playerIds: players,
    };
    expectErrorCode(
      () => createProject({ ...base, createdBy: "0" }, undefined, db),
      "validation_error"
    );
    expectErrorCode(
      () => createProject({ ...base, createdBy: "01" }, undefined, db),
      "validation_error"
    );
    expectErrorCode(
      () => createProject({ ...base, createdBy: "abc" }, undefined, db),
      "validation_error"
    );
    expectErrorCode(
      () =>
        createProject(
          { ...base, requestId: "req-v2", playerIds: [players[0], players[0]] },
          undefined,
          db
        ),
      "validation_error"
    );
    expectErrorCode(
      () =>
        createProject(
          { ...base, requestId: "req-v3", playerIds: [] },
          undefined,
          db
        ),
      "validation_error"
    );
    expect(count(db, "schedule_projects")).toBe(0);
  });
});

describe("regenerate", () => {
  it("已打保留不变，旧 pending 变 superseded，display_no 不复用", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
      [players[0], players[2], players[1], players[3]],
      [players[0], players[3], players[1], players[2]],
    ]);
    const detail = getProject(created.id, db);
    const playedFixture = detail.fixtures[0];
    const marked = markPlayed(
      {
        projectId: created.id,
        fixtureId: playedFixture.id,
        expectedFixtureRevision: playedFixture.revision,
        actorId: players[0],
        playedAt: "2026-10-01",
      },
      db
    );
    expect(marked.status).toBe("played_unscored");

    // markPlayed 已把项目 revision 推到 1；续排 CAS 必须基于最新 revision。
    const result = regenerate(
      {
        projectId: created.id,
        expectedRevision: 1,
        requestId: "req-regen-1",
        payloadHash: "hash-r1",
        newPlayerIds: players,
        engineInput: { matches: 2 },
        engineResult: engineResult([
          [players[1], players[2], players[0], players[3]],
          [players[1], players[3], players[0], players[2]],
        ]),
        actorId: players[0],
      },
      db
    );

    expect(result.replayed).toBe(false);
    expect(result.revision).toBe(2);
    expect(result.supersededCount).toBe(2);
    expect(result.fixtures.map((f) => f.displayNo)).toEqual([4, 5]);
    expect(result.fixtures.every((f) => f.status === "pending")).toBe(true);

    const after = getProject(created.id, db);
    // 已打场次原样保留（ID/阵容/日期不变）。
    const kept = after.fixtures.find((f) => f.id === playedFixture.id);
    expect(kept).toMatchObject({
      status: "played_unscored",
      playedAt: "2026-10-01",
      pa1: playedFixture.pa1,
      pa2: playedFixture.pa2,
      pb1: playedFixture.pb1,
      pb2: playedFixture.pb2,
    });
    expect(after.fixtures).toHaveLength(3); // 1 played_unscored + 2 pending
    expect(after.superseded).toHaveLength(2);
    expect(after.superseded.map((f) => f.displayNo).sort((a, b) => a - b)).toEqual([
      2, 3,
    ]);
    expect(after.nextFixtureNo).toBe(6);
  });

  it("revision CAS 失败 → revision_conflict 并携带当前 revision，项目不动", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    const err = expectErrorCode(
      () =>
        regenerate(
          {
            projectId: created.id,
            expectedRevision: 7,
            requestId: "req-cas",
            payloadHash: "h",
            newPlayerIds: players,
            engineInput: {},
            engineResult: engineResult([
              [players[0], players[1], players[2], players[3]],
            ]),
            actorId: players[0],
          },
          db
        ),
      "revision_conflict"
    );
    expect(err.currentRevision).toBe(0);
    expect(raw<{ revision: number }>(db, `SELECT revision FROM schedule_projects WHERE id=?`, created.id).revision).toBe(0);
    expect(count(db, "schedule_generations")).toBe(1); // 只有创建那一代
  });

  it("幂等重放：同 requestId 同 hash 返回既有 generation，hash 不同抛冲突", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    const input = {
      projectId: created.id,
      expectedRevision: 0,
      requestId: "req-idem",
      payloadHash: "h-same",
      newPlayerIds: players,
      engineInput: { matches: 1 },
      engineResult: engineResult([
        [players[1], players[2], players[0], players[3]],
      ]),
      actorId: players[0],
    };
    const first = regenerate(input, db);
    expect(first.replayed).toBe(false);
    const second = regenerate(input, db);
    expect(second.replayed).toBe(true);
    expect(second.generationId).toBe(first.generationId);
    expect(second.fixtures.map((f) => f.id)).toEqual(
      first.fixtures.map((f) => f.id)
    );
    expect(second.supersededCount).toBe(0);
    expect(count(db, "schedule_generations")).toBe(2);
    // CAS 过期也不影响重放语义（先幂等后 CAS）。
    expect(
      regenerate({ ...input, expectedRevision: 99 }, db).replayed
    ).toBe(true);
    expectErrorCode(
      () => regenerate({ ...input, payloadHash: "h-other" }, db),
      "idempotency_conflict"
    );
  });

  it("matches=0：只保存名单变更、清空未打、不产生 fixtures", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
      [players[0], players[2], players[1], players[3]],
    ]);
    const result = regenerate(
      {
        projectId: created.id,
        expectedRevision: 0,
        requestId: "req-zero",
        payloadHash: "h0",
        newPlayerIds: players.slice(0, 3),
        engineInput: { matches: 0 },
        engineResult: null,
        actorId: players[0],
      },
      db
    );
    expect(result.fixtures).toEqual([]);
    expect(result.supersededCount).toBe(2);
    expect(result.leftPlayers).toEqual([players[3]]);
    expect(result.revision).toBe(1);

    const detail = getProject(created.id, db);
    expect(detail.fixtures).toEqual([]);
    expect(detail.superseded).toHaveLength(2);
    expect(detail.members.map((m) => m.playerId).sort()).toEqual(
      players.slice(0, 3).sort()
    );
    const left = raw<{ active: number; left_at: string | null }>(
      db,
      `SELECT active, left_at FROM schedule_project_players WHERE project_id=? AND player_id=?`,
      created.id,
      Number(players[3])
    );
    expect(left.active).toBe(0);
    expect(left.left_at).toEqual(expect.any(String));
    // 仍记录一代（空 schedule），供幂等重放。
    expect(count(db, "schedule_generations")).toBe(2);
  });

  it("名单外球员 → player_not_found 且零副作用", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    expectErrorCode(
      () =>
        regenerate(
          {
            projectId: created.id,
            expectedRevision: 0,
            requestId: "req-noplayer",
            payloadHash: "h",
            newPlayerIds: [...players, "88888"],
            engineInput: {},
            engineResult: engineResult([
              [players[0], players[1], players[2], "88888"],
            ]),
            actorId: players[0],
          },
          db
        ),
      "player_not_found"
    );
    expect(raw<{ revision: number }>(db, `SELECT revision FROM schedule_projects WHERE id=?`, created.id).revision).toBe(0);
    expect(
      rawAll<{ status: string }>(
        db,
        `SELECT status FROM schedule_fixtures WHERE project_id=?`,
        created.id
      ).every((f) => f.status === "pending")
    ).toBe(true);
  });

  it("未知项目 → not_found", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    expectErrorCode(
      () =>
        regenerate(
          {
            projectId: 424242,
            expectedRevision: 0,
            requestId: "r",
            payloadHash: "h",
            newPlayerIds: players,
            engineInput: {},
            engineResult: null,
            actorId: players[0],
          },
          db
        ),
      "not_found"
    );
  });
});

describe("公平起点", () => {
  it("3 人各已打 3 场时加 1 新人：新人 fairness_base=3、actualPlayed=0，继续在场者不动", () => {
    const db = mkDb();
    const players = seedPlayers(db, 5);
    const roster = players.slice(0, 4);
    const created = createWith(db, roster, [
      [roster[0], roster[1], roster[2], roster[3]],
      [roster[0], roster[2], roster[1], roster[3]],
      [roster[0], roster[3], roster[1], roster[2]],
    ]);
    for (const f of getProject(created.id, db).fixtures) {
      markPlayed(
        {
          projectId: created.id,
          fixtureId: f.id,
          expectedFixtureRevision: f.revision,
          actorId: roster[0],
          playedAt: "2026-10-01",
        },
        db
      );
    }
    for (const p of roster) {
      expect(
        raw<{ c: number }>(
          db,
          `SELECT COUNT(*) AS c FROM schedule_fixtures
           WHERE project_id=? AND status IN ('played_unscored','recorded')
             AND (pa1=? OR pa2=? OR pb1=? OR pb2=?)`,
          created.id,
          p,
          p,
          p,
          p
        ).c
      ).toBe(3);
    }

    const newbie = players[4];
    // 3 次 markPlayed 各递增一次项目 revision → 当前为 3。
    regenerate(
      {
        projectId: created.id,
        expectedRevision: 3,
        requestId: "req-fair",
        payloadHash: "hf",
        newPlayerIds: [...roster, newbie],
        engineInput: { matches: 1 },
        engineResult: engineResult([
          [roster[0], roster[1], newbie, roster[2]],
        ]),
        actorId: roster[0],
      },
      db
    );

    const detail = getProject(created.id, db);
    const newbieMember = detail.members.find((m) => m.playerId === newbie);
    expect(newbieMember).toMatchObject({
      fairnessBase: 3,
      playedAtJoin: 0,
      actualPlayed: 0,
      pendingCount: 1,
    });
    for (const p of roster) {
      const m = detail.members.find((mm) => mm.playerId === p);
      expect(m).toMatchObject({ fairnessBase: 0, playedAtJoin: 0, actualPlayed: 3 });
    }
  });

  it("重新加入者视为新在场阶段：起点重设、历史行保留、不追补离开期间场次", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
      [players[0], players[2], players[1], players[3]],
    ]);
    // 打 1 场（四人全上），再让 p4 离开（matches=0 保存名单变更）。
    const f0 = getProject(created.id, db).fixtures[0];
    markPlayed(
      {
        projectId: created.id,
        fixtureId: f0.id,
        expectedFixtureRevision: f0.revision,
        actorId: players[0],
        playedAt: "2026-10-01",
      },
      db
    );
    // markPlayed 后项目 revision=1；离开续排（matches=0）后 revision=2。
    regenerate(
      {
        projectId: created.id,
        expectedRevision: 1,
        requestId: "req-leave",
        payloadHash: "hL",
        newPlayerIds: players.slice(0, 3),
        engineInput: { matches: 0 },
        engineResult: null,
        actorId: players[0],
      },
      db
    );

    // p4 重新加入：played_at_join 带入其历史真实已打（1 场）。
    regenerate(
      {
        projectId: created.id,
        expectedRevision: 2,
        requestId: "req-rejoin",
        payloadHash: "hR",
        newPlayerIds: players,
        engineInput: { matches: 1 },
        engineResult: engineResult([
          [players[0], players[1], players[3], players[2]],
        ]),
        actorId: players[0],
      },
      db
    );

    const row = raw<{ active: number; left_at: string | null; joined_at: string | null; fairness_base: number; played_at_join: number }>(
      db,
      `SELECT active, left_at, joined_at, fairness_base, played_at_join
       FROM schedule_project_players WHERE project_id=? AND player_id=?`,
      created.id,
      Number(players[3])
    );
    // 历史行保留（同 PK 行更新而非删除），新阶段起点。
    expect(row.active).toBe(1);
    expect(row.left_at).toBeNull();
    expect(row.joined_at).toEqual(expect.any(String));
    expect(row.played_at_join).toBe(1);
    // 继续在场者 effective = 0 + 1 - 0 = 1 → 新人 base = 1。
    expect(row.fairness_base).toBe(1);
    const detail = getProject(created.id, db);
    const rejoined = detail.members.find((m) => m.playerId === players[3]);
    expect(rejoined).toMatchObject({
      fairnessBase: 1,
      playedAtJoin: 1,
      actualPlayed: 1,
    });
    // 继续在场者（p1..p3）起点不动。
    for (const p of players.slice(0, 3)) {
      expect(
        detail.members.find((m) => m.playerId === p)
      ).toMatchObject({ fairnessBase: 0, playedAtJoin: 0, actualPlayed: 1 });
    }
  });
});

describe("场次状态机", () => {
  function setupOneFixture(db: Database.Database) {
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    const fixture = getProject(created.id, db).fixtures[0];
    return { players, created, fixture };
  }

  it("markPlayed：pending → played_unscored，写日期/标记人/版本号", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    const after = markPlayed(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: fixture.revision,
        actorId: players[1],
        playedAt: "2026-10-02",
      },
      db
    );
    expect(after).toMatchObject({
      status: "played_unscored",
      playedAt: "2026-10-02",
      markedBy: players[1],
      revision: 1,
      matchId: null,
    });
    expect(after.markedAt).toEqual(expect.any(String));
    // 已打计入公平统计：markPlayed 递增项目 revision（并发续排 CAS 失效）。
    expect(
      raw<{ revision: number }>(db, `SELECT revision FROM schedule_projects WHERE id=?`, created.id).revision
    ).toBe(1);
  });

  it("markPlayed 状态不符（played_unscored / recorded / superseded）→ conflict", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    markPlayed(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: 0,
        actorId: players[0],
        playedAt: "2026-10-02",
      },
      db
    );
    expectErrorCode(
      () =>
        markPlayed(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: 1,
            actorId: players[0],
            playedAt: "2026-10-03",
          },
          db
        ),
      "state_conflict"
    );

    // 造一个 superseded fixture：加一场 pending 再续排清空。
    const created2 = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
      [players[0], players[2], players[1], players[3]],
    ]);
    regenerate(
      {
        projectId: created2.id,
        expectedRevision: 0,
        requestId: "req-ss",
        payloadHash: "h",
        newPlayerIds: players,
        engineInput: { matches: 0 },
        engineResult: null,
        actorId: players[0],
      },
      db
    );
    const supersededFixture = getProject(created2.id, db).superseded[0];
    expectErrorCode(
      () =>
        markPlayed(
          {
            projectId: created2.id,
            fixtureId: supersededFixture.id,
            expectedFixtureRevision: supersededFixture.revision,
            actorId: players[0],
            playedAt: "2026-10-03",
          },
          db
        ),
      "state_conflict"
    );
  });

  it("markPlayed revision CAS 失败 → conflict", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    expectErrorCode(
      () =>
        markPlayed(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: 9,
            actorId: players[0],
            playedAt: "2026-10-02",
          },
          db
        ),
      "state_conflict"
    );
  });

  it("两步只计一次出场：played_unscored → recorded 后 actualPlayed 仍为 1", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    const marked = markPlayed(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: 0,
        actorId: players[0],
        playedAt: "2026-10-01",
      },
      db
    );
    const matchId = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[1]),
        pb1: Number(players[2]),
        pb2: Number(players[3]),
        scoreA: 21,
        scoreB: 18,
        playedAt: "2026-10-01",
        enteredBy: Number(players[0]),
      },
      db
    );
    const attached = attachScore(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: marked.revision,
        matchId,
        actorId: players[0],
      },
      db
    );
    expect(attached).toMatchObject({
      status: "recorded",
      matchId,
      playedAt: "2026-10-01", // 保留原「打过」日期
      scoreA: 21,
      scoreB: 18,
      revision: 2,
    });
    // 项目 revision 随标记已打 +1、记分再 +1。
    expect(
      raw<{ revision: number }>(db, `SELECT revision FROM schedule_projects WHERE id=?`, created.id).revision
    ).toBe(2);
    const member = getProject(created.id, db).members.find(
      (m) => m.playerId === players[0]
    );
    expect(member).toMatchObject({ actualPlayed: 1, pendingCount: 0 });
  });

  it("pending → recorded 直录比分（playedAt 必填且须与 match 一致）", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    const matchId = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[1]),
        pb1: Number(players[2]),
        pb2: Number(players[3]),
        scoreA: 19,
        scoreB: 21,
        playedAt: "2026-10-05",
      },
      db
    );
    const attached = attachScore(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: fixture.revision,
        matchId,
        actorId: players[2],
        playedAt: "2026-10-05",
      },
      db
    );
    expect(attached).toMatchObject({
      status: "recorded",
      playedAt: "2026-10-05",
      markedBy: players[2],
      scoreA: 19,
      scoreB: 21,
    });

    // 已 recorded 再 attachScore → conflict。
    const matchId2 = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[1]),
        pb1: Number(players[2]),
        pb2: Number(players[3]),
        scoreA: 21,
        scoreB: 5,
        playedAt: "2026-10-05",
      },
      db
    );
    expectErrorCode(
      () =>
        attachScore(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: attached.revision,
            matchId: matchId2,
            actorId: players[0],
          },
          db
        ),
      "state_conflict"
    );
  });

  it("attachScore 校验 match：不存在 / 四人不一致 / 日期不一致", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    expectErrorCode(
      () =>
        attachScore(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: 0,
            matchId: 77777,
            actorId: players[0],
            playedAt: "2026-10-05",
          },
          db
        ),
      "not_found"
    );

    const wrongLineup = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[2]),
        pb1: Number(players[1]),
        pb2: Number(players[3]),
        scoreA: 21,
        scoreB: 10,
        playedAt: "2026-10-05",
      },
      db
    );
    expectErrorCode(
      () =>
        attachScore(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: 0,
            matchId: wrongLineup,
            actorId: players[0],
            playedAt: "2026-10-05",
          },
          db
        ),
      "match_mismatch"
    );

    const wrongDate = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[1]),
        pb1: Number(players[2]),
        pb2: Number(players[3]),
        scoreA: 21,
        scoreB: 10,
        playedAt: "2026-10-06",
      },
      db
    );
    expectErrorCode(
      () =>
        attachScore(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: 0,
            matchId: wrongDate,
            actorId: players[0],
            playedAt: "2026-10-05",
          },
          db
        ),
      "match_mismatch"
    );

    // played_unscored → recorded 时 match 日期必须等于已打日期。
    const marked = markPlayed(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: 0,
        actorId: players[0],
        playedAt: "2026-10-01",
      },
      db
    );
    const wrongDate2 = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[1]),
        pb1: Number(players[2]),
        pb2: Number(players[3]),
        scoreA: 21,
        scoreB: 10,
        playedAt: "2026-10-02",
      },
      db
    );
    expectErrorCode(
      () =>
        attachScore(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: marked.revision,
            matchId: wrongDate2,
            actorId: players[0],
          },
          db
        ),
      "match_mismatch"
    );
  });

  it("detachScore：recorded → played_unscored，保留日期、解除 match_id", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    markPlayed(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: 0,
        actorId: players[0],
        playedAt: "2026-10-01",
      },
      db
    );
    const matchId = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[1]),
        pb1: Number(players[2]),
        pb2: Number(players[3]),
        scoreA: 21,
        scoreB: 18,
        playedAt: "2026-10-01",
      },
      db
    );
    const attached = attachScore(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: 1,
        matchId,
        actorId: players[0],
      },
      db
    );
    // recorded 状态下 matchId 对不上 → match_mismatch。
    const other = setupOneFixture(db);
    const otherMatch = addMatch(
      {
        pa1: Number(other.players[0]),
        pa2: Number(other.players[1]),
        pb1: Number(other.players[2]),
        pb2: Number(other.players[3]),
        scoreA: 21,
        scoreB: 12,
        playedAt: "2026-10-07",
      },
      db
    );
    expectErrorCode(
      () =>
        detachScore(
          { projectId: created.id, fixtureId: fixture.id, matchId: otherMatch },
          db
        ),
      "match_mismatch"
    );
    const detached = detachScore(
      { projectId: created.id, fixtureId: fixture.id, matchId },
      db
    );
    expect(detached).toMatchObject({
      status: "played_unscored",
      matchId: null,
      playedAt: "2026-10-01",
      scoreA: null,
      scoreB: null,
    });
    expect(detached.revision).toBe(attached.revision + 1);
    // 项目 revision：markPlayed +1、attach +1、detach +1。
    expect(
      raw<{ revision: number }>(db, `SELECT revision FROM schedule_projects WHERE id=?`, created.id).revision
    ).toBe(3);

    // 重复 detach（已 played_unscored）→ conflict。
    expectErrorCode(
      () => detachScore({ projectId: created.id, fixtureId: fixture.id, matchId }, db),
      "state_conflict"
    );
  });

  it("其他项目的 fixture → not_found（不泄露存在性）", () => {
    const db = mkDb();
    const a = setupOneFixture(db);
    const b = setupOneFixture(db);
    expectErrorCode(
      () =>
        markPlayed(
          {
            projectId: b.created.id,
            fixtureId: a.fixture.id,
            expectedFixtureRevision: 0,
            actorId: b.players[0],
            playedAt: "2026-10-01",
          },
          db
        ),
      "not_found"
    );
    expectErrorCode(
      () => getProject(987654, db),
      "not_found"
    );
  });

  it("非法 playedAt 日历日期被拒绝", () => {
    const db = mkDb();
    const { players, created, fixture } = setupOneFixture(db);
    expectErrorCode(
      () =>
        markPlayed(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: 0,
            actorId: players[0],
            playedAt: "2026-13-40",
          },
          db
        ),
      "validation_error"
    );
    expectErrorCode(
      () =>
        markPlayed(
          {
            projectId: created.id,
            fixtureId: fixture.id,
            expectedFixtureRevision: 0,
            actorId: players[0],
            playedAt: "2026-02-30",
          },
          db
        ),
      "validation_error"
    );
  });
});

describe("外键约束（RESTRICT，无级联）", () => {
  it("删除被引用项目/球员/比赛均被拒", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    // 项目被 project_players / generations / fixtures 引用。
    expect(() =>
      db.prepare(`DELETE FROM schedule_projects WHERE id = ?`).run(created.id)
    ).toThrow(/FOREIGN KEY/i);
    // 球员被 project_players 引用。
    expect(() =>
      db.prepare(`DELETE FROM players WHERE id = ?`).run(Number(players[1]))
    ).toThrow(/FOREIGN KEY/i);
    // marked_by 引用球员：markPlayed 后删除被拒。
    const fixture = getProject(created.id, db).fixtures[0];
    markPlayed(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: fixture.revision,
        actorId: players[2],
        playedAt: "2026-10-01",
      },
      db
    );
    expect(() =>
      db.prepare(`DELETE FROM players WHERE id = ?`).run(Number(players[2]))
    ).toThrow(/FOREIGN KEY/i);
    // match_id 引用 matches。
    const matchId = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[1]),
        pb1: Number(players[2]),
        pb2: Number(players[3]),
        scoreA: 21,
        scoreB: 18,
        playedAt: "2026-10-01",
      },
      db
    );
    attachScore(
      {
        projectId: created.id,
        fixtureId: fixture.id,
        expectedFixtureRevision: 1,
        matchId,
        actorId: players[0],
      },
      db
    );
    expect(() => db.prepare(`DELETE FROM matches WHERE id = ?`).run(matchId)).toThrow(
      /FOREIGN KEY/i
    );
  });

  it("fixture 挂不到其他项目的 generation（复合 FK）", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const a = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    const b = createWith(
      db,
      players,
      [[players[0], players[1], players[2], players[3]]]
    );
    const genA = raw<{ id: number }>(
      db,
      `SELECT id FROM schedule_generations WHERE project_id = ?`,
      a.id
    );
    expect(() =>
      db
        .prepare(
          `INSERT INTO schedule_fixtures
             (project_id, generation_id, display_no, pa1, pa2, pb1, pb2, status)
           VALUES (?, ?, 99, '1', '2', '3', '4', 'pending')`
        )
        .run(b.id, genA.id)
    ).toThrow(/FOREIGN KEY/i);
  });
});

describe("listProjects / getProject", () => {
  it("倒序游标分页，limit 上限 30，摘要计数正确", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const ids: number[] = [];
    for (let i = 0; i < 35; i++) {
      ids.push(
        createWith(
          db,
          players,
          [[players[0], players[1], players[2], players[3]]]
        ).id
      );
    }
    const page1 = listProjects({ limit: 30 }, db);
    expect(page1.projects).toHaveLength(30);
    expect(page1.nextCursor).not.toBeNull();
    expect(page1.projects[0].id).toBe(ids[34]);
    expect(page1.projects[29].id).toBe(ids[5]);
    expect(page1.projects[0]).toMatchObject({
      name: "周五场",
      model: "glicko2",
      revision: 0,
      activePlayers: 4,
      pendingCount: 1,
      playedUnscoredCount: 0,
      recordedCount: 0,
    });

    const page2 = listProjects({ cursor: page1.nextCursor!, limit: 30 }, db);
    expect(page2.projects.map((p) => p.id)).toEqual(ids.slice(0, 5).reverse());
    expect(page2.nextCursor).toBeNull();
  });

  it("getProject：成员统计、recorded 比分、superseded、最近 generation 快照", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
      [players[0], players[2], players[1], players[3]],
      [players[0], players[3], players[1], players[2]],
    ]);
    const [f1, f2, f3] = getProject(created.id, db).fixtures;

    // f1 打未记分；f2 完整记分；f3 保持 pending。
    const m1 = markPlayed(
      {
        projectId: created.id,
        fixtureId: f1.id,
        expectedFixtureRevision: f1.revision,
        actorId: players[0],
        playedAt: "2026-10-01",
      },
      db
    );
    const matchId = addMatch(
      {
        pa1: Number(players[0]),
        pa2: Number(players[2]),
        pb1: Number(players[1]),
        pb2: Number(players[3]),
        scoreA: 22,
        scoreB: 20,
        playedAt: "2026-10-02",
      },
      db
    );
    attachScore(
      {
        projectId: created.id,
        fixtureId: f2.id,
        expectedFixtureRevision: f2.revision,
        matchId,
        actorId: players[1],
        playedAt: "2026-10-02",
      },
      db
    );

    const detail = getProject(created.id, db);
    expect(detail.fixtures).toHaveLength(3);
    const withScore = detail.fixtures.find((f) => f.id === f2.id);
    expect(withScore).toMatchObject({
      status: "recorded",
      matchId,
      scoreA: 22,
      scoreB: 20,
    });
    expect(detail.fixtures.find((f) => f.id === f1.id)).toMatchObject({
      status: "played_unscored",
      scoreA: null,
      scoreB: null,
    });
    // 每人都出场 2 次（f1 + f2），待打 1 次（f3）。
    for (const m of detail.members) {
      expect(m.actualPlayed).toBe(2);
      expect(m.pendingCount).toBe(1);
    }

    // 续排一次产生 superseded，并检查最新 generation 快照。
    // markPlayed +1、attachScore +1 → 当前 revision=2。
    regenerate(
      {
        projectId: created.id,
        expectedRevision: 2,
        requestId: "req-snap",
        payloadHash: "hS",
        newPlayerIds: players,
        engineInput: { playerIds: players, matches: 1, seed: 9 },
        engineResult: engineResult([
          [players[1], players[0], players[3], players[2]],
        ]),
        actorId: players[0],
      },
      db
    );
    const after = getProject(created.id, db);
    expect(after.superseded.map((f) => f.id)).toEqual([f3.id]);
    expect(after.fixtures).toHaveLength(3); // f1 played_unscored + f2 recorded + 新 pending
    expect(after.latestGeneration).toMatchObject({
      sourceRevision: 2,
      input: { playerIds: players, matches: 1, seed: 9 },
      metrics: {
        J: 0,
        imbalancedMatchCount: 0,
        totalImbalanceExcess: 0,
        directRepeatCount: 0,
      },
    });
    expect(after.revision).toBe(3);
  });
});

describe("fixtureWinRates 落库", () => {
  it("createProject：逐场胜率随 fixture 落库；缺省为 null", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(
      db,
      players,
      [[players[0], players[1], players[2], players[3]]],
      { fixtureWinRates: [0.625] }
    );
    expect(getProject(created.id, db).fixtures[0].generatedWinRate).toBe(0.625);

    const plain = createWith(db, players, [
      [players[0], players[3], players[1], players[2]],
    ]);
    expect(getProject(plain.id, db).fixtures[0].generatedWinRate).toBeNull();
  });

  it("createProject：长度不符/越界/缺 engineResult 均拒绝", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    expectErrorCode(
      () =>
        createWith(
          db,
          players,
          [[players[0], players[1], players[2], players[3]]],
          { fixtureWinRates: [0.5, 0.6] }
        ),
      "validation_error"
    );
    expectErrorCode(
      () =>
        createWith(
          db,
          players,
          [[players[0], players[1], players[2], players[3]]],
          { fixtureWinRates: [1.5] }
        ),
      "validation_error"
    );
    expectErrorCode(
      () =>
        createProject(
          {
            name: "无引擎",
            model: "legacy",
            createdBy: players[0],
            requestId: `req-no-engine-${Math.random().toString(36).slice(2, 8)}`,
            payloadHash: "h",
            playerIds: players,
            fixtureWinRates: [0.5],
          },
          undefined,
          db
        ),
      "validation_error"
    );
  });

  it("regenerate：新批次胜率落库，长度不符拒绝", () => {
    const db = mkDb();
    const players = seedPlayers(db, 4);
    const created = createWith(db, players, [
      [players[0], players[1], players[2], players[3]],
    ]);
    const result = regenerate(
      {
        projectId: created.id,
        expectedRevision: 0,
        requestId: "req-regen-rates",
        payloadHash: "hash-rates",
        newPlayerIds: players,
        engineInput: { matches: 1 },
        engineResult: engineResult([
          [players[0], players[2], players[1], players[3]],
        ]),
        fixtureWinRates: [0.5],
        actorId: players[0],
      },
      db
    );
    expect(result.fixtures[0].generatedWinRate).toBe(0.5);

    expectErrorCode(
      () =>
        regenerate(
          {
            projectId: created.id,
            expectedRevision: 1,
            requestId: "req-regen-bad-rates",
            payloadHash: "hash-bad-rates",
            newPlayerIds: players,
            engineInput: { matches: 1 },
            engineResult: engineResult([
              [players[0], players[2], players[1], players[3]],
            ]),
            fixtureWinRates: [0.5, 0.6],
            actorId: players[0],
          },
          db
        ),
      "validation_error"
    );
  });
});
