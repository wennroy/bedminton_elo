import type Database from "better-sqlite3";
import { getDb } from "./db";
import type { SchedulingResult } from "./scheduling/types";

/**
 * 排阵项目的存储层。计算逻辑（搜索/评分/目标函数）由调用方注入，
 * 本模块不 import 任何搜索或评分模块，只负责事务内的事实写入与读取。
 *
 * 对外约定：球员 ID 一律为 string（与 T1 DTO 对齐），DB 内部 INTEGER
 * 主键由本层适配；项目/场次/代次 ID 为 number。
 */

/** 引擎输入/结果快照的结构版本，随快照字段演进递增。 */
export const SNAPSHOT_SCHEMA_VERSION = 1;

export const MAX_PAGE_SIZE = 30;
const DEFAULT_PAGE_SIZE = 20;

export type FixtureStatus = "pending" | "played_unscored" | "recorded" | "superseded";

export const FIXTURE_STATUSES: readonly FixtureStatus[] = [
  "pending",
  "played_unscored",
  "recorded",
  "superseded",
];

export type ScheduleProjectErrorCode =
  | "not_found"
  | "idempotency_conflict"
  | "revision_conflict"
  | "state_conflict"
  | "validation_error"
  | "player_not_found"
  | "match_mismatch";

/** 带明确 code 的存储层错误，服务层可据此映射 HTTP 状态码。 */
export class ScheduleProjectError extends Error {
  readonly code: ScheduleProjectErrorCode;
  /** revision_conflict 时携带服务端当前 revision，便于客户端重试。 */
  readonly currentRevision?: number;

  constructor(
    code: ScheduleProjectErrorCode,
    message: string,
    currentRevision?: number
  ) {
    super(message);
    this.name = "ScheduleProjectError";
    this.code = code;
    this.currentRevision = currentRevision;
  }
}

export function isScheduleProjectError(
  err: unknown,
  code?: ScheduleProjectErrorCode
): err is ScheduleProjectError {
  return (
    err instanceof ScheduleProjectError &&
    (code === undefined || err.code === code)
  );
}

export interface FixtureDto {
  id: number;
  projectId: number;
  generationId: number;
  displayNo: number;
  pa1: string;
  pa2: string;
  pb1: string;
  pb2: string;
  generatedWinRate: number | null;
  status: FixtureStatus;
  revision: number;
  playedAt: string | null;
  markedAt: string | null;
  markedBy: string | null;
  matchId: number | null;
  /** recorded 时联 matches 只读取比分；其余状态为 null。 */
  scoreA: number | null;
  scoreB: number | null;
}

export interface ProjectMember {
  playerId: string;
  fairnessBase: number;
  playedAtJoin: number;
  joinedAt: string | null;
  leftAt: string | null;
  /** 本项目内 status IN ('played_unscored','recorded') 且该人出场的不同 fixture 数（实时统计，无缓存列）。 */
  actualPlayed: number;
  /** 本项目内 pending 且该人出场的不同 fixture 数。 */
  pendingCount: number;
}

export interface ProjectSummary {
  id: number;
  name: string;
  model: string;
  revision: number;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  activePlayers: number;
  pendingCount: number;
  playedUnscoredCount: number;
  recordedCount: number;
}

export interface ProjectDetail {
  id: number;
  name: string;
  model: string;
  revision: number;
  nextFixtureNo: number;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  members: ProjectMember[];
  /** 有效场次：pending + played_unscored + recorded，按 display_no 升序。 */
  fixtures: FixtureDto[];
  /** 已被取代的场次（旧链接只读展示），按 display_no 升序。 */
  superseded: FixtureDto[];
  latestGeneration: {
    id: number;
    requestId: string;
    sourceRevision: number;
    input: unknown;
    metrics: unknown;
    createdAt: string;
  } | null;
}

// ---------------------------------------------------------------------------
// 参数校验
// ---------------------------------------------------------------------------

function resolveDb(db?: Database.Database): Database.Database {
  return db ?? getDb();
}

function requireSafeInt(value: unknown, field: string, min = 1): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < min) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 必须是 >= ${min} 的安全整数，实际: ${JSON.stringify(value)}`
    );
  }
  return value;
}

/** 球员 ID 对外为 string，内部必须是正整数字符串（对齐 players.id 形状）。 */
function requirePlayerId(value: unknown, field: string): string {
  if (typeof value !== "string" || !/^[1-9]\d*$/.test(value)) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 必须是正整数字符串，实际: ${JSON.stringify(value)}`
    );
  }
  if (!Number.isSafeInteger(Number(value))) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 超出安全整数范围: ${value}`
    );
  }
  return value;
}

function requireUniquePlayerIds(value: unknown, field: string): string[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 必须是非空数组，实际: ${JSON.stringify(value)}`
    );
  }
  const ids = value.map((v) => requirePlayerId(v, field));
  if (new Set(ids).size !== ids.length) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 存在重复球员 ID`
    );
  }
  return ids;
}

function requireNonEmptyString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 必须是非空字符串，实际: ${JSON.stringify(value)}`
    );
  }
  return value;
}

function requireIsoDate(value: unknown, field: string): string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 必须是 YYYY-MM-DD 日历日期，实际: ${JSON.stringify(value)}`
    );
  }
  const t = Date.parse(`${value}T00:00:00Z`);
  if (Number.isNaN(t) || new Date(t).toISOString().slice(0, 10) !== value) {
    throw new ScheduleProjectError(
      "validation_error",
      `${field} 不是真实日历日期: ${value}`
    );
  }
  return value;
}

function validateEngineResult(result: SchedulingResult): void {
  if (result === null || typeof result !== "object") {
    throw new ScheduleProjectError(
      "validation_error",
      "engineResult 必须是 SchedulingResult 对象"
    );
  }
  if (!Array.isArray(result.schedule)) {
    throw new ScheduleProjectError(
      "validation_error",
      "engineResult.schedule 必须是数组"
    );
  }
  for (const [i, m] of result.schedule.entries()) {
    const ids = [m?.a1, m?.a2, m?.b1, m?.b2];
    if (ids.some((v) => typeof v !== "string" || v.length === 0)) {
      throw new ScheduleProjectError(
        "validation_error",
        `engineResult.schedule[${i}] 四名选手必须是非空字符串`
      );
    }
    if (new Set(ids).size !== 4) {
      throw new ScheduleProjectError(
        "validation_error",
        `engineResult.schedule[${i}] 四名选手必须互不相同`
      );
    }
  }
}

/** 逐场生成胜率：与 schedule 同序，每项为 [0,1] 有限数或 null。 */
function validateFixtureWinRates(
  rates: (number | null)[],
  scheduleLength: number
): void {
  if (!Array.isArray(rates) || rates.length !== scheduleLength) {
    throw new ScheduleProjectError(
      "validation_error",
      `fixtureWinRates 长度必须等于场次总数 ${scheduleLength}`
    );
  }
  for (const [i, r] of rates.entries()) {
    if (r === null) continue;
    if (typeof r !== "number" || !Number.isFinite(r) || r < 0 || r > 1) {
      throw new ScheduleProjectError(
        "validation_error",
        `fixtureWinRates[${i}] 必须是 [0,1] 有限数或 null`
      );
    }
  }
}

// ---------------------------------------------------------------------------
// 行映射与通用查询
// ---------------------------------------------------------------------------

interface FixtureRow {
  id: number;
  projectId: number;
  generationId: number;
  displayNo: number;
  pa1: string;
  pa2: string;
  pb1: string;
  pb2: string;
  generatedWinRate: number | null;
  status: FixtureStatus;
  revision: number;
  playedAt: string | null;
  markedAt: string | null;
  markedBy: string | null;
  matchId: number | null;
  scoreA: number | null;
  scoreB: number | null;
}

const FIXTURE_DTO_SELECT = `
SELECT
  f.id,
  f.project_id AS projectId,
  f.generation_id AS generationId,
  f.display_no AS displayNo,
  f.pa1,
  f.pa2,
  f.pb1,
  f.pb2,
  f.generated_win_rate AS generatedWinRate,
  f.status,
  f.revision,
  f.played_at AS playedAt,
  f.marked_at AS markedAt,
  f.marked_by AS markedBy,
  f.match_id AS matchId,
  m.score_a AS scoreA,
  m.score_b AS scoreB
FROM schedule_fixtures f
LEFT JOIN matches m ON m.id = f.match_id
`;

function getFixtureDto(
  conn: Database.Database,
  projectId: number,
  fixtureId: number
): FixtureDto {
  const row = conn
    .prepare(`${FIXTURE_DTO_SELECT} WHERE f.id = ? AND f.project_id = ?`)
    .get(fixtureId, projectId) as FixtureRow | undefined;
  if (!row) {
    // 不泄露存在性：fixture 不存在或不属于该项目统一报 not_found。
    throw new ScheduleProjectError(
      "not_found",
      `项目 ${projectId} 下不存在场次 ${fixtureId}`
    );
  }
  return row;
}

function listFixtureDtos(
  conn: Database.Database,
  projectId: number,
  statuses: FixtureStatus[]
): FixtureDto[] {
  const placeholders = statuses.map(() => "?").join(", ");
  return conn
    .prepare(
      `${FIXTURE_DTO_SELECT}
       WHERE f.project_id = ? AND f.status IN (${placeholders})
       ORDER BY f.display_no ASC, f.id ASC`
    )
    .all(projectId, ...statuses) as FixtureDto[];
}

/** 某一 generation 创建的全部 fixtures（不限状态），供幂等重放返回既有结果。 */
function listFixturesByGeneration(
  conn: Database.Database,
  projectId: number,
  generationId: number
): FixtureDto[] {
  return conn
    .prepare(
      `${FIXTURE_DTO_SELECT}
       WHERE f.project_id = ? AND f.generation_id = ?
       ORDER BY f.display_no ASC, f.id ASC`
    )
    .all(projectId, generationId) as FixtureDto[];
}

/** 本项目内该球员出场的 played（played_unscored + recorded）fixture 数。 */
function countPlayerPlayed(
  conn: Database.Database,
  projectId: number,
  playerId: string
): number {
  const row = conn
    .prepare(
      `SELECT COUNT(*) AS c FROM schedule_fixtures
       WHERE project_id = ? AND status IN ('played_unscored', 'recorded')
         AND (pa1 = ? OR pa2 = ? OR pb1 = ? OR pb2 = ?)`
    )
    .get(projectId, playerId, playerId, playerId, playerId) as { c: number };
  return row.c;
}

function countPlayerPending(
  conn: Database.Database,
  projectId: number,
  playerId: string
): number {
  const row = conn
    .prepare(
      `SELECT COUNT(*) AS c FROM schedule_fixtures
       WHERE project_id = ? AND status = 'pending'
         AND (pa1 = ? OR pa2 = ? OR pb1 = ? OR pb2 = ?)`
    )
    .get(projectId, playerId, playerId, playerId, playerId) as { c: number };
  return row.c;
}

function assertPlayersExist(conn: Database.Database, idNums: number[]): void {
  const stmt = conn.prepare(`SELECT 1 FROM players WHERE id = ?`);
  for (const n of idNums) {
    if (!stmt.get(n)) {
      throw new ScheduleProjectError("player_not_found", `球员 ${n} 不存在`);
    }
  }
}

// ---------------------------------------------------------------------------
// createProject
// ---------------------------------------------------------------------------

export interface CreateProjectInput {
  name: string;
  model: string;
  /** 创建者球员 ID（string）。 */
  createdBy: string;
  /** 幂等键：同 requestId + 同 payloadHash 重放返回已建项目。 */
  requestId: string;
  payloadHash: string;
  /** 初始名单（球员 ID string，唯一）。 */
  playerIds: string[];
  /** 引擎输入快照（JSON 可序列化），随 generation 落库。 */
  engineInput?: unknown;
  /** 逐场生成胜率（与 engineResult.schedule 同序，取值 [0,1] 或 null）。 */
  fixtureWinRates?: (number | null)[];
}

export interface CreateProjectResult {
  id: number;
  revision: number;
  replayed: boolean;
}

export function createProject(
  input: CreateProjectInput,
  engineResult?: SchedulingResult,
  db?: Database.Database
): CreateProjectResult {
  const conn = resolveDb(db);
  const name = requireNonEmptyString(input.name, "name");
  const model = requireNonEmptyString(input.model, "model");
  const createdBy = requirePlayerId(input.createdBy, "createdBy");
  const requestId = requireNonEmptyString(input.requestId, "requestId");
  const payloadHash = requireNonEmptyString(input.payloadHash, "payloadHash");
  const playerIds = requireUniquePlayerIds(input.playerIds, "playerIds");
  if (engineResult !== undefined) validateEngineResult(engineResult);
  if (input.fixtureWinRates !== undefined) {
    if (engineResult === undefined) {
      throw new ScheduleProjectError(
        "validation_error",
        "fixtureWinRates 需要与 engineResult 一起提供"
      );
    }
    validateFixtureWinRates(input.fixtureWinRates, engineResult.schedule.length);
  }

  // 幂等重放（事务外只读查询即可判定）。
  const existing = conn
    .prepare(
      `SELECT id, create_payload_hash AS h, revision FROM schedule_projects WHERE create_request_id = ?`
    )
    .get(requestId) as { id: number; h: string | null; revision: number } | undefined;
  if (existing) {
    if (existing.h !== payloadHash) {
      throw new ScheduleProjectError(
        "idempotency_conflict",
        `create_request_id '${requestId}' 已用于不同 payload`
      );
    }
    return { id: existing.id, revision: existing.revision, replayed: true };
  }

  const idNums = playerIds.map(Number);

  const txFn = conn.transaction(() => {
    // createdBy 与成员一起做存在性预检（FK 兜底），给服务层干净的错误码。
    assertPlayersExist(conn, [...new Set([...idNums, Number(createdBy)])]);
    const info = conn
      .prepare(
        `INSERT INTO schedule_projects
           (name, model, next_fixture_no, created_by, create_request_id, create_payload_hash)
         VALUES (?, ?, 1, ?, ?, ?)`
      )
      .run(name, model, createdBy, requestId, payloadHash);
    const projectId = Number(info.lastInsertRowid);

    const insMember = conn.prepare(
      `INSERT INTO schedule_project_players
         (project_id, player_id, active, fairness_base, played_at_join, joined_at)
       VALUES (?, ?, 1, 0, 0, datetime('now'))`
    );
    for (const n of idNums) insMember.run(projectId, n);

    if (engineResult) {
      insertGeneration(conn, projectId, {
        requestId,
        payloadHash,
        sourceRevision: 0,
        engineInput: input.engineInput ?? null,
        engineResult,
        fixtureWinRates: input.fixtureWinRates,
        createdBy,
      });
    }
    return projectId;
  });

  const projectId = txFn.immediate();
  return { id: projectId, revision: 0, replayed: false };
}

// ---------------------------------------------------------------------------
// regenerate
// ---------------------------------------------------------------------------

export interface RegenerateInput {
  projectId: number;
  expectedRevision: number;
  requestId: string;
  payloadHash: string;
  /** 完整目标名单（球员 ID string，唯一）。 */
  newPlayerIds: string[];
  /** 引擎输入快照（JSON 可序列化）。 */
  engineInput: unknown;
  /** 引擎结果；matches=0 时为 null（只保存名单变更、清空未打）。 */
  engineResult: SchedulingResult | null;
  /** 逐场生成胜率（与 engineResult.schedule 同序，取值 [0,1] 或 null）。 */
  fixtureWinRates?: (number | null)[];
  actorId: string;
}

export interface RegenerateResult {
  projectId: number;
  /** 本次生成提交后的项目 revision。 */
  revision: number;
  generationId: number;
  /** 本次生成创建的 fixtures（matches=0 时为空数组）。 */
  fixtures: FixtureDto[];
  supersededCount: number;
  joinedPlayers: string[];
  leftPlayers: string[];
  replayed: boolean;
}

export function regenerate(
  input: RegenerateInput,
  db?: Database.Database
): RegenerateResult {
  const conn = resolveDb(db);
  const projectId = requireSafeInt(input.projectId, "projectId");
  const expectedRevision = requireSafeInt(
    input.expectedRevision,
    "expectedRevision",
    0
  );
  const requestId = requireNonEmptyString(input.requestId, "requestId");
  const payloadHash = requireNonEmptyString(input.payloadHash, "payloadHash");
  const actorId = requirePlayerId(input.actorId, "actorId");
  const newPlayerIds = requireUniquePlayerIds(input.newPlayerIds, "newPlayerIds");
  if (input.engineResult != null) validateEngineResult(input.engineResult);
  if (input.fixtureWinRates !== undefined) {
    if (input.engineResult == null) {
      throw new ScheduleProjectError(
        "validation_error",
        "fixtureWinRates 需要与非空 engineResult 一起提供"
      );
    }
    validateFixtureWinRates(input.fixtureWinRates, input.engineResult.schedule.length);
  }

  const txFn = conn.transaction(() => {
    const project = conn
      .prepare(`SELECT id, revision FROM schedule_projects WHERE id = ?`)
      .get(projectId) as { id: number; revision: number } | undefined;
    if (!project) {
      throw new ScheduleProjectError("not_found", `项目 ${projectId} 不存在`);
    }

    // 幂等：同 (project, request_id) 且 hash 一致直接返回既有结果。
    const existingGen = conn
      .prepare(
        `SELECT id, payload_hash AS h FROM schedule_generations
         WHERE project_id = ? AND request_id = ?`
      )
      .get(projectId, requestId) as { id: number; h: string | null } | undefined;
    if (existingGen) {
      if (existingGen.h !== payloadHash) {
        throw new ScheduleProjectError(
          "idempotency_conflict",
          `request_id '${requestId}' 已用于不同 payload`
        );
      }
      const fixtures = listFixturesByGeneration(conn, projectId, existingGen.id);
      const currentRevision = (
        conn
          .prepare(`SELECT revision FROM schedule_projects WHERE id = ?`)
          .get(projectId) as { revision: number }
      ).revision;
      return {
        projectId,
        revision: currentRevision,
        generationId: existingGen.id,
        fixtures,
        supersededCount: 0,
        joinedPlayers: [],
        leftPlayers: [],
        replayed: true,
      };
    }

    // revision CAS：不等则携带当前 revision 抛错，整事务回滚。
    if (project.revision !== expectedRevision) {
      throw new ScheduleProjectError(
        "revision_conflict",
        `项目 revision 已变化：期望 ${expectedRevision}，当前 ${project.revision}`,
        project.revision
      );
    }

    const newNums = newPlayerIds.map(Number);
    assertPlayersExist(conn, newNums);

    // 旧 pending 全部置 superseded（played_unscored/recorded 原样保留，
    // fixture ID 与 display_no 永不复用）。
    const superseded = conn
      .prepare(
        `UPDATE schedule_fixtures
         SET status = 'superseded', revision = revision + 1
         WHERE project_id = ? AND status = 'pending'`
      )
      .run(projectId);

    // 名单变更：离开者置 inactive；新加入/重新加入者获得公平起点。
    const rosterRows = conn
      .prepare(
        `SELECT player_id AS pid, active FROM schedule_project_players WHERE project_id = ?`
      )
      .all(projectId) as Array<{ pid: number; active: number }>;
    const activeSet = new Set(
      rosterRows.filter((r) => r.active === 1).map((r) => r.pid)
    );
    const newSet = new Set(newNums);
    const continuing = newNums.filter((n) => activeSet.has(n));
    const leaving = [...activeSet].filter((n) => !newSet.has(n));
    const joining = newNums.filter((n) => !activeSet.has(n));

    // 继续在场者的 effective_i = fairness_base + actualPlayed - played_at_join，
    // 新加入者起点取继续在场者的最低 effective（无继续在场者则 0）。
    let minEffective = Infinity;
    for (const n of continuing) {
      const row = conn
        .prepare(
          `SELECT fairness_base AS fb, played_at_join AS paj
           FROM schedule_project_players WHERE project_id = ? AND player_id = ?`
        )
        .get(projectId, n) as { fb: number; paj: number };
      const effective =
        row.fb + countPlayerPlayed(conn, projectId, String(n)) - row.paj;
      if (effective < minEffective) minEffective = effective;
    }
    const joinBase = continuing.length === 0 ? 0 : minEffective;

    const markLeft = conn.prepare(
      `UPDATE schedule_project_players SET active = 0, left_at = datetime('now')
       WHERE project_id = ? AND player_id = ? AND active = 1`
    );
    for (const n of leaving) markLeft.run(projectId, n);

    const rowExists = conn.prepare(
      `SELECT 1 FROM schedule_project_players WHERE project_id = ? AND player_id = ?`
    );
    const rejoin = conn.prepare(
      `UPDATE schedule_project_players
       SET active = 1, left_at = NULL, joined_at = datetime('now'),
           fairness_base = ?, played_at_join = ?
       WHERE project_id = ? AND player_id = ?`
    );
    const insJoin = conn.prepare(
      `INSERT INTO schedule_project_players
         (project_id, player_id, active, fairness_base, played_at_join, joined_at)
       VALUES (?, ?, 1, ?, ?, datetime('now'))`
    );
    for (const n of joining) {
      // played_at_join = 其当前在本项目内的真实已打数（重新加入者带入历史）。
      const actual = countPlayerPlayed(conn, projectId, String(n));
      if (rowExists.get(projectId, n)) {
        rejoin.run(joinBase, actual, projectId, n);
      } else {
        insJoin.run(projectId, n, joinBase, actual);
      }
    }

    const generationId = insertGeneration(conn, projectId, {
      requestId,
      payloadHash,
      sourceRevision: project.revision,
      engineInput: input.engineInput,
      engineResult: input.engineResult,
      fixtureWinRates: input.fixtureWinRates,
      createdBy: actorId,
    });

    conn
      .prepare(
        `UPDATE schedule_projects SET revision = revision + 1, updated_at = datetime('now')
         WHERE id = ?`
      )
      .run(projectId);

    const fixtures = listFixturesByGeneration(conn, projectId, generationId);

    return {
      projectId,
      revision: project.revision + 1,
      generationId,
      fixtures,
      supersededCount: superseded.changes,
      joinedPlayers: joining.map(String),
      leftPlayers: leaving.map(String),
      replayed: false,
    };
  });

  return txFn.immediate();
}

/** 插入 generation 及其 fixtures；display_no 从 next_fixture_no 继续递增。 */
function insertGeneration(
  conn: Database.Database,
  projectId: number,
  g: {
    requestId: string;
    payloadHash: string;
    sourceRevision: number;
    engineInput: unknown;
    engineResult: SchedulingResult | null;
    fixtureWinRates?: (number | null)[];
    createdBy: string;
  }
): number {
  const metricsJson =
    g.engineResult != null ? JSON.stringify(g.engineResult.metrics) : null;
  const info = conn
    .prepare(
      `INSERT INTO schedule_generations
         (project_id, request_id, payload_hash, source_revision,
          snapshot_schema_version, input_json, result_metrics_json, created_by)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      projectId,
      g.requestId,
      g.payloadHash,
      g.sourceRevision,
      SNAPSHOT_SCHEMA_VERSION,
      JSON.stringify(g.engineInput ?? null),
      metricsJson,
      g.createdBy
    );
  const generationId = Number(info.lastInsertRowid);

  const schedule = g.engineResult?.schedule ?? [];
  if (schedule.length === 0) return generationId;

  if (g.fixtureWinRates !== undefined) {
    validateFixtureWinRates(g.fixtureWinRates, schedule.length);
  }

  const project = conn
    .prepare(`SELECT next_fixture_no AS n FROM schedule_projects WHERE id = ?`)
    .get(projectId) as { n: number };
  let nextNo = project.n;

  const ins = conn.prepare(
    `INSERT INTO schedule_fixtures
       (project_id, generation_id, display_no, pa1, pa2, pb1, pb2, generated_win_rate, status)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')`
  );
  for (const [i, m] of schedule.entries()) {
    ins.run(
      projectId,
      generationId,
      nextNo,
      m.a1,
      m.a2,
      m.b1,
      m.b2,
      g.fixtureWinRates?.[i] ?? null
    );
    nextNo += 1;
  }
  conn
    .prepare(`UPDATE schedule_projects SET next_fixture_no = ? WHERE id = ?`)
    .run(nextNo, projectId);
  return generationId;
}

// ---------------------------------------------------------------------------
// listProjects / getProject
// ---------------------------------------------------------------------------

export interface ListProjectsParams {
  cursor?: number;
  limit?: number;
}

export interface ListProjectsResult {
  projects: ProjectSummary[];
  nextCursor: number | null;
}

export function listProjects(
  params: ListProjectsParams = {},
  db?: Database.Database
): ListProjectsResult {
  const conn = resolveDb(db);
  const limit =
    params.limit === undefined
      ? DEFAULT_PAGE_SIZE
      : requireSafeInt(params.limit, "limit");
  const clamped = Math.min(limit, MAX_PAGE_SIZE);
  const cursor =
    params.cursor === undefined ? null : requireSafeInt(params.cursor, "cursor");

  const rows = conn
    .prepare(
      `SELECT id, name, model, revision, created_by AS createdBy,
              created_at AS createdAt, updated_at AS updatedAt
       FROM schedule_projects
       ${cursor !== null ? "WHERE id < ?" : ""}
       ORDER BY id DESC
       LIMIT ?`
    )
    .all(...(cursor !== null ? [cursor, clamped + 1] : [clamped + 1])) as Array<
    Omit<ProjectSummary, "activePlayers" | "pendingCount" | "playedUnscoredCount" | "recordedCount">
  >;

  const hasMore = rows.length > clamped;
  const page = hasMore ? rows.slice(0, clamped) : rows;

  const projects = page.map((row) => {
    const counts = conn
      .prepare(
        `SELECT
           (SELECT COUNT(*) FROM schedule_project_players
             WHERE project_id = ? AND active = 1) AS activePlayers,
           (SELECT COUNT(*) FROM schedule_fixtures
             WHERE project_id = ? AND status = 'pending') AS pendingCount,
           (SELECT COUNT(*) FROM schedule_fixtures
             WHERE project_id = ? AND status = 'played_unscored') AS playedUnscoredCount,
           (SELECT COUNT(*) FROM schedule_fixtures
             WHERE project_id = ? AND status = 'recorded') AS recordedCount`
      )
      .get(row.id, row.id, row.id, row.id) as {
      activePlayers: number;
      pendingCount: number;
      playedUnscoredCount: number;
      recordedCount: number;
    };
    return { ...row, ...counts };
  });

  return {
    projects,
    nextCursor: hasMore && page.length > 0 ? page[page.length - 1].id : null,
  };
}

export function getProject(
  projectId: number,
  db?: Database.Database
): ProjectDetail {
  const conn = resolveDb(db);
  const id = requireSafeInt(projectId, "projectId");
  const project = conn
    .prepare(
      `SELECT id, name, model, revision, next_fixture_no AS nextFixtureNo,
              created_by AS createdBy, created_at AS createdAt, updated_at AS updatedAt
       FROM schedule_projects WHERE id = ?`
    )
    .get(id) as
    | Pick<
        ProjectDetail,
        "id" | "name" | "model" | "revision" | "nextFixtureNo" | "createdBy" | "createdAt" | "updatedAt"
      >
    | undefined;
  if (!project) {
    throw new ScheduleProjectError("not_found", `项目 ${projectId} 不存在`);
  }

  const memberRows = conn
    .prepare(
      `SELECT player_id AS pid, fairness_base AS fairnessBase,
              played_at_join AS playedAtJoin, joined_at AS joinedAt, left_at AS leftAt
       FROM schedule_project_players
       WHERE project_id = ? AND active = 1
       ORDER BY player_id ASC`
    )
    .all(id) as Array<{
    pid: number;
    fairnessBase: number;
    playedAtJoin: number;
    joinedAt: string | null;
    leftAt: string | null;
  }>;
  const members: ProjectMember[] = memberRows.map((r) => {
    const playerId = String(r.pid);
    return {
      playerId,
      fairnessBase: r.fairnessBase,
      playedAtJoin: r.playedAtJoin,
      joinedAt: r.joinedAt,
      leftAt: r.leftAt,
      actualPlayed: countPlayerPlayed(conn, id, playerId),
      pendingCount: countPlayerPending(conn, id, playerId),
    };
  });

  const fixtures = listFixtureDtos(conn, id, [
    "pending",
    "played_unscored",
    "recorded",
  ]);
  const superseded = listFixtureDtos(conn, id, ["superseded"]);

  const genRow = conn
    .prepare(
      `SELECT id, request_id AS requestId, source_revision AS sourceRevision,
              input_json AS inputJson, result_metrics_json AS metricsJson,
              created_at AS createdAt
       FROM schedule_generations
       WHERE project_id = ?
       ORDER BY id DESC
       LIMIT 1`
    )
    .get(id) as
    | {
        id: number;
        requestId: string;
        sourceRevision: number;
        inputJson: string;
        metricsJson: string | null;
        createdAt: string;
      }
    | undefined;

  return {
    ...project,
    members,
    fixtures,
    superseded,
    latestGeneration: genRow
      ? {
          id: genRow.id,
          requestId: genRow.requestId,
          sourceRevision: genRow.sourceRevision,
          input: JSON.parse(genRow.inputJson) as unknown,
          metrics: genRow.metricsJson ? (JSON.parse(genRow.metricsJson) as unknown) : null,
          createdAt: genRow.createdAt,
        }
      : null,
  };
}

// ---------------------------------------------------------------------------
// markPlayed / attachScore / detachScore（场次状态机）
// ---------------------------------------------------------------------------

export interface MarkPlayedInput {
  projectId: number;
  fixtureId: number;
  expectedFixtureRevision: number;
  actorId: string;
  /** 日历日期 YYYY-MM-DD。 */
  playedAt: string;
}

export function markPlayed(input: MarkPlayedInput, db?: Database.Database): FixtureDto {
  const conn = resolveDb(db);
  const projectId = requireSafeInt(input.projectId, "projectId");
  const fixtureId = requireSafeInt(input.fixtureId, "fixtureId");
  const expectedRevision = requireSafeInt(
    input.expectedFixtureRevision,
    "expectedFixtureRevision",
    0
  );
  const actorId = requirePlayerId(input.actorId, "actorId");
  const playedAt = requireIsoDate(input.playedAt, "playedAt");

  const txFn = conn.transaction(() => {
    const fixture = getFixtureDto(conn, projectId, fixtureId);
    if (fixture.revision !== expectedRevision) {
      throw new ScheduleProjectError(
        "state_conflict",
        `场次 revision 已变化：期望 ${expectedRevision}，当前 ${fixture.revision}`
      );
    }
    if (fixture.status !== "pending") {
      throw new ScheduleProjectError(
        "state_conflict",
        `仅 pending 可标记已打，当前状态 '${fixture.status}'`
      );
    }
    conn
      .prepare(
        `UPDATE schedule_fixtures
         SET status = 'played_unscored', played_at = ?, marked_at = datetime('now'),
             marked_by = ?, revision = revision + 1
         WHERE id = ?`
      )
      .run(playedAt, actorId, fixtureId);
    // 已打计入各人 actualPlayed/effective，必须递增项目 revision：
    // 让携带旧快照的并发续排在 CAS 处 409，而不是用陈旧公平/历史输入提交新批次。
    conn
      .prepare(
        `UPDATE schedule_projects SET revision = revision + 1, updated_at = datetime('now')
         WHERE id = ?`
      )
      .run(projectId);
    return getFixtureDto(conn, projectId, fixtureId);
  });

  return txFn.immediate();
}

export interface AttachScoreInput {
  projectId: number;
  fixtureId: number;
  expectedFixtureRevision: number;
  /** 由调用方（服务层）插入 matches 后传入的 match 主键。 */
  matchId: number;
  actorId: string;
  /** 仅 pending → recorded 时需要；played_unscored → recorded 时沿用原 played_at。 */
  playedAt?: string;
}

export function attachScore(input: AttachScoreInput, db?: Database.Database): FixtureDto {
  const conn = resolveDb(db);
  const projectId = requireSafeInt(input.projectId, "projectId");
  const fixtureId = requireSafeInt(input.fixtureId, "fixtureId");
  const expectedRevision = requireSafeInt(
    input.expectedFixtureRevision,
    "expectedFixtureRevision",
    0
  );
  const matchId = requireSafeInt(input.matchId, "matchId");
  const actorId = requirePlayerId(input.actorId, "actorId");

  const txFn = conn.transaction(() => {
    const fixture = getFixtureDto(conn, projectId, fixtureId);
    if (fixture.revision !== expectedRevision) {
      throw new ScheduleProjectError(
        "state_conflict",
        `场次 revision 已变化：期望 ${expectedRevision}，当前 ${fixture.revision}`
      );
    }
    if (fixture.status !== "pending" && fixture.status !== "played_unscored") {
      throw new ScheduleProjectError(
        "state_conflict",
        `状态 '${fixture.status}' 不能录入比分（仅 pending / played_unscored 可转 recorded）`
      );
    }

    const match = conn
      .prepare(
        `SELECT pa1, pa2, pb1, pb2, played_at AS playedAt FROM matches WHERE id = ?`
      )
      .get(matchId) as
      | { pa1: number; pa2: number; pb1: number; pb2: number; playedAt: string }
      | undefined;
    if (!match) {
      throw new ScheduleProjectError("not_found", `比赛 ${matchId} 不存在`);
    }
    // 四人一致且队伍归属一致（A 队/B 队不可对调）。
    if (
      String(match.pa1) !== fixture.pa1 ||
      String(match.pa2) !== fixture.pa2 ||
      String(match.pb1) !== fixture.pb1 ||
      String(match.pb2) !== fixture.pb2
    ) {
      throw new ScheduleProjectError(
        "match_mismatch",
        `比赛 ${matchId} 的四人/队伍归属与场次 ${fixtureId} 不一致`
      );
    }

    let playedAt: string;
    if (fixture.status === "pending") {
      playedAt = requireIsoDate(input.playedAt, "playedAt");
      if (match.playedAt !== playedAt) {
        throw new ScheduleProjectError(
          "match_mismatch",
          `比赛日期 ${match.playedAt} 与传入场次日期 ${playedAt} 不一致`
        );
      }
    } else {
      // played_unscored 保留原「打过」日期，不移动。
      playedAt = fixture.playedAt as string;
      if (match.playedAt !== playedAt) {
        throw new ScheduleProjectError(
          "match_mismatch",
          `比赛日期 ${match.playedAt} 与场次已打日期 ${playedAt} 不一致`
        );
      }
    }

    conn
      .prepare(
        `UPDATE schedule_fixtures
         SET status = 'recorded', match_id = ?, played_at = ?, marked_at = datetime('now'),
             marked_by = ?, revision = revision + 1
         WHERE id = ?`
      )
      .run(matchId, playedAt, actorId, fixtureId);
    conn
      .prepare(
        `UPDATE schedule_projects SET revision = revision + 1, updated_at = datetime('now')
         WHERE id = ?`
      )
      .run(projectId);
    return getFixtureDto(conn, projectId, fixtureId);
  });

  return txFn.immediate();
}

export interface DetachScoreInput {
  projectId: number;
  fixtureId: number;
  matchId: number;
}

/**
 * 撤回联动：recorded → played_unscored，解除 match_id，保留 played_at「打过」事实。
 *
 * 本函数不带独立事务——由 T4 删除比赛的事务包裹调用（多语句需调用方保证原子性）。
 */
export function detachScore(input: DetachScoreInput, db?: Database.Database): FixtureDto {
  const conn = resolveDb(db);
  const projectId = requireSafeInt(input.projectId, "projectId");
  const fixtureId = requireSafeInt(input.fixtureId, "fixtureId");
  const matchId = requireSafeInt(input.matchId, "matchId");

  const fixture = getFixtureDto(conn, projectId, fixtureId);
  if (fixture.status !== "recorded") {
    throw new ScheduleProjectError(
      "state_conflict",
      `状态 '${fixture.status}' 不能撤回比分（仅 recorded 可回 played_unscored）`
    );
  }
  if (fixture.matchId !== matchId) {
    throw new ScheduleProjectError(
      "match_mismatch",
      `场次 ${fixtureId} 关联的比赛是 ${fixture.matchId}，不是 ${matchId}`
    );
  }

  conn
    .prepare(
      `UPDATE schedule_fixtures
       SET status = 'played_unscored', match_id = NULL, revision = revision + 1
       WHERE id = ?`
    )
    .run(fixtureId);
  conn
    .prepare(
      `UPDATE schedule_projects SET revision = revision + 1, updated_at = datetime('now')
       WHERE id = ?`
    )
    .run(projectId);
  return getFixtureDto(conn, projectId, fixtureId);
}
