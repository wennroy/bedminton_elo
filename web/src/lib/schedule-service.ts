import { createHash, randomInt } from "node:crypto";
import type Database from "better-sqlite3";
import { getDb } from "./db";
import { predictElo, INITIAL_RATING } from "./elo";
import { readRatingConfig } from "./rating-config";
import { loadGlickoSnapshot } from "./rating-service";
import { predictDoubles } from "./ratings/doubles";
import { shanghaiLocalDateFromInstant } from "./ratings/calendar";
import type { RatingModel } from "./ratings/types";
import { recomputeAllRatings, mergePlayers } from "./repo";
import { assertValidScores, MatchValidationError } from "./match-validation";
import { recordMatch, type RecordMatchResult } from "./match-recording";
import {
  createProject,
  getProject,
  isScheduleProjectError,
  markPlayed,
  regenerate,
  ScheduleProjectError,
  type FixtureDto,
  type ProjectDetail,
} from "./schedule-projects";
import {
  ALGORITHM_VERSION,
  OBJECTIVE_VERSION,
  canonicalMatchKey,
  type SchedulingInput,
  type SchedulingResult,
} from "./scheduling/types";
import { solveScheduling, SEARCH_BUDGET } from "./scheduling/search";

/**
 * 排阵项目的服务层：参数/身份校验、评分快照一次捕获、引擎求解编排、
 * repo 错误 → HTTP 语义映射。不直接处理 HTTP，路由只负责解析请求体。
 *
 * 服务器不信客户端的阵容/历史/已打次数：阵容一律从 DB 内的 fixture 读取，
 * 历史与已打统计一律由 DB 现算。
 */

export const MIN_PLAYERS = 4;
export const MAX_PLAYERS = 64;
export const MIN_CREATE_MATCHES = 1;
export const MAX_MATCHES = 99;
export const MAX_SEED = 0xffffffff;
export const MAX_NAME_LENGTH = 60;

// ---------------------------------------------------------------------------
// 服务层错误（带 HTTP 语义；repo 层的 ScheduleProjectError 由映射器处理）
// ---------------------------------------------------------------------------

export type ScheduleServiceErrorCode =
  | "validation_error"
  | "player_not_found"
  | "not_found"
  | "idempotency_conflict"
  | "revision_conflict"
  | "state_conflict"
  | "already_recorded"
  | "fixture_superseded"
  | "match_mismatch"
  | "rating_unavailable"
  | "conflict";

/** 带明确 HTTP 语义的服务层错误；extra 合并进响应体（如当前 fixture / revision）。 */
export class ScheduleServiceError extends Error {
  readonly status: number;
  readonly code: ScheduleServiceErrorCode;
  readonly extra: Record<string, unknown>;

  constructor(
    status: number,
    code: ScheduleServiceErrorCode,
    message: string,
    extra: Record<string, unknown> = {}
  ) {
    super(message);
    this.name = "ScheduleServiceError";
    this.status = status;
    this.code = code;
    this.extra = extra;
  }
}

export function isScheduleServiceError(err: unknown): err is ScheduleServiceError {
  return err instanceof ScheduleServiceError;
}

// ---------------------------------------------------------------------------
// 输入校验（全部严格；失败抛 400 validation_error / player_not_found）
// ---------------------------------------------------------------------------

function resolveDb(db?: Database.Database): Database.Database {
  return db ?? getDb();
}

function badRequest(message: string): never {
  throw new ScheduleServiceError(400, "validation_error", message);
}

export function requireInt(
  value: unknown,
  field: string,
  min: number,
  max: number
): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    badRequest(`${field} 必须是安全整数，实际: ${JSON.stringify(value)}`);
  }
  if (value < min || value > max) {
    badRequest(`${field} 必须在 ${min}..${max}，实际: ${value}`);
  }
  return value;
}

function requirePlayerIdString(value: unknown, field: string): string {
  if (typeof value !== "string" || !/^[1-9]\d*$/.test(value)) {
    badRequest(`${field} 必须是正整数字符串，实际: ${JSON.stringify(value)}`);
  }
  if (!Number.isSafeInteger(Number(value))) {
    badRequest(`${field} 超出安全整数范围: ${value}`);
  }
  return value;
}

/** 球员 ID 数组：去重但不静默吞错——重复 / 非正整数字符串一律 400。 */
export function requireUniquePlayerIds(value: unknown, field: string): string[] {
  if (!Array.isArray(value) || value.length === 0) {
    badRequest(`${field} 必须是非空数组，实际: ${JSON.stringify(value)}`);
  }
  const ids = value.map((v) => requirePlayerIdString(v, field));
  if (new Set(ids).size !== ids.length) {
    badRequest(`${field} 存在重复球员 ID`);
  }
  return ids;
}

/** 数值化排序，保证引擎输入与 payloadHash 的确定性。 */
export function sortPlayerIds(ids: string[]): string[] {
  return [...ids].sort((a, b) => Number(a) - Number(b));
}

export function requireActorId(value: unknown): string {
  return requirePlayerIdString(value, "actorId");
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function requireUuid(value: unknown, field: string): string {
  if (typeof value !== "string" || !UUID_RE.test(value)) {
    badRequest(`${field} 必须是 UUID，实际: ${JSON.stringify(value)}`);
  }
  return value.toLowerCase();
}

/** 项目名：trim 后 1..60；undefined/null → null（调用方决定缺省值）。 */
export function parseProjectName(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") {
    badRequest(`name 必须是字符串，实际: ${JSON.stringify(value)}`);
  }
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > MAX_NAME_LENGTH) {
    badRequest(`name trim 后长度必须在 1..${MAX_NAME_LENGTH}，实际: ${trimmed.length}`);
  }
  return trimmed;
}

export function parseBooleanFlag(value: unknown, field: string, defaultValue: boolean): boolean {
  if (value === undefined || value === null) return defaultValue;
  if (typeof value !== "boolean") {
    badRequest(`${field} 必须是 boolean，实际: ${JSON.stringify(value)}`);
  }
  return value;
}

/** seed：缺省由服务器取新随机 seed；显式传必须 uint32。 */
export function resolveSeed(value: unknown): number {
  if (value === undefined || value === null) {
    return randomInt(0, MAX_SEED + 1);
  }
  return requireInt(value, "seed", 0, MAX_SEED);
}

export function parseModelOption(value: unknown): RatingModel | null {
  if (value === undefined || value === null) return null;
  if (value !== "glicko2" && value !== "legacy") {
    badRequest(`model 必须是 "glicko2" | "legacy"，实际: ${JSON.stringify(value)}`);
  }
  return value;
}

/** 日历日期 YYYY-MM-DD（真实存在）；与 repo 层同口径。 */
export function requireIsoDate(value: unknown, field: string): string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    badRequest(`${field} 必须是 YYYY-MM-DD 日历日期，实际: ${JSON.stringify(value)}`);
  }
  const t = Date.parse(`${value}T00:00:00Z`);
  if (Number.isNaN(t) || new Date(t).toISOString().slice(0, 10) !== value) {
    badRequest(`${field} 不是真实日历日期: ${value}`);
  }
  return value;
}

export function shanghaiToday(): string {
  return shanghaiLocalDateFromInstant(new Date().toISOString());
}

export function assertPlayerExists(conn: Database.Database, playerId: string): void {
  const row = conn.prepare(`SELECT 1 FROM players WHERE id = ?`).get(Number(playerId));
  if (!row) {
    throw new ScheduleServiceError(
      400,
      "player_not_found",
      `球员 ${playerId} 不存在`
    );
  }
}

export function assertPlayersExist(
  conn: Database.Database,
  playerIds: string[]
): void {
  for (const id of playerIds) assertPlayerExists(conn, id);
}

// ---------------------------------------------------------------------------
// 评分快照（一次捕获）与胜率回调
// ---------------------------------------------------------------------------

export interface WinProbabilityHandle {
  winProbability: (a1: string, a2: string, b1: string, b2: string) => number;
  meta: {
    configVersion: string | null;
    asOf: string;
    inputHash: string | null;
  };
  /**
   * 可还原胜率的选中人员评分状态：glicko2 为 Record<playerId, RatingState>，
   * legacy 为 Record<playerId, elo>。随 generation 快照落库（按名单过滤）。
   */
  playerRatings: Record<string, unknown>;
  /** glicko2 的完整 config；legacy 为 null。 */
  ratingConfig: unknown | null;
}

function ratingUnavailable(state: string, reason: string): ScheduleServiceError {
  return new ScheduleServiceError(
    409,
    "rating_unavailable",
    `rating service unavailable: ${reason}`,
    { state, reason }
  );
}

/** 全部球员 ID（string），用于把无比赛球员按默认口径补进评分快照。 */
function listAllPlayerIds(conn: Database.Database): string[] {
  const rows = conn.prepare(`SELECT id FROM players`).all() as { id: number }[];
  return rows.map((r) => String(r.id));
}

/**
 * 按模型一次捕获评分快照并构造胜率回调。
 * glicko2：ready 快照不可用（stale/unavailable/读失败）→ 409 rating_unavailable，
 * 不伪造胜率、不静默退回 legacy；legacy：predictElo 回调。
 */
export function captureWinProbability(
  model: RatingModel,
  conn: Database.Database,
  asOf: string
): WinProbabilityHandle {
  if (model === "glicko2") {
    let snapshot: ReturnType<typeof loadGlickoSnapshot>;
    try {
      snapshot = loadGlickoSnapshot(conn, asOf);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      throw ratingUnavailable("unavailable", `rating service failed: ${reason}`);
    }
    if (snapshot.state !== "ready") {
      const detail =
        snapshot.state === "stale"
          ? `${snapshot.reason}; last good as of ${snapshot.lastGood.asOf}`
          : snapshot.reason;
      throw ratingUnavailable(snapshot.state, detail);
    }
    const record = readRatingConfig(conn);
    if (record === null) {
      throw ratingUnavailable(
        "unavailable",
        "rating config disappeared after ready snapshot"
      );
    }
    const states = { ...snapshot.replay.current };
    const config = record.config;
    // 无比赛球员不在 replay.current 中；按 predictDoubles 的默认口径补齐，
    // 使快照对任意名单自洽可复算（resolveDoublesState 同口径）。
    for (const id of listAllPlayerIds(conn)) {
      if (states[id] === undefined) {
        states[id] = {
          r: config.initialRating,
          rd: config.initialRd,
          volatility: config.initialVolatility,
        };
      }
    }
    return {
      winProbability: (a1, a2, b1, b2) =>
        predictDoubles(
          [Number(a1), Number(a2)],
          [Number(b1), Number(b2)],
          states,
          config
        ),
      meta: {
        configVersion: snapshot.replay.configVersion,
        asOf,
        inputHash: snapshot.inputHash,
      },
      playerRatings: states,
      ratingConfig: config,
    };
  }

  let ratings: Map<number, { elo: number }>;
  try {
    ratings = recomputeAllRatings(conn);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw ratingUnavailable("unavailable", `legacy rating recompute failed: ${reason}`);
  }
  const eloRatings: Record<string, number> = Object.fromEntries(
    [...ratings].map(([id, r]) => [String(id), r.elo])
  );
  // 无比赛球员按 predictElo 的默认口径（INITIAL_RATING）补齐，快照自洽可复算。
  for (const id of listAllPlayerIds(conn)) {
    if (eloRatings[id] === undefined) eloRatings[id] = INITIAL_RATING;
  }
  return {
    winProbability: (a1, a2, b1, b2) =>
      predictElo(a1, a2, b1, b2, eloRatings).teamAWin,
    meta: { configVersion: null, asOf, inputHash: null },
    playerRatings: eloRatings,
    ratingConfig: null,
  };
}

/** 显式模型优先；缺省取站点当前模型（无配置默认 legacy）。 */
export function resolveModel(requested: RatingModel | null): RatingModel {
  if (requested !== null) return requested;
  return readRatingConfig()?.activeModel ?? "legacy";
}

// ---------------------------------------------------------------------------
// 通用工具
// ---------------------------------------------------------------------------

export function hashPayload(payload: unknown): string {
  return createHash("sha256").update(JSON.stringify(payload)).digest("hex");
}

/** 给项目详情补充球员姓名映射（成员、场次参与者、操作者）。 */
export function buildNamesMap(
  conn: Database.Database,
  detail: ProjectDetail
): Record<string, string> {
  const ids = new Set<string>();
  for (const m of detail.members) ids.add(m.playerId);
  for (const f of [...detail.fixtures, ...detail.superseded]) {
    ids.add(f.pa1);
    ids.add(f.pa2);
    ids.add(f.pb1);
    ids.add(f.pb2);
    if (f.markedBy) ids.add(f.markedBy);
  }
  ids.add(detail.createdBy);
  if (ids.size === 0) return {};
  const rows = conn
    .prepare(`SELECT id, name FROM players`)
    .all() as Array<{ id: number; name: string }>;
  const byId = new Map(rows.map((r) => [String(r.id), r.name]));
  const names: Record<string, string> = {};
  for (const id of ids) {
    const name = byId.get(id);
    if (name !== undefined) names[id] = name;
  }
  return names;
}

// ---------------------------------------------------------------------------
// createProject（POST /api/schedule/projects）
// ---------------------------------------------------------------------------

export interface CreateProjectServiceResult {
  project: ProjectDetail;
  names: Record<string, string>;
  replayed: boolean;
}

export function createProjectService(params: {
  name?: unknown;
  playerIds?: unknown;
  matches?: unknown;
  avoidPlayedRematches?: unknown;
  seed?: unknown;
  model?: unknown;
  actorId?: unknown;
  requestId?: unknown;
  db?: Database.Database;
}): CreateProjectServiceResult {
  const conn = resolveDb(params.db);
  const name = parseProjectName(params.name) ?? `${shanghaiToday()} 配对`;
  const playerIds = sortPlayerIds(
    requireUniquePlayerIds(params.playerIds, "playerIds")
  );
  if (playerIds.length < MIN_PLAYERS || playerIds.length > MAX_PLAYERS) {
    badRequest(`playerIds 人数必须在 ${MIN_PLAYERS}..${MAX_PLAYERS}，实际: ${playerIds.length}`);
  }
  const matches = requireInt(params.matches, "matches", MIN_CREATE_MATCHES, MAX_MATCHES);
  const avoidPlayedRematches = parseBooleanFlag(
    params.avoidPlayedRematches,
    "avoidPlayedRematches",
    true
  );
  const seed = resolveSeed(params.seed);
  const model = resolveModel(parseModelOption(params.model));
  const actorId = requireActorId(params.actorId);
  const requestId = requireUuid(params.requestId, "requestId");
  assertPlayerExists(conn, actorId);
  assertPlayersExist(conn, playerIds);

  // 评分快照一次捕获；不可用直接 409，不进引擎。
  const asOf = new Date().toISOString();
  const handle = captureWinProbability(model, conn, asOf);

  const engineInput: SchedulingInput = {
    playerIds,
    matches,
    effectiveAppearances: Object.fromEntries(playerIds.map((id) => [id, 0])),
    playedMatchCounts: {},
    avoidPlayedRematches,
    seed,
    algorithmVersion: ALGORITHM_VERSION,
    objectiveVersion: OBJECTIVE_VERSION,
    winProbability: handle.winProbability,
  };
  const engineResult = solveScheduling(engineInput);
  const fixtureWinRates = computeFixtureWinRates(engineResult, handle);

  const payloadHash = hashPayload({
    v: 1,
    name,
    model,
    playerIds,
    matches,
    avoidPlayedRematches,
    seed,
    algorithmVersion: ALGORITHM_VERSION,
    objectiveVersion: OBJECTIVE_VERSION,
  });

  const created = createProject(
    {
      name,
      model,
      createdBy: actorId,
      requestId,
      payloadHash,
      playerIds,
      engineInput: buildGenerationSnapshot(
        stripCallback(engineInput),
        handle,
        model,
        playerIds
      ),
      fixtureWinRates,
    },
    engineResult,
    conn
  );

  const project = getProject(created.id, conn);
  return {
    project,
    names: buildNamesMap(conn, project),
    replayed: created.replayed,
  };
}

/** input_json 快照不含函数回调（JSON.stringify 也会丢，显式剔除更干净）。 */
function stripCallback(input: SchedulingInput): Record<string, unknown> {
  const rest: Record<string, unknown> = { ...input };
  delete rest.winProbability;
  return rest;
}

/**
 * generation 落库快照：完整引擎输入（去回调）+ 可还原胜率的选中人员评分/config、
 * model/configVersion/asOf/inputHash 与固定搜索预算。不包含服务器捕获时刻以外的
 * 易变数据之外的字段——注意 payloadHash 必须只覆盖客户端 payload，不含本快照，
 * 否则同一 requestId 的网络重试会因 asOf 漂移被误判为 idempotency_conflict。
 */
function buildGenerationSnapshot(
  base: Record<string, unknown>,
  handle: WinProbabilityHandle,
  model: RatingModel,
  playerIds: string[]
): Record<string, unknown> {
  const selected: Record<string, unknown> = {};
  for (const id of playerIds) {
    if (id in handle.playerRatings) selected[id] = handle.playerRatings[id];
  }
  return {
    ...base,
    rating: {
      model,
      configVersion: handle.meta.configVersion,
      asOf: handle.meta.asOf,
      inputHash: handle.meta.inputHash,
      config: handle.ratingConfig,
      playerRatings: selected,
    },
    budget: SEARCH_BUDGET,
  };
}

/** 用同一评分快照回调逐场计算生成胜率（随 fixture 落库的原快照，不冒充最新预测）。 */
function computeFixtureWinRates(
  result: SchedulingResult,
  handle: WinProbabilityHandle
): number[] {
  return result.schedule.map((m) =>
    handle.winProbability(m.a1, m.a2, m.b1, m.b2)
  );
}

// ---------------------------------------------------------------------------
// regenerate（POST /api/schedule/projects/[id]/generate）
// ---------------------------------------------------------------------------

export interface RegenerateServiceResult {
  project: ProjectDetail;
  names: Record<string, string>;
  generationId: number;
  fixtures: FixtureDto[];
  supersededCount: number;
  joinedPlayers: string[];
  leftPlayers: string[];
  replayed: boolean;
}

export function regenerateProjectService(params: {
  projectId: number;
  playerIds?: unknown;
  matches?: unknown;
  avoidPlayedRematches?: unknown;
  seed?: unknown;
  expectedRevision?: unknown;
  actorId?: unknown;
  requestId?: unknown;
  db?: Database.Database;
}): RegenerateServiceResult {
  const conn = resolveDb(params.db);
  const projectId = requireInt(params.projectId, "projectId", 1, Number.MAX_SAFE_INTEGER);
  const expectedRevision = requireInt(
    params.expectedRevision,
    "expectedRevision",
    0,
    Number.MAX_SAFE_INTEGER
  );
  const requestId = requireUuid(params.requestId, "requestId");
  const actorId = requireActorId(params.actorId);
  const playerIds = sortPlayerIds(
    requireUniquePlayerIds(params.playerIds, "playerIds")
  );
  // matches=0 合法：只保存名单、清空未打——此时允许 <4 人（合同：
  // 「少于 4 人不能生成比赛，但允许以 matches=0 保存名单并清空未打安排」），
  // 名单仍需非空且不超上限；matches>0 才要求 4..64 人。
  const matches = requireInt(params.matches, "matches", 0, MAX_MATCHES);
  const minPlayers = matches > 0 ? MIN_PLAYERS : 1;
  if (playerIds.length < minPlayers || playerIds.length > MAX_PLAYERS) {
    badRequest(
      `playerIds 人数必须在 ${minPlayers}..${MAX_PLAYERS}，实际: ${playerIds.length}`
    );
  }
  const avoidPlayedRematches = parseBooleanFlag(
    params.avoidPlayedRematches,
    "avoidPlayedRematches",
    true
  );
  const seed = resolveSeed(params.seed);
  assertPlayerExists(conn, actorId);

  // 事务外读取一致快照（成员/已打历史/有效场次）。
  const detail = getProject(projectId, conn);
  assertPlayersExist(conn, playerIds);
  const model = detail.model as RatingModel;

  // effectiveAppearances = fairness_base + actualPlayed - played_at_join；
  // 新加入/重新加入者起点取继续在场者最低 effective（与 repo 落库口径一致）。
  const activeMap = new Map(detail.members.map((m) => [m.playerId, m]));
  const effectiveAppearances: Record<string, number> = {};
  let minEffective = Infinity;
  for (const id of playerIds) {
    const member = activeMap.get(id);
    if (member) {
      const effective =
        member.fairnessBase + member.actualPlayed - member.playedAtJoin;
      effectiveAppearances[id] = effective;
      if (effective < minEffective) minEffective = effective;
    }
  }
  const joinBase = Number.isFinite(minEffective) ? minEffective : 0;
  for (const id of playerIds) {
    if (!activeMap.has(id)) effectiveAppearances[id] = joinBase;
  }

  // 已打完整对阵历史：只统计本项目 played_unscored + recorded 的 canonical 键。
  const playedMatchCounts: Record<string, number> = {};
  for (const f of detail.fixtures) {
    if (f.status === "played_unscored" || f.status === "recorded") {
      const key = canonicalMatchKey(f.pa1, f.pa2, f.pb1, f.pb2);
      playedMatchCounts[key] = (playedMatchCounts[key] ?? 0) + 1;
    }
  }

  const asOf = new Date().toISOString();
  // matches=0 只是保存名单/清空未打，不生成对阵——不需要评分快照，
  // 避免 glicko2 快照不可用时连名单都存不下。
  const handle = matches > 0 ? captureWinProbability(model, conn, asOf) : null;

  let engineResult: SchedulingResult | null = null;
  let engineInputSnapshot: Record<string, unknown> = {
    v: 1,
    playerIds,
    matches,
    avoidPlayedRematches,
    seed,
    effectiveAppearances,
    playedMatchCounts,
    algorithmVersion: ALGORITHM_VERSION,
    objectiveVersion: OBJECTIVE_VERSION,
  };
  let fixtureWinRates: number[] | undefined;
  if (matches > 0) {
    const engineInput: SchedulingInput = {
      playerIds,
      matches,
      effectiveAppearances,
      playedMatchCounts,
      avoidPlayedRematches,
      seed,
      algorithmVersion: ALGORITHM_VERSION,
      objectiveVersion: OBJECTIVE_VERSION,
      winProbability: handle!.winProbability,
    };
    engineResult = solveScheduling(engineInput);
    fixtureWinRates = computeFixtureWinRates(engineResult, handle!);
  } else {
    engineInputSnapshot = {
      ...engineInputSnapshot,
      effectiveAppearances: null,
      playedMatchCounts: null,
    };
  }

  const payloadHash = hashPayload(engineInputSnapshot);
  const storedSnapshot =
    handle !== null
      ? buildGenerationSnapshot(engineInputSnapshot, handle, model, playerIds)
      : engineInputSnapshot;
  const result = regenerate(
    {
      projectId,
      expectedRevision,
      requestId,
      payloadHash,
      newPlayerIds: playerIds,
      engineInput: storedSnapshot,
      engineResult,
      fixtureWinRates,
      actorId,
    },
    conn
  );

  const project = getProject(projectId, conn);
  return {
    project,
    names: buildNamesMap(conn, project),
    generationId: result.generationId,
    fixtures: result.fixtures,
    supersededCount: result.supersededCount,
    joinedPlayers: result.joinedPlayers,
    leftPlayers: result.leftPlayers,
    replayed: result.replayed,
  };
}

// ---------------------------------------------------------------------------
// rename（PATCH /api/schedule/projects/[id]）：仅改名，revision CAS
// ---------------------------------------------------------------------------

export interface RenameProjectServiceResult {
  id: number;
  name: string;
  revision: number;
}

export function renameProjectService(params: {
  projectId: number;
  name?: unknown;
  expectedRevision?: unknown;
  actorId?: unknown;
  db?: Database.Database;
}): RenameProjectServiceResult {
  const conn = resolveDb(params.db);
  const projectId = requireInt(params.projectId, "projectId", 1, Number.MAX_SAFE_INTEGER);
  const name = parseProjectName(params.name);
  if (name === null) {
    badRequest("name 不能为空");
  }
  const expectedRevision = requireInt(
    params.expectedRevision,
    "expectedRevision",
    0,
    Number.MAX_SAFE_INTEGER
  );
  const actorId = requireActorId(params.actorId);
  assertPlayerExists(conn, actorId);

  const tx = conn.transaction(() => {
    const row = conn
      .prepare(`SELECT revision FROM schedule_projects WHERE id = ?`)
      .get(projectId) as { revision: number } | undefined;
    if (!row) {
      throw new ScheduleServiceError(404, "not_found", `项目 ${projectId} 不存在`);
    }
    if (row.revision !== expectedRevision) {
      throw new ScheduleServiceError(
        409,
        "revision_conflict",
        `项目 revision 已变化：期望 ${expectedRevision}，当前 ${row.revision}`,
        { projectRevision: row.revision }
      );
    }
    conn
      .prepare(
        `UPDATE schedule_projects
         SET name = ?, revision = revision + 1, updated_at = datetime('now')
         WHERE id = ?`
      )
      .run(name, projectId);
    return row.revision + 1;
  });
  const revision = tx.immediate();
  return { id: projectId, name: name as string, revision };
}

// ---------------------------------------------------------------------------
// markPlayed（POST .../matches/[fixtureId]/played）：仅 pending → played_unscored
// ---------------------------------------------------------------------------

function alreadyRecorded(fixture: FixtureDto): ScheduleServiceError {
  return new ScheduleServiceError(
    409,
    "already_recorded",
    `场次 ${fixture.id} 已录入比分`,
    { fixture }
  );
}

function fixtureSuperseded(fixture: FixtureDto): ScheduleServiceError {
  return new ScheduleServiceError(
    409,
    "fixture_superseded",
    `场次 ${fixture.id} 已被取代（superseded）`,
    { fixture }
  );
}

function fixtureRevisionConflict(
  fixture: FixtureDto,
  expectedRevision: number
): ScheduleServiceError {
  return new ScheduleServiceError(
    409,
    "revision_conflict",
    `场次 revision 已变化：期望 ${expectedRevision}，当前 ${fixture.revision}`,
    { fixture }
  );
}

/** 读取单场 fixture（不存在 → 404 not_found，不泄露存在性）。 */
export function readFixture(
  conn: Database.Database,
  projectId: number,
  fixtureId: number
): FixtureDto {
  const row = conn
    .prepare(
      `SELECT
         f.id,
         f.project_id AS projectId,
         f.generation_id AS generationId,
         f.display_no AS displayNo,
         f.pa1, f.pa2, f.pb1, f.pb2,
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
       WHERE f.id = ? AND f.project_id = ?`
    )
    .get(fixtureId, projectId) as FixtureDto | undefined;
  if (!row) {
    throw new ScheduleServiceError(
      404,
      "not_found",
      `项目 ${projectId} 下不存在场次 ${fixtureId}`
    );
  }
  return row;
}

/**
 * 预检 fixture 状态与 revision，给出精确 409 语义
 * （already_recorded / fixture_superseded / revision_conflict）。
 */
export function preflightMutableFixture(
  conn: Database.Database,
  projectId: number,
  fixtureId: number,
  expectedFixtureRevision: number
): FixtureDto {
  const fixture = readFixture(conn, projectId, fixtureId);
  if (fixture.status === "recorded") throw alreadyRecorded(fixture);
  if (fixture.status === "superseded") throw fixtureSuperseded(fixture);
  if (fixture.revision !== expectedFixtureRevision) {
    throw fixtureRevisionConflict(fixture, expectedFixtureRevision);
  }
  return fixture;
}

/**
 * repo 在事务内重查出的状态/revision 竞态：按当前状态重分类，
 * 让并发失败的请求读到对方的已提交结果（含比分 / 最新 revision）。
 */
export function classifyFixtureRace(
  conn: Database.Database,
  projectId: number,
  fixtureId: number,
  expectedRevision: number,
  original: unknown
): never {
  if (isScheduleServiceError(original)) throw original;
  if (original instanceof MatchValidationError) {
    throw new ScheduleServiceError(400, "validation_error", original.message);
  }
  if (isScheduleProjectError(original)) {
    let current: FixtureDto | null = null;
    try {
      current = readFixture(conn, projectId, fixtureId);
    } catch {
      current = null;
    }
    if (current !== null) {
      if (current.status === "recorded") throw alreadyRecorded(current);
      if (current.status === "superseded") throw fixtureSuperseded(current);
      if (current.revision !== expectedRevision) {
        throw fixtureRevisionConflict(current, expectedRevision);
      }
      throw new ScheduleServiceError(409, "state_conflict", original.message, {
        fixture: current,
      });
    }
    if (original instanceof ScheduleProjectError) {
      if (original.code === "match_mismatch") {
        throw new ScheduleServiceError(409, "match_mismatch", original.message);
      }
      throw new ScheduleServiceError(409, "state_conflict", original.message);
    }
  }
  throw original;
}

export function markPlayedService(params: {
  projectId: number;
  fixtureId: number;
  expectedFixtureRevision?: unknown;
  actorId?: unknown;
  playedAt?: unknown;
  db?: Database.Database;
}): { fixture: FixtureDto } {
  const conn = resolveDb(params.db);
  const projectId = requireInt(params.projectId, "projectId", 1, Number.MAX_SAFE_INTEGER);
  const fixtureId = requireInt(params.fixtureId, "fixtureId", 1, Number.MAX_SAFE_INTEGER);
  const expectedRevision = requireInt(
    params.expectedFixtureRevision,
    "expectedFixtureRevision",
    0,
    Number.MAX_SAFE_INTEGER
  );
  const actorId = requireActorId(params.actorId);
  assertPlayerExists(conn, actorId);
  const playedAt =
    params.playedAt === undefined || params.playedAt === null
      ? shanghaiToday()
      : requireIsoDate(params.playedAt, "playedAt");

  preflightMutableFixture(conn, projectId, fixtureId, expectedRevision);

  try {
    const fixture = markPlayed(
      {
        projectId,
        fixtureId,
        expectedFixtureRevision: expectedRevision,
        actorId,
        playedAt,
      },
      conn
    );
    return { fixture };
  } catch (error) {
    classifyFixtureRace(conn, projectId, fixtureId, expectedRevision, error);
  }
}

// ---------------------------------------------------------------------------
// 旧入口联动
// ---------------------------------------------------------------------------

/**
 * 项目记分（POST .../matches/[fixtureId]/score）：
 * 服务端从 DB 取阵容（不信任客户端）；事务内插入 matches 并 attachScore；
 * 提交后走公共录分服务构建评分结果 + revalidation。评分构建失败降级为
 * pending/空数组，比赛已保存仍返回 201。
 */
export function recordFixtureScoreService(params: {
  projectId: number;
  fixtureId: number;
  scoreA?: unknown;
  scoreB?: unknown;
  expectedFixtureRevision?: unknown;
  actorId?: unknown;
  playedAt?: unknown;
  db?: Database.Database;
}): RecordMatchResult {
  const conn = resolveDb(params.db);
  const projectId = requireInt(params.projectId, "projectId", 1, Number.MAX_SAFE_INTEGER);
  const fixtureId = requireInt(params.fixtureId, "fixtureId", 1, Number.MAX_SAFE_INTEGER);
  const expectedRevision = requireInt(
    params.expectedFixtureRevision,
    "expectedFixtureRevision",
    0,
    Number.MAX_SAFE_INTEGER
  );
  const actorId = requireActorId(params.actorId);
  assertPlayerExists(conn, actorId);
  const scoreA = requireInt(params.scoreA, "scoreA", 0, Number.MAX_SAFE_INTEGER);
  const scoreB = requireInt(params.scoreB, "scoreB", 0, Number.MAX_SAFE_INTEGER);
  // 沿用既有整数比分规则（非负整数且不相等）。
  try {
    assertValidScores(scoreA, scoreB);
  } catch (error) {
    if (error instanceof MatchValidationError) {
      throw new ScheduleServiceError(400, "validation_error", error.message);
    }
    throw error;
  }

  // 服务端取阵容 + 精确 409 语义预检。
  const fixture = preflightMutableFixture(conn, projectId, fixtureId, expectedRevision);

  // 日期口径：pending 缺省上海当天（可显式传）；played_unscored 沿用原「打过」日期。
  let playedAt: string;
  if (fixture.status === "pending") {
    playedAt =
      params.playedAt === undefined || params.playedAt === null
        ? shanghaiToday()
        : requireIsoDate(params.playedAt, "playedAt");
  } else {
    playedAt = fixture.playedAt as string;
  }

  try {
    return recordMatch({
      pa1: Number(fixture.pa1),
      pa2: Number(fixture.pa2),
      pb1: Number(fixture.pb1),
      pb2: Number(fixture.pb2),
      scoreA,
      scoreB,
      playedAt,
      enteredBy: Number(actorId),
      db: conn,
      fixtureLink: {
        projectId,
        fixtureId,
        expectedFixtureRevision: expectedRevision,
        actorId,
      },
    });
  } catch (error) {
    classifyFixtureRace(conn, projectId, fixtureId, expectedRevision, error);
  }
}

/**
 * admin 改分后同步：若比赛被 fixture 引用（recorded），让项目 revision++
 * 并刷新 updated_at，客户端据此刷新；fixture 保持 recorded，阵容与
 * generated_win_rate 不动。
 */
export function noteMatchScoreEdited(
  matchId: number,
  db?: Database.Database
): void {
  const conn = resolveDb(db);
  const fixture = conn
    .prepare(`SELECT project_id AS projectId FROM schedule_fixtures WHERE match_id = ?`)
    .get(matchId) as { projectId: number } | undefined;
  if (!fixture) return;
  conn
    .prepare(
      `UPDATE schedule_projects
       SET revision = revision + 1, updated_at = datetime('now')
       WHERE id = ?`
    )
    .run(fixture.projectId);
}

export interface MergePlayerResult {
  affectedProjects: number[];
}

/**
 * admin 球员合并（含排阵联动）：
 * 同一事务预检并迁移——
 * 1) 目标 ID 不存在 → 拒绝；
 * 2) 合并后任一真实比赛或有效/历史 fixture 出现同场同人两次 → 拒绝；
 * 3) 两 ID 在同一项目均有成员记录 → 拒绝（说明两名球员都有项目参与记录）；
 * 无冲突时更新项目成员行、fixture 槽位、generations.created_by、
 * fixtures.marked_by、projects.created_by 及 matches 既有引用，
 * 受影响项目 revision++ / updated_at；旧 generation 输入快照（input_json）
 * 保持不可变。任一步失败整体回滚，禁止部分迁移。
 */
export function mergePlayerWithScheduleRefs(
  fromId: number,
  toId: number,
  db?: Database.Database
): MergePlayerResult {
  const conn = resolveDb(db);
  if (!Number.isSafeInteger(fromId) || !Number.isSafeInteger(toId)) {
    badRequest("球员 ID 必须是安全整数");
  }
  if (fromId === toId) {
    badRequest("不能将球员合并到自身");
  }
  const toRow = conn.prepare(`SELECT 1 FROM players WHERE id = ?`).get(toId);
  if (!toRow) {
    throw new ScheduleServiceError(
      400,
      "player_not_found",
      `目标球员 ${toId} 不存在`
    );
  }

  const fromText = String(fromId);
  const toText = String(toId);

  const tx = conn.transaction(() => {
    // 2) 同场双人对撞预检：matches。
    const matchRows = conn
      .prepare(
        `SELECT id, pa1, pa2, pb1, pb2 FROM matches
         WHERE pa1 = ? OR pa2 = ? OR pb1 = ? OR pb2 = ?`
      )
      .all(fromId, fromId, fromId, fromId) as Array<{
      id: number;
      pa1: number;
      pa2: number;
      pb1: number;
      pb2: number;
    }>;
    for (const row of matchRows) {
      if ([row.pa1, row.pa2, row.pb1, row.pb2].includes(toId)) {
        throw new ScheduleServiceError(
          400,
          "state_conflict",
          `合并后比赛 ${row.id} 将出现同场重复球员（${fromId} → ${toId}），已拒绝`
        );
      }
    }

    // 2) 同场双人对撞预检：有效/历史 fixture（槽位为 TEXT ID）。
    const fixtureRows = conn
      .prepare(
        `SELECT id, pa1, pa2, pb1, pb2 FROM schedule_fixtures
         WHERE pa1 = ? OR pa2 = ? OR pb1 = ? OR pb2 = ?`
      )
      .all(fromText, fromText, fromText, fromText) as Array<{
      id: number;
      pa1: string;
      pa2: string;
      pb1: string;
      pb2: string;
    }>;
    for (const row of fixtureRows) {
      if ([row.pa1, row.pa2, row.pb1, row.pb2].includes(toText)) {
        throw new ScheduleServiceError(
          400,
          "state_conflict",
          `合并后场次 ${row.id} 将出现同场重复球员（${fromId} → ${toId}），已拒绝`
        );
      }
    }

    // 3) 同一项目双人成员预检。
    const sharedProject = conn
      .prepare(
        `SELECT a.project_id AS pid FROM schedule_project_players a
         JOIN schedule_project_players b
           ON a.project_id = b.project_id AND b.player_id = ?
         WHERE a.player_id = ?`
      )
      .get(toId, fromId) as { pid: number } | undefined;
    if (sharedProject) {
      throw new ScheduleServiceError(
        400,
        "state_conflict",
        `球员 ${fromId} 与 ${toId} 在项目 ${sharedProject.pid} 中均有成员记录，请先处理项目参与记录`
      );
    }

    // 受影响项目集合：成员、fixture 槽位、created_by、generations.created_by。
    const affected = new Set<number>();
    const memberProjects = conn
      .prepare(
        `SELECT project_id AS pid FROM schedule_project_players WHERE player_id IN (?, ?)`
      )
      .all(fromId, toId) as Array<{ pid: number }>;
    for (const row of memberProjects) affected.add(row.pid);
    const fixtureProjects = conn
      .prepare(
        `SELECT DISTINCT project_id AS pid FROM schedule_fixtures
         WHERE pa1 IN (?, ?) OR pa2 IN (?, ?) OR pb1 IN (?, ?) OR pb2 IN (?, ?)`
      )
      .all(
        fromText, toText, fromText, toText, fromText, toText, fromText, toText
      ) as Array<{ pid: number }>;
    for (const row of fixtureProjects) affected.add(row.pid);
    const createdProjects = conn
      .prepare(
        `SELECT id AS pid FROM schedule_projects WHERE created_by IN (?, ?)`
      )
      .all(fromText, toText) as Array<{ pid: number }>;
    for (const row of createdProjects) affected.add(row.pid);
    const generationProjects = conn
      .prepare(
        `SELECT DISTINCT project_id AS pid FROM schedule_generations WHERE created_by IN (?, ?)`
      )
      .all(fromText, toText) as Array<{ pid: number }>;
    for (const row of generationProjects) affected.add(row.pid);

    // 4) 排阵引用迁移必须先于旧球员删除（players 行被 schedule_* 多列 FK 引用），
    //    matches 引用与旧球员删除由 repo.mergePlayers 收尾。
    //    注意：repo.mergePlayers 自带事务，在 better-sqlite3 中作为嵌套 savepoint 并入本事务。
    conn
      .prepare(`UPDATE schedule_project_players SET player_id = ? WHERE player_id = ?`)
      .run(toId, fromId);
    for (const col of ["pa1", "pa2", "pb1", "pb2", "marked_by"] as const) {
      conn
        .prepare(`UPDATE schedule_fixtures SET ${col} = ? WHERE ${col} = ?`)
        .run(toText, fromText);
    }
    conn
      .prepare(`UPDATE schedule_generations SET created_by = ? WHERE created_by = ?`)
      .run(toText, fromText);
    conn
      .prepare(`UPDATE schedule_projects SET created_by = ? WHERE created_by = ?`)
      .run(toText, fromText);

    mergePlayers(fromId, toId, conn);

    // 6) 受影响项目 revision++ / updated_at。
    for (const pid of affected) {
      conn
        .prepare(
          `UPDATE schedule_projects
           SET revision = revision + 1, updated_at = datetime('now')
           WHERE id = ?`
        )
        .run(pid);
    }
    return { affectedProjects: [...affected].sort((a, b) => a - b) };
  });

  return tx.immediate();
}

/**
 * admin 球员删除预检：有任何项目成员（含历史）/fixture/操作者引用时拒绝。
 * 返回拒绝说明；无引用返回 null（调用方继续原有 matches 检查与删除）。
 */
export function scheduleReferenceBlockReason(
  playerId: number,
  db?: Database.Database
): string | null {
  const conn = resolveDb(db);
  const sid = String(playerId);
  const checks: Array<{ label: string; sql: string; params: (string | number)[] }> = [
    {
      label: "项目成员记录",
      sql: `SELECT 1 FROM schedule_project_players WHERE player_id = ? LIMIT 1`,
      params: [playerId],
    },
    {
      label: "排阵场次出场",
      sql: `SELECT 1 FROM schedule_fixtures WHERE pa1 = ? OR pa2 = ? OR pb1 = ? OR pb2 = ? LIMIT 1`,
      params: [sid, sid, sid, sid],
    },
    {
      label: "项目创建者",
      sql: `SELECT 1 FROM schedule_projects WHERE created_by = ? LIMIT 1`,
      params: [sid],
    },
    {
      label: "代次创建者",
      sql: `SELECT 1 FROM schedule_generations WHERE created_by = ? LIMIT 1`,
      params: [sid],
    },
    {
      label: "场次登记操作者",
      sql: `SELECT 1 FROM schedule_fixtures WHERE marked_by = ? LIMIT 1`,
      params: [sid],
    },
  ];
  for (const check of checks) {
    const hit = conn.prepare(check.sql).get(...check.params);
    if (hit) {
      return `该球员存在排阵项目${check.label}，请先合并到其他球员`;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// 错误 → HTTP 映射
// ---------------------------------------------------------------------------

export interface HttpErrorShape {
  status: number;
  body: Record<string, unknown>;
}

/**
 * repo 层 code → HTTP：
 * not_found→404；validation_error/player_not_found→400；
 * idempotency_conflict→409；revision_conflict→409 携带 projectRevision；
 * state_conflict/match_mismatch→409（服务层已换成精确语义时不会走到这）。
 */
export function toScheduleHttp(err: unknown): HttpErrorShape {
  if (isScheduleServiceError(err)) {
    return {
      status: err.status,
      body: { error: err.message, code: err.code, ...err.extra },
    };
  }
  if (err instanceof ScheduleProjectError) {
    switch (err.code) {
      case "not_found":
        return { status: 404, body: { error: err.message, code: "not_found" } };
      case "validation_error":
        return {
          status: 400,
          body: { error: err.message, code: "validation_error" },
        };
      case "player_not_found":
        return {
          status: 400,
          body: { error: err.message, code: "player_not_found" },
        };
      case "idempotency_conflict":
        return {
          status: 409,
          body: { error: err.message, code: "idempotency_conflict" },
        };
      case "revision_conflict":
        return {
          status: 409,
          body: {
            error: err.message,
            code: "revision_conflict",
            projectRevision: err.currentRevision,
          },
        };
      case "state_conflict":
        return { status: 409, body: { error: err.message, code: "state_conflict" } };
      case "match_mismatch":
        return { status: 409, body: { error: err.message, code: "match_mismatch" } };
    }
  }
  if (err instanceof MatchValidationError) {
    return { status: 400, body: { error: err.message, code: err.code } };
  }
  // SQLite 并发/约束：明确可重试或需刷新的 5xx/409。
  const sqliteCode =
    err !== null && typeof err === "object" && "code" in err
      ? String((err as { code: unknown }).code)
      : "";
  if (sqliteCode === "SQLITE_BUSY" || sqliteCode === "SQLITE_LOCKED") {
    const message = err instanceof Error ? err.message : String(err);
    return {
      status: 503,
      body: { error: message, code: "db_busy", retryable: true },
    };
  }
  if (sqliteCode.startsWith("SQLITE_CONSTRAINT")) {
    const message = err instanceof Error ? err.message : String(err);
    return { status: 409, body: { error: message, code: "conflict" } };
  }
  const message = err instanceof Error ? err.message : "Unknown error";
  return { status: 500, body: { error: message, code: "internal_error" } };
}
