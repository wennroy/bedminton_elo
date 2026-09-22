import { createHash } from "node:crypto";
import type Database from "better-sqlite3";
import { getDb } from "./db";
import { readRatingConfig, type RatingConfigRecord } from "./rating-config";
import { listPlayers, listRawMatches, type Player, type RawMatch } from "./repo";
import {
  ratingSegmentAt,
  shanghaiLocalDateFromInstant,
} from "./ratings/calendar";
import { ratingConfigVersion } from "./ratings/config";
import { replayRatings } from "./ratings/replay";
import type {
  PlayerId,
  RatingMatch,
  RatingReplay,
  RatingSegment,
} from "./ratings/types";

/**
 * 统一评分服务结果（共享契约）。stale 明确给出旧成功快照，不用于新预测；
 * unavailable 不伪造 1000 分，也不静默退回 Legacy。
 */
export type RatingServiceResult =
  | { state: "ready"; model: "glicko2"; inputHash: string; replay: RatingReplay }
  | { state: "stale"; model: "glicko2"; reason: string; lastGood: RatingReplay }
  | { state: "unavailable"; model: "glicko2"; reason: string };

/** 最近成功快照的 meta 键前缀；完整键再拼接配置版本。 */
export const RATING_LAST_GOOD_KEY_PREFIX = "ratings.last-good.v1:";

export function ratingLastGoodKey(configVersion: string): string {
  return `${RATING_LAST_GOOD_KEY_PREFIX}${configVersion}`;
}

/** meta 中保存的最近成功快照；值为完整可序列化结果。 */
export interface LastGoodSnapshot {
  inputHash: string;
  asOf: string;
  replay: RatingReplay;
}

interface NormalizedRatingInput {
  configVersion: string;
  players: ReadonlyArray<{ id: PlayerId; name: string }>;
  matches: readonly RatingMatch[];
}

interface ConsistentSnapshot {
  record: RatingConfigRecord | null;
  players: Player[];
  matches: RawMatch[];
}

function resolveDb(db?: Database.Database): Database.Database {
  return db ?? getDb();
}

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

/**
 * 数据指纹：覆盖配置版本、球员目录（id+name，改名即失效）与全部比赛
 * 事实（id/playedAt/createdAt 录入顺序/比分/四方参与者 ID）。
 * 改分、撤回、合并、改名、换参数版本都会改变指纹。
 */
export function computeGlickoInputHash(input: NormalizedRatingInput): string {
  return sha256(JSON.stringify(input));
}

function normalizeSnapshot(
  snapshot: ConsistentSnapshot,
  configVersion: string
): NormalizedRatingInput {
  const players = snapshot.players
    .map((player) => ({ id: player.id, name: player.name }))
    .sort((first, second) => first.id - second.id);
  const matches: RatingMatch[] = snapshot.matches.map((row) => ({
    id: row.id,
    playedAt: row.playedAt,
    createdAt: row.createdAt,
    teamA: [row.pa1, row.pa2],
    teamB: [row.pb1, row.pb2],
    scoreA: row.scoreA,
    scoreB: row.scoreB,
  }));
  return { configVersion, players, matches };
}

/**
 * 进程内 memo：同一连接、同一输入指纹 + 上海日期 + asOf 所属区段时
 * 不重复全历史重放。memo 只缓存成功结果，失败路径不落 memo，
 * 下一次调用会重试计算。跨连接（多进程/重启）由 meta 最近成功快照负责，
 * 且必须指纹一致才复用。
 */
const snapshotMemo = new WeakMap<Database.Database, Map<string, RatingReplay>>();

function memoFor(conn: Database.Database): Map<string, RatingReplay> {
  let memo = snapshotMemo.get(conn);
  if (memo === undefined) {
    memo = new Map();
    snapshotMemo.set(conn, memo);
  }
  return memo;
}

/**
 * 读取 Glicko-2 评分快照。asOf 由调用方一次捕获传入，服务内部不取时间。
 *
 * 判别表（计算失败 = replayRatings 抛数值/校验错误）：
 * - 无配置（未初始化）→ unavailable（不冒充 Legacy 结果）；
 * - 有配置即可重放，空历史返回 ready（引擎对空历史正常输出）；
 * - 成功 → ready，并原子写入最近成功快照（计算/序列化异常保留旧值）；
 * - 失败且存在「inputHash 与当前输入一致」的最近成功快照（同输入曾成功、
 *   本次重算失败，理论上罕见）→ 直接复用该快照 ready（其 replay.asOf
 *   如实反映旧时点）；
 * - 失败且缓存是旧输入的 → stale（lastGood 带旧 asOf，不得用于新预测）；
 * - 失败且无缓存或缓存序列化损坏 → unavailable。
 */
export function loadGlickoSnapshot(
  db: Database.Database | undefined,
  asOf: string
): RatingServiceResult {
  const conn = resolveDb(db);
  let snapshot: ConsistentSnapshot;
  try {
    snapshot = readConsistentSnapshot(conn);
  } catch (error) {
    return {
      state: "unavailable",
      model: "glicko2",
      reason: `failed to read rating input snapshot: ${errorMessage(error)}`,
    };
  }
  if (snapshot.record === null) {
    return {
      state: "unavailable",
      model: "glicko2",
      reason: "rating config not initialized",
    };
  }

  const { config } = snapshot.record;
  const configVersion = ratingConfigVersion(config);
  const normalized = normalizeSnapshot(snapshot, configVersion);
  const inputHash = computeGlickoInputHash(normalized);

  // asOf 是连续时间：缓存有效性以「上海日期 + asOf 所属区段」为界，
  // 跨周界/季界必须重算，不能复用旧段结果；同边界内才允许复用。
  const shanghaiDate = shanghaiLocalDateFromInstant(asOf);
  const segment: RatingSegment = ratingSegmentAt(asOf, config.firstSeasonStart);

  const key = `${inputHash}|${shanghaiDate}|${segment.id}`;
  const memoized = memoFor(conn).get(key);
  if (memoized !== undefined) {
    return { state: "ready", model: "glicko2", inputHash, replay: memoized };
  }

  const cached = readLastGoodSnapshot(conn, configVersion);
  if (
    cached !== undefined &&
    isCacheUsable(cached, inputHash, shanghaiDate, segment.id)
  ) {
    memoFor(conn).set(key, cached.replay);
    return { state: "ready", model: "glicko2", inputHash, replay: cached.replay };
  }

  try {
    const replay = replayRatings(normalized.matches, normalized.players.map((p) => p.id), {
      config,
      asOf,
    });
    writeLastGoodSnapshot(conn, configVersion, { inputHash, asOf, replay });
    memoFor(conn).set(key, replay);
    return { state: "ready", model: "glicko2", inputHash, replay };
  } catch (error) {
    const reason = `replay failed: ${errorMessage(error)}`;
    if (cached !== undefined && cached.inputHash === inputHash) {
      // 同输入曾成功而本次重算失败（罕见，如同步代码变更）：直接复用旧
      // 成功结果，其 replay.asOf 如实为旧时点。
      return { state: "ready", model: "glicko2", inputHash, replay: cached.replay };
    }
    if (cached !== undefined) {
      return { state: "stale", model: "glicko2", reason, lastGood: cached.replay };
    }
    return { state: "unavailable", model: "glicko2", reason };
  }
}

/** 在同一事务中读取配置、球员目录与原始比赛，保证一致快照。 */
function readConsistentSnapshot(conn: Database.Database): ConsistentSnapshot {
  const read = conn.transaction(() => ({
    record: readRatingConfig(conn),
    players: listPlayers(conn),
    matches: listRawMatches(conn),
  }));
  return read();
}

/**
 * 缓存可复用当且仅当：输入指纹一致，且缓存快照的 asOf 上海日期与
 * 区段和当前请求相同（同一边界内）。配置版本不符的键根本不会被读到。
 */
function isCacheUsable(
  cached: LastGoodSnapshot,
  inputHash: string,
  shanghaiDate: string,
  segmentId: string
): boolean {
  if (cached.inputHash !== inputHash) return false;
  let cachedDate: string;
  try {
    cachedDate = shanghaiLocalDateFromInstant(cached.asOf);
  } catch {
    return false;
  }
  return cachedDate === shanghaiDate && cached.replay.currentSegment.id === segmentId;
}

/** 读取最近成功快照；缺失或序列化损坏时返回 undefined（视为无缓存）。 */
function readLastGoodSnapshot(
  conn: Database.Database,
  configVersion: string
): LastGoodSnapshot | undefined {
  const row = conn
    .prepare(`SELECT value FROM meta WHERE key = ?`)
    .get(ratingLastGoodKey(configVersion)) as { value: string } | undefined;
  if (row === undefined) return undefined;
  try {
    const parsed = JSON.parse(row.value) as Partial<LastGoodSnapshot>;
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      typeof parsed.inputHash !== "string" ||
      typeof parsed.asOf !== "string" ||
      typeof parsed.replay !== "object" ||
      parsed.replay === null ||
      typeof (parsed.replay as Partial<RatingReplay>).currentSegment !== "object" ||
      typeof (parsed.replay as Partial<RatingReplay>).currentSegment?.id !== "string"
    ) {
      return undefined;
    }
    return parsed as LastGoodSnapshot;
  } catch {
    return undefined;
  }
}

/** 单次原子写入最近成功快照；计算/序列化异常时保留旧值不覆写。 */
function writeLastGoodSnapshot(
  conn: Database.Database,
  configVersion: string,
  snapshot: LastGoodSnapshot
): void {
  try {
    conn
      .prepare(`INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)`)
      .run(ratingLastGoodKey(configVersion), JSON.stringify(snapshot));
  } catch {
    // 快照只是可删除重建的缓存：写入失败保留旧值，不影响本次 ready 结果。
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
