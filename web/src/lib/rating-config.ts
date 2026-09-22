import type Database from "better-sqlite3";
import { getDb } from "./db";
import {
  nextQuarterStart,
  shanghaiLocalDateFromInstant,
} from "./ratings/calendar";
import {
  createRatingConfig,
  validateRatingConfig,
} from "./ratings/config";
import type { LocalDate, RatingConfig, RatingModel } from "./ratings/types";

/**
 * 持久化评分配置的 meta 键。单键保存「同一时刻唯一生效」的配置：
 * 换参数版本即整体替换 config 内容，但不重置 activeModel 与
 * firstSeasonStart 的历史含义（读取/重启/月份变化都不得移动首赛季起点）。
 */
export const RATING_CONFIG_META_KEY = "ratings.config.v1";

/** meta 中保存的配置记录：共享 RatingConfig + 当前默认模型 + 初始化时点。 */
export interface RatingConfigRecord {
  config: RatingConfig;
  activeModel: RatingModel;
  /** 首次初始化时的 ISO instant，仅作信息展示，不参与任何重算。 */
  initializedAt: string;
}

/** 可通过命令行覆盖的数值参数（algorithmVersion/timeZone 固定不变）。 */
export interface RatingConfigParams {
  paramsVersion?: string;
  initialRating?: number;
  initialRd?: number;
  minRd?: number;
  maxRd?: number;
  initialVolatility?: number;
  tau?: number;
  seasonLower?: number;
  seasonUpper?: number;
  seasonRetention?: number;
  seasonRdFloor?: number;
}

export interface InitializeRatingConfigOptions extends RatingConfigParams {
  /** 显式首赛季季度起点；缺省时取初始化当天之后的第一个季度起点。 */
  firstSeasonStart?: LocalDate;
  /** 可注入的「今天」（ISO instant），便于固定日期测试；默认取当前时间。 */
  todayInstant?: string;
}

export interface InitializeRatingConfigResult {
  /** true 表示写入了新配置；false 表示已存在完全一致配置，未覆写。 */
  created: boolean;
  /** true 表示用新的 paramsVersion 整体替换了旧配置内容。 */
  replaced: boolean;
  record: RatingConfigRecord;
  previous: RatingConfigRecord | null;
}

function resolveDb(db?: Database.Database): Database.Database {
  return db ?? getDb();
}

/**
 * 初始化当天（上海日期）之后的首个自然季度起点。
 * 由初始化命令一次性固化，之后的读取、重启、月份变化都不得重新推算。
 */
export function defaultFirstSeasonStart(todayInstant: string): LocalDate {
  return nextQuarterStart(shanghaiLocalDateFromInstant(todayInstant));
}

/**
 * 读取持久化配置；未初始化返回 null，绝不隐式写入默认值。
 * 已存在但内容损坏（非 JSON / 形状不对 / 过不了引擎校验）时抛错，
 * 由调用方决定按 unavailable 处理，避免把损坏数据当成「未初始化」。
 */
export function readRatingConfig(db?: Database.Database): RatingConfigRecord | null {
  const conn = resolveDb(db);
  const row = conn
    .prepare(`SELECT value FROM meta WHERE key = ?`)
    .get(RATING_CONFIG_META_KEY) as { value: string } | undefined;
  if (row === undefined) return null;
  return parseRatingConfigRecord(row.value);
}

function parseRatingConfigRecord(raw: string): RatingConfigRecord {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`stored rating config at ${RATING_CONFIG_META_KEY} is not valid JSON`);
  }
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error(`stored rating config at ${RATING_CONFIG_META_KEY} has an invalid shape`);
  }
  const record = parsed as Partial<RatingConfigRecord>;
  if (typeof record.config !== "object" || record.config === null) {
    throw new Error(`stored rating config at ${RATING_CONFIG_META_KEY} is missing config`);
  }
  // 持久化配置必须仍能通过引擎校验，否则说明存储被污染。
  const config = validateRatingConfig(record.config as RatingConfig);
  if (record.activeModel !== "glicko2" && record.activeModel !== "legacy") {
    throw new Error(`stored rating config at ${RATING_CONFIG_META_KEY} has an invalid activeModel`);
  }
  if (typeof record.initializedAt !== "string" || record.initializedAt === "") {
    throw new Error(`stored rating config at ${RATING_CONFIG_META_KEY} is missing initializedAt`);
  }
  return {
    config,
    activeModel: record.activeModel,
    initializedAt: record.initializedAt,
  };
}

/**
 * 初始化（或升级参数版本）持久化配置，唯一允许写入配置的路径。
 * - 未初始化：默认初始化 activeModel=legacy，firstSeasonStart 取初始化
 *   当天之后的第一个季度起点（可用显式值覆盖）；
 * - 已存在且 paramsVersion 相同：参数完全一致则不覆写；参数不同则拒绝，
 *   不允许同版本悄悄换常数；
 * - 已存在且 paramsVersion 不同：整体替换配置内容，但保留 activeModel，
 *   未显式指定时保留原 firstSeasonStart（首赛季起点的历史含义不变）。
 */
export function initializeRatingConfig(
  options: InitializeRatingConfigOptions = {},
  db?: Database.Database
): InitializeRatingConfigResult {
  const conn = resolveDb(db);
  const existing = readRatingConfig(conn);
  const todayInstant = options.todayInstant ?? new Date().toISOString();

  if (existing !== null) {
    const requested = buildRequestedConfig(existing.config, options);
    if (requested.paramsVersion === existing.config.paramsVersion) {
      if (configEquals(requested, existing.config)) {
        return { created: false, replaced: false, record: existing, previous: existing };
      }
      throw new Error(
        `paramsVersion "${requested.paramsVersion}" is already initialized with different parameters; ` +
          "bump paramsVersion to change rating constants"
      );
    }
    const record: RatingConfigRecord = {
      config: requested,
      activeModel: existing.activeModel,
      initializedAt: existing.initializedAt,
    };
    writeRatingConfigRecord(conn, record);
    return { created: true, replaced: true, record, previous: existing };
  }

  const config = createRatingConfig({
    firstSeasonStart:
      options.firstSeasonStart ?? defaultFirstSeasonStart(todayInstant),
    ...paramsOverrides(options),
  });
  const record: RatingConfigRecord = {
    config,
    activeModel: "legacy",
    initializedAt: todayInstant,
  };
  writeRatingConfigRecord(conn, record);
  return { created: true, replaced: false, record, previous: null };
}

/**
 * 与 initializeRatingConfig 完全相同的校验与推演，但不写入数据库，
 * 供 CLI --dry-run 输出将要发生的差异。
 */
export function planInitializeRatingConfig(
  options: InitializeRatingConfigOptions = {},
  db?: Database.Database
): InitializeRatingConfigResult {
  const conn = resolveDb(db);
  const existing = readRatingConfig(conn);
  const todayInstant = options.todayInstant ?? new Date().toISOString();
  if (existing === null) {
    const config = createRatingConfig({
      firstSeasonStart:
        options.firstSeasonStart ?? defaultFirstSeasonStart(todayInstant),
      ...paramsOverrides(options),
    });
    return {
      created: true,
      replaced: false,
      record: { config, activeModel: "legacy", initializedAt: todayInstant },
      previous: null,
    };
  }
  const requested = buildRequestedConfig(existing.config, options);
  if (requested.paramsVersion === existing.config.paramsVersion) {
    if (configEquals(requested, existing.config)) {
      return { created: false, replaced: false, record: existing, previous: existing };
    }
    throw new Error(
      `paramsVersion "${requested.paramsVersion}" is already initialized with different parameters; ` +
        "bump paramsVersion to change rating constants"
    );
  }
  return {
    created: true,
    replaced: true,
    record: {
      config: requested,
      activeModel: existing.activeModel,
      initializedAt: existing.initializedAt,
    },
    previous: existing,
  };
}

/**
 * 只切换 activeModel：不清空比赛、不重写历史、不动 last-good 缓存键
 * 以外的 meta。未初始化时抛错；目标模型与当前一致时为幂等 no-op。
 */
export function setActiveModel(model: RatingModel, db?: Database.Database): RatingConfigRecord {
  if (model !== "glicko2" && model !== "legacy") {
    throw new RangeError("model must be glicko2 or legacy");
  }
  const conn = resolveDb(db);
  const existing = readRatingConfig(conn);
  if (existing === null) {
    throw new Error("rating config not initialized; run `rating-config init` first");
  }
  if (existing.activeModel === model) return existing;
  const record: RatingConfigRecord = { ...existing, activeModel: model };
  writeRatingConfigRecord(conn, record);
  return record;
}

function buildRequestedConfig(
  existing: RatingConfig,
  options: InitializeRatingConfigOptions
): RatingConfig {
  return createRatingConfig({
    // 换参数版本时不显式指定首赛季起点，则保留原值（不重置历史含义）。
    firstSeasonStart: options.firstSeasonStart ?? existing.firstSeasonStart,
    ...paramsOverrides(options),
    paramsVersion: options.paramsVersion ?? existing.paramsVersion,
  });
}

function paramsOverrides(options: RatingConfigParams): RatingConfigParams {
  const overrides: RatingConfigParams = {};
  for (const key of NUMERIC_PARAM_KEYS) {
    const value = options[key];
    if (value !== undefined) {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new RangeError(`${key} must be a finite number`);
      }
      overrides[key] = value;
    }
  }
  if (options.paramsVersion !== undefined) overrides.paramsVersion = options.paramsVersion;
  return overrides;
}

const NUMERIC_PARAM_KEYS = [
  "initialRating",
  "initialRd",
  "minRd",
  "maxRd",
  "initialVolatility",
  "tau",
  "seasonLower",
  "seasonUpper",
  "seasonRetention",
  "seasonRdFloor",
] as const;

type NumericParamKey = (typeof NUMERIC_PARAM_KEYS)[number];

function configEquals(first: RatingConfig, second: RatingConfig): boolean {
  return (
    first.paramsVersion === second.paramsVersion &&
    first.firstSeasonStart === second.firstSeasonStart &&
    NUMERIC_PARAM_KEYS.every((key: NumericParamKey) => first[key] === second[key])
  );
}

function writeRatingConfigRecord(conn: Database.Database, record: RatingConfigRecord): void {
  // 单条 INSERT OR REPLACE 即原子写入；序列化异常向上抛，调用方保留旧值。
  conn
    .prepare(`INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)`)
    .run(RATING_CONFIG_META_KEY, JSON.stringify(record));
}

/** 供 CLI --dry-run 输出具体配置差异；无差异时返回空数组。 */
export function describeRatingConfigChanges(
  previous: RatingConfigRecord | null,
  next: RatingConfigRecord
): string[] {
  const lines: string[] = [];
  const fields: ReadonlyArray<{ label: string; value: (record: RatingConfigRecord) => string }> = [
    { label: "activeModel", value: (record) => record.activeModel },
    { label: "paramsVersion", value: (record) => record.config.paramsVersion },
    { label: "firstSeasonStart", value: (record) => record.config.firstSeasonStart },
    ...NUMERIC_PARAM_KEYS.map((key) => ({
      label: key,
      value: (record: RatingConfigRecord) => String(record.config[key]),
    })),
  ];
  for (const field of fields) {
    const nextValue = field.value(next);
    if (previous === null) {
      lines.push(`${field.label} = ${nextValue}`);
    } else {
      const previousValue = field.value(previous);
      if (previousValue !== nextValue) {
        lines.push(`${field.label}: ${previousValue} -> ${nextValue}`);
      }
    }
  }
  return lines;
}
