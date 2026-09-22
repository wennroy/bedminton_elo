import { pathToFileURL } from "node:url";
import { createDb } from "../src/lib/db";
import {
  describeRatingConfigChanges,
  planInitializeRatingConfig,
  initializeRatingConfig,
  readRatingConfig,
  setActiveModel,
  type RatingConfigParams,
  type RatingConfigRecord,
} from "../src/lib/rating-config";
import type { RatingModel } from "../src/lib/ratings/types";

export interface CliOptions {
  readonly command: "init" | "status" | "activate";
  readonly help: boolean;
  readonly db?: string;
  readonly model?: RatingModel;
  readonly dryRun: boolean;
  readonly params: RatingConfigParams;
  readonly firstSeasonStart?: string;
}

export interface CliRunResult {
  readonly exitCode: 0 | 1;
  readonly record?: RatingConfigRecord;
  readonly error?: string;
}

export interface CliIo {
  readonly stdout: (message: string) => void;
  readonly stderr: (message: string) => void;
}

const HELP = `Usage: tsx scripts/rating-config.ts <command> [flags]

Commands:
  init      初始化持久化评分配置（默认 activeModel=legacy，不覆写已存在配置）
  status    查看当前持久化配置（未初始化时提示维持 Legacy）
  activate  切换 activeModel（--model glicko2 | legacy）

Flags:
  --db <path>                  数据库文件（默认 DATABASE_URL 或 ./badminton.db）
  --model <glicko2|legacy>     activate 的目标模型
  --first-season-start <date>  显式首赛季季度起点 YYYY-MM-DD（须为季度首日）
  --params-version <version>   参数版本标识；参数变化必须更换版本号
  --initial-rating <n>         覆盖数值参数（init 时生效）
  --initial-rd <n>
  --min-rd <n>
  --max-rd <n>
  --initial-volatility <n>
  --tau <n>
  --season-lower <n>
  --season-upper <n>
  --season-retention <n>
  --season-rd-floor <n>
  --dry-run                    只输出将要发生的配置差异，不写入数据库
  --help
`;

const COMMANDS = new Set(["init", "status", "activate"]);

const NUMERIC_FLAGS: ReadonlyArray<{ flag: string; key: NumericParamKey }> = [
  { flag: "--initial-rating", key: "initialRating" },
  { flag: "--initial-rd", key: "initialRd" },
  { flag: "--min-rd", key: "minRd" },
  { flag: "--max-rd", key: "maxRd" },
  { flag: "--initial-volatility", key: "initialVolatility" },
  { flag: "--tau", key: "tau" },
  { flag: "--season-lower", key: "seasonLower" },
  { flag: "--season-upper", key: "seasonUpper" },
  { flag: "--season-retention", key: "seasonRetention" },
  { flag: "--season-rd-floor", key: "seasonRdFloor" },
];

const VALUE_FLAGS = new Set([
  "--db",
  "--model",
  "--first-season-start",
  "--params-version",
  ...NUMERIC_FLAGS.map((entry) => entry.flag),
]);

type NumericParamKey = Exclude<keyof RatingConfigParams, "paramsVersion">;

/** Parses a deliberately small, explicit flag surface for repeatable config management. */
export function parseArgs(args: readonly string[]): CliOptions {
  if (args[0] === "--help") {
    if (args.length !== 1) throw new RangeError("--help cannot be combined with other flags");
    return { command: "status", help: true, dryRun: false, params: {} };
  }
  const command = args[0];
  if (command === undefined || !COMMANDS.has(command)) {
    throw new RangeError(`first argument must be one of: ${[...COMMANDS].join(", ")}`);
  }
  const values = new Map<string, string | true>();
  let dryRun = false;
  const params: RatingConfigParams = {};
  for (let index = 1; index < args.length; index += 1) {
    const flag = args[index];
    if (flag === "--help") {
      if (values.size !== 0 || dryRun) throw new RangeError("--help cannot be combined with other flags");
      return { command: command as CliOptions["command"], help: true, dryRun: false, params };
    }
    if (flag === "--dry-run") {
      if (dryRun) throw new RangeError("duplicate flag: --dry-run");
      dryRun = true;
      continue;
    }
    if (!VALUE_FLAGS.has(flag)) throw new RangeError(`unknown flag: ${flag}`);
    if (values.has(flag)) throw new RangeError(`duplicate flag: ${flag}`);
    const value = args[index + 1];
    if (value === undefined || value.startsWith("--")) throw new RangeError(`missing value for ${flag}`);
    values.set(flag, value);
    index += 1;
  }

  const numeric = NUMERIC_FLAGS.find((entry) => values.has(entry.flag));
  if (numeric !== undefined) {
    for (const entry of NUMERIC_FLAGS) {
      const raw = stringFlag(values, entry.flag);
      if (raw === undefined) continue;
      const parsed = Number(raw);
      if (raw.trim() === "" || !Number.isFinite(parsed)) {
        throw new RangeError(`${entry.flag} must be a finite number`);
      }
      params[entry.key] = parsed;
    }
  }

  const modelValue = stringFlag(values, "--model");
  const model = modelValue as RatingModel | undefined;
  if (modelValue !== undefined && model !== "glicko2" && model !== "legacy") {
    throw new RangeError("--model must be glicko2 or legacy");
  }
  const firstSeasonStart = stringFlag(values, "--first-season-start");
  const paramsVersion = stringFlag(values, "--params-version");
  if (paramsVersion !== undefined) params.paramsVersion = paramsVersion;

  // 命令面校验：status/activate 不接受配置参数，activate 必须给 --model。
  if (
    command !== "init" &&
    (numeric !== undefined || firstSeasonStart !== undefined || paramsVersion !== undefined)
  ) {
    throw new RangeError(`config flags are only valid with the init command`);
  }
  if (command === "activate" && model === undefined) {
    throw new RangeError("activate requires --model glicko2|legacy");
  }

  return {
    command: command as CliOptions["command"],
    help: false,
    ...(stringFlag(values, "--db") === undefined ? {} : { db: stringFlag(values, "--db")! }),
    ...(model === undefined ? {} : { model }),
    dryRun,
    params,
    ...(firstSeasonStart === undefined ? {} : { firstSeasonStart }),
  };
}

/** Value flags always store strings; --help (true) is handled before this point. */
function stringFlag(values: Map<string, string | true>, flag: string): string | undefined {
  const value = values.get(flag);
  if (value === undefined) return undefined;
  if (value === true) throw new RangeError(`missing value for ${flag}`);
  return value;
}

/** Runs the config command and returns an exit indication suitable for tests and the executable entry point. */
export function run(args: readonly string[], io: CliIo = defaultIo()): CliRunResult {
  let db: ReturnType<typeof createDb> | undefined;
  try {
    const options = parseArgs(args);
    if (options.help) {
      io.stdout(HELP);
      return { exitCode: 0 };
    }
    db = createDb(options.db);
    switch (options.command) {
      case "init":
        return runInit(options, db, io);
      case "status":
        return runStatus(db, io);
      case "activate":
        return runActivate(options, db, io);
      default:
        throw new RangeError(`unknown command: ${(options as CliOptions).command}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr(`rating-config: ${message}\n`);
    return { exitCode: 1, error: message };
  } finally {
    db?.close();
  }
}

function runInit(
  options: CliOptions,
  db: ReturnType<typeof createDb>,
  io: CliIo
): CliRunResult {
  const initOptions = {
    ...(options.firstSeasonStart === undefined
      ? {}
      : { firstSeasonStart: options.firstSeasonStart }),
    ...options.params,
  };
  // 无论是否 dry-run 都先走完整校验（日期、参数冲突等），失败即拒绝且不写入。
  const planned = planInitializeRatingConfig(initOptions, db);
  const changes = describeRatingConfigChanges(planned.previous, planned.record);
  if (options.dryRun) {
    printChanges(planned.previous, changes, io);
    io.stdout("(dry-run: no changes written)\n");
    return { exitCode: 0, record: planned.record };
  }
  const result = initializeRatingConfig(initOptions, db);
  if (!result.created) {
    io.stdout("rating config already initialized with identical parameters; no changes\n");
  } else if (result.replaced) {
    io.stdout(
      `replaced rating config params: ${result.previous!.config.paramsVersion} -> ${result.record.config.paramsVersion} (activeModel unchanged: ${result.record.activeModel})\n`
    );
    printChanges(result.previous, changes, io);
  } else {
    io.stdout(
      `initialized rating config: activeModel=${result.record.activeModel} paramsVersion=${result.record.config.paramsVersion} firstSeasonStart=${result.record.config.firstSeasonStart}\n`
    );
  }
  return { exitCode: 0, record: result.record };
}

function printChanges(
  previous: RatingConfigRecord | null,
  changes: string[],
  io: CliIo
): void {
  if (previous === null) {
    for (const line of changes) io.stdout(`+ ${line}\n`);
    return;
  }
  if (changes.length === 0) {
    io.stdout("no config changes\n");
    return;
  }
  for (const line of changes) io.stdout(`~ ${line}\n`);
}

function runStatus(db: ReturnType<typeof createDb>, io: CliIo): CliRunResult {
  const record = readRatingConfig(db);
  if (record === null) {
    io.stdout("rating config: not initialized (running Legacy)\n");
    return { exitCode: 0 };
  }
  io.stdout(
    [
      "rating config: initialized",
      `  activeModel: ${record.activeModel}`,
      `  paramsVersion: ${record.config.paramsVersion}`,
      `  firstSeasonStart: ${record.config.firstSeasonStart}`,
      `  initializedAt: ${record.initializedAt}`,
    ].join("\n") + "\n"
  );
  return { exitCode: 0, record };
}

function runActivate(
  options: CliOptions,
  db: ReturnType<typeof createDb>,
  io: CliIo
): CliRunResult {
  if (options.dryRun) {
    const current = readRatingConfig(db);
    if (current === null) throw new Error("rating config not initialized; run `rating-config init` first");
    if (current.activeModel === options.model) {
      io.stdout(`activeModel already ${options.model}; no changes\n`);
    } else {
      io.stdout(`~ activeModel: ${current.activeModel} -> ${options.model}\n`);
    }
    io.stdout("(dry-run: no changes written)\n");
    return { exitCode: 0, record: current };
  }
  const record = setActiveModel(options.model!, db);
  io.stdout(`activeModel: ${record.activeModel}\n`);
  return { exitCode: 0, record };
}

function defaultIo(): CliIo {
  return {
    stdout: (message) => process.stdout.write(message),
    stderr: (message) => process.stderr.write(message),
  };
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const result = run(process.argv.slice(2));
  if (result.exitCode !== 0) process.exitCode = result.exitCode;
}
