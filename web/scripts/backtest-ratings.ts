import { createHash } from "node:crypto";
import { lstatSync, realpathSync, statSync, writeFileSync, type Stats } from "node:fs";
import { basename, dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import Database from "better-sqlite3";
import { shanghaiLocalDateFromInstant } from "../src/lib/ratings/calendar";
import { createRatingConfig } from "../src/lib/ratings/config";
import { evaluateRatings, type EvaluationInput, type EvaluationReport } from "../src/lib/ratings/evaluation";
import type { RatingMatch } from "../src/lib/ratings/types";
import { syntheticRatingScenario } from "../test/fixtures/ratings-scenarios";

export interface CliOptions {
  readonly help: boolean;
  readonly fixture?: "synthetic";
  readonly db?: string;
  readonly asOf?: string;
  readonly firstSeasonStart?: string;
  readonly out?: string;
}

export interface CliRunResult {
  readonly exitCode: 0 | 1;
  readonly report?: EvaluationReport;
  readonly error?: string;
}

export interface CliIo {
  readonly stdout: (message: string) => void;
  readonly stderr: (message: string) => void;
}

const HELP = `Usage: tsx scripts/backtest-ratings.ts (--fixture synthetic | --db <path>) --as-of <ISO with timezone> --first-season-start <YYYY-MM-DD quarter start> --out <path>\n`;

/** Parses a deliberately small, explicit flag surface for repeatable backtests. */
export function parseArgs(args: readonly string[]): CliOptions {
  const values = new Map<string, string | true>();
  const flags = new Set(["--help", "--fixture", "--db", "--as-of", "--first-season-start", "--out"]);
  for (let index = 0; index < args.length; index += 1) {
    const flag = args[index];
    if (!flags.has(flag)) throw new RangeError(`unknown flag: ${flag}`);
    if (values.has(flag)) throw new RangeError(`duplicate flag: ${flag}`);
    if (flag === "--help") {
      values.set(flag, true);
      continue;
    }
    const value = args[index + 1];
    if (value === undefined || value.startsWith("--")) throw new RangeError(`missing value for ${flag}`);
    values.set(flag, value);
    index += 1;
  }
  if (values.has("--help")) {
    if (values.size !== 1) throw new RangeError("--help cannot be combined with other flags");
    return { help: true };
  }

  const fixture = values.get("--fixture");
  const db = values.get("--db");
  if ((fixture === undefined && db === undefined) || (fixture !== undefined && db !== undefined)) {
    throw new RangeError("provide exactly one source: --fixture synthetic or --db <path>");
  }
  if (fixture !== undefined && fixture !== "synthetic") throw new RangeError("--fixture must be synthetic");
  const asOf = requiredString(values, "--as-of");
  const firstSeasonStart = requiredString(values, "--first-season-start");
  const out = requiredString(values, "--out");
  shanghaiLocalDateFromInstant(asOf);
  createRatingConfig({ firstSeasonStart });
  return {
    help: false,
    ...(fixture === undefined ? {} : { fixture: "synthetic" }),
    ...(db === undefined ? {} : { db: String(db) }),
    asOf,
    firstSeasonStart,
    out,
  };
}

/** Runs the read-only backtest and returns an exit indication suitable for tests and the executable entry point. */
export function run(args: readonly string[], io: CliIo = defaultIo()): CliRunResult {
  try {
    const options = parseArgs(args);
    if (options.help) {
      io.stdout(HELP);
      return { exitCode: 0 };
    }
    if (options.db !== undefined) assertOutputIsNotDatabaseFile(options.db, options.out!);
    const input = options.fixture === "synthetic" ? syntheticInput() : databaseInput(options.db!);
    const report = evaluateRatings(input, {
      config: createRatingConfig({ firstSeasonStart: options.firstSeasonStart! }),
      asOf: options.asOf!,
    });
    writeFileSync(options.out!, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    io.stdout(`wrote rating backtest report to ${options.out}\n`);
    return { exitCode: 0, report };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr(`backtest-ratings: ${message}\n`);
    return { exitCode: 1, error: message };
  }
}

function syntheticInput(): EvaluationInput {
  return withInputHash({
    source: { kind: "synthetic", inputVersion: "synthetic-v1", inputHash: syntheticRatingScenario.source.inputHash },
    playerIds: [...syntheticRatingScenario.playerIds],
    matches: syntheticRatingScenario.matches.map((match) => ({ ...match })),
  });
}

function databaseInput(file: string): EvaluationInput {
  const stat = statSync(file);
  if (!stat.isFile()) throw new RangeError("--db must name an existing regular file");
  let db: Database.Database | undefined;
  try {
    db = new Database(file, { readonly: true, fileMustExist: true });
    const { playerRows, matchRows } = db.transaction(() => ({
      playerRows: db!.prepare("SELECT id FROM players ORDER BY id").all(),
      matchRows: db!
        .prepare(
          `SELECT id, pa1, pa2, pb1, pb2, score_a AS scoreA, score_b AS scoreB,
                  played_at AS playedAt, created_at AS createdAt
           FROM matches
           ORDER BY played_at, created_at, id`
        )
        .all(),
    }))();
    const playerIds = playerRows.map((row, index) => numericColumn(row, "id", `players row ${index + 1}`));
    const matches = matchRows.map((row, index) => readMatch(row, index + 1));
    return withInputHash({
      source: { kind: "database", inputVersion: "sqlite-rating-input-v1", inputHash: "pending" },
      playerIds,
      matches,
    });
  } finally {
    db?.close();
  }
}

function assertOutputIsNotDatabaseFile(databaseFile: string, outputFile: string): void {
  const outputLink = existingLstat(outputFile);
  if (outputLink?.isSymbolicLink()) {
    throw new RangeError("--out must not be a symbolic link in database mode");
  }
  const canonicalDatabase = realpathSync(databaseFile);
  const databaseStat = statSync(canonicalDatabase);
  const explicitSidecars = new Set([
    `${canonicalDatabase}-wal`,
    `${canonicalDatabase}-shm`,
    `${canonicalDatabase}-journal`,
  ]);
  const canonicalOutput = canonicalizeOutputPath(outputFile);
  if (explicitSidecars.has(canonicalOutput)) {
    throw new RangeError("--out must not target a database sidecar file");
  }
  const sidecarStats = [...explicitSidecars].map(existingStat);
  try {
    const outputStat = statSync(canonicalOutput);
    if (canonicalOutput === canonicalDatabase || sameFile(outputStat, databaseStat)) {
      throw new RangeError("--out must not refer to the read-only database input");
    }
    if (sidecarStats.some((sidecarStat) => sidecarStat !== undefined && sameFile(outputStat, sidecarStat))) {
      throw new RangeError("--out must not refer to a database sidecar file");
    }
  } catch (error) {
    if (error instanceof RangeError) throw error;
    const systemError = error as NodeJS.ErrnoException;
    if (systemError.code !== "ENOENT") throw error;
  }
}

function existingLstat(file: string): Stats | undefined {
  try {
    return lstatSync(file);
  } catch (error) {
    const systemError = error as NodeJS.ErrnoException;
    if (systemError.code === "ENOENT") return undefined;
    throw error;
  }
}

function existingStat(file: string): Stats | undefined {
  try {
    return statSync(file);
  } catch (error) {
    const systemError = error as NodeJS.ErrnoException;
    if (systemError.code === "ENOENT") return undefined;
    throw error;
  }
}

function sameFile(first: Stats, second: Stats): boolean {
  return first.dev === second.dev && first.ino === second.ino;
}

function canonicalizeOutputPath(file: string): string {
  try {
    return realpathSync(file);
  } catch (error) {
    const systemError = error as NodeJS.ErrnoException;
    if (systemError.code !== "ENOENT") throw error;
    return join(realpathSync(dirname(file)), basename(file));
  }
}

function withInputHash(input: EvaluationInput): EvaluationInput {
  const inputHash = createHash("sha256")
    .update(JSON.stringify({ playerIds: input.playerIds, matches: input.matches }))
    .digest("hex");
  return { ...input, source: { ...input.source, inputHash } };
}

function readMatch(row: unknown, rowNumber: number): RatingMatch {
  return {
    id: numericColumn(row, "id", `matches row ${rowNumber}`),
    teamA: [
      numericColumn(row, "pa1", `matches row ${rowNumber}`),
      numericColumn(row, "pa2", `matches row ${rowNumber}`),
    ],
    teamB: [
      numericColumn(row, "pb1", `matches row ${rowNumber}`),
      numericColumn(row, "pb2", `matches row ${rowNumber}`),
    ],
    scoreA: numericColumn(row, "scoreA", `matches row ${rowNumber}`),
    scoreB: numericColumn(row, "scoreB", `matches row ${rowNumber}`),
    playedAt: stringColumn(row, "playedAt", `matches row ${rowNumber}`),
    createdAt: stringColumn(row, "createdAt", `matches row ${rowNumber}`),
  };
}

function numericColumn(row: unknown, key: string, context: string): number {
  const value = recordColumn(row, key, context);
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new RangeError(`${context}.${key} must be a safe integer`);
  }
  return value;
}

function stringColumn(row: unknown, key: string, context: string): string {
  const value = recordColumn(row, key, context);
  if (typeof value !== "string") throw new RangeError(`${context}.${key} must be a string`);
  return value;
}

function recordColumn(row: unknown, key: string, context: string): unknown {
  if (typeof row !== "object" || row === null || !(key in row)) {
    throw new RangeError(`${context} is missing ${key}`);
  }
  return (row as Record<string, unknown>)[key];
}

function requiredString(values: ReadonlyMap<string, string | true>, flag: string): string {
  const value = values.get(flag);
  if (typeof value !== "string" || value === "") throw new RangeError(`missing value for ${flag}`);
  return value;
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
