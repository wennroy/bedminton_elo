import type { LocalDate, RatingConfig } from "./types";

/**
 * The shared doubles coordinates are x = (r - 1000) / (2 * displayScale) and
 * phi = RD / (2 * displayScale); volatility is in x-coordinate weekly units.
 * This intentionally differs from the conventional Glicko-2 display conversion.
 */
export const GLICKO2_DISPLAY_SCALE = 173.7178;
export const GLICKO2_INTERNAL_SCALE = 2 * GLICKO2_DISPLAY_SCALE;

const INTERNAL_RATING_CENTER = 1000;

export function displayRatingToInternalX(rating: number): number {
  return (rating - INTERNAL_RATING_CENTER) / GLICKO2_INTERNAL_SCALE;
}

export function displayRdToInternalPhi(rd: number): number {
  return rd / GLICKO2_INTERNAL_SCALE;
}

export const DEFAULT_RATING_PARAMS = Object.freeze({
  algorithmVersion: "glicko2-doubles-v1" as const,
  paramsVersion: "p1",
  timeZone: "Asia/Shanghai" as const,
  initialRating: INTERNAL_RATING_CENTER,
  initialRd: 180,
  minRd: 60,
  maxRd: 250,
  // This is Glicko-2 internal volatility in weekly periods, not rating points.
  initialVolatility: 0.06,
  tau: 0.3,
  seasonLower: 900,
  seasonUpper: 1100,
  seasonRetention: 0.75,
  seasonRdFloor: 90,
});

export type RatingConfigInput = Pick<RatingConfig, "firstSeasonStart"> &
  Partial<Omit<RatingConfig, "algorithmVersion" | "firstSeasonStart" | "timeZone">>;

export function createRatingConfig(input: RatingConfigInput): RatingConfig {
  return validateRatingConfig({ ...DEFAULT_RATING_PARAMS, ...input });
}

export function validateRatingConfig(config: RatingConfig): RatingConfig {
  if (config.algorithmVersion !== "glicko2-doubles-v1") {
    throw new RangeError('algorithmVersion must be "glicko2-doubles-v1"');
  }
  if (config.timeZone !== "Asia/Shanghai") {
    throw new RangeError('timeZone must be "Asia/Shanghai"');
  }
  assertNonEmptyString(config.paramsVersion, "paramsVersion");
  assertQuarterStart(config.firstSeasonStart);

  const numericValues = {
    initialRating: config.initialRating,
    initialRd: config.initialRd,
    minRd: config.minRd,
    maxRd: config.maxRd,
    initialVolatility: config.initialVolatility,
    tau: config.tau,
    seasonLower: config.seasonLower,
    seasonUpper: config.seasonUpper,
    seasonRetention: config.seasonRetention,
    seasonRdFloor: config.seasonRdFloor,
  };
  for (const [name, value] of Object.entries(numericValues)) {
    assertFiniteNumber(value, name);
  }
  for (const name of [
    "initialRating",
    "initialRd",
    "minRd",
    "maxRd",
    "seasonLower",
    "seasonUpper",
    "seasonRdFloor",
  ] as const) {
    assertNonNegativeNumber(numericValues[name], name);
  }

  if (config.minRd > config.maxRd) {
    throw new RangeError("minRd must be less than or equal to maxRd");
  }
  if (config.initialRd < config.minRd || config.initialRd > config.maxRd) {
    throw new RangeError("initialRd must be within minRd and maxRd");
  }
  if (config.seasonRdFloor < config.minRd || config.seasonRdFloor > config.maxRd) {
    throw new RangeError("seasonRdFloor must be within minRd and maxRd");
  }
  if (config.initialVolatility <= 0) {
    throw new RangeError("initialVolatility must be greater than zero");
  }
  if (config.tau <= 0) {
    throw new RangeError("tau must be greater than zero");
  }
  if (config.seasonLower >= config.seasonUpper) {
    throw new RangeError("seasonLower must be less than seasonUpper");
  }
  if (config.seasonRetention < 0 || config.seasonRetention > 1) {
    throw new RangeError("seasonRetention must be between zero and one");
  }

  return config;
}

export function ratingConfigVersion(config: RatingConfig): string {
  const validConfig = validateRatingConfig(config);
  return [
    validConfig.algorithmVersion,
    encodeURIComponent(validConfig.paramsVersion),
    validConfig.firstSeasonStart,
    encodeURIComponent(validConfig.timeZone),
    validConfig.initialRating,
    validConfig.initialRd,
    validConfig.minRd,
    validConfig.maxRd,
    validConfig.initialVolatility,
    validConfig.tau,
    validConfig.seasonLower,
    validConfig.seasonUpper,
    validConfig.seasonRetention,
    validConfig.seasonRdFloor,
  ].join("|");
}

function assertNonEmptyString(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${name} must be a non-empty string`);
  }
}

function assertQuarterStart(value: LocalDate): void {
  const match = /^(\d{4})-(01|04|07|10)-01$/.exec(value);
  if (match === null || Number(match[1]) === 0) {
    throw new RangeError("firstSeasonStart must be a valid YYYY-MM-DD quarter start");
  }
}

function assertFiniteNumber(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new RangeError(`${name} must be a finite number`);
  }
}

function assertNonNegativeNumber(value: number, name: string): void {
  if (value < 0) {
    throw new RangeError(`${name} must be greater than or equal to zero`);
  }
}
