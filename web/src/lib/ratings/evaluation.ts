import { INITIAL_RATING, K_DOUBLES, predictElo } from "../elo";
import {
  GLICKO2_INTERNAL_SCALE,
  displayRatingToInternalX,
  displayRdToInternalPhi,
  ratingConfigVersion,
  validateRatingConfig,
} from "./config";
import { isFutureShanghaiLocalDate, isValidLocalDate, weekStart } from "./calendar";
import { replayRatings } from "./replay";
import type {
  PlayerId,
  RatingConfig,
  RatingEvent,
  RatingMatch,
  RatingState,
} from "./types";

const LOG_LOSS_EPSILON = 1e-15;
const DEFAULT_CALIBRATION_EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1] as const;

export type EvaluationSourceKind = "synthetic" | "database";

export interface EvaluationSource {
  readonly kind: EvaluationSourceKind;
  readonly inputVersion: string;
  readonly inputHash: string;
}

export interface EvaluationInput {
  readonly source: EvaluationSource;
  readonly playerIds: readonly PlayerId[];
  readonly matches: readonly RatingMatch[];
}

export interface EvaluationEvidenceThresholds {
  readonly minimumMatches?: number;
  readonly minimumWeeks?: number;
}

export interface EvaluationOptions {
  readonly config: RatingConfig;
  readonly asOf: string;
  readonly evidence?: EvaluationEvidenceThresholds;
}

export interface Prediction {
  readonly matchId: number;
  readonly probability: number;
  readonly result: 0 | 1;
}

export interface PredictionSample extends Prediction {
  readonly weekStart: string;
}

export interface PredictionScore {
  readonly count: number;
  readonly brier: number | null;
  readonly logLoss: number | null;
}

export interface CalibrationBin {
  readonly lower: number;
  readonly upper: number;
  readonly count: number;
  readonly meanPrediction: number | null;
  readonly observedRate: number | null;
}

export interface CalibrationSummary {
  readonly count: number;
  readonly bins: readonly CalibrationBin[];
}

export interface WeeklyCalibration {
  readonly weekStart: string;
  readonly score: PredictionScore;
  readonly calibration: CalibrationSummary;
}

export interface EvaluatedModel {
  readonly id: "legacy-elo-k16-v1" | "naive-glicko2-no-partner-v1" | "glicko2-doubles-v1";
  readonly comparisonNote?: string;
  readonly samples: readonly PredictionSample[];
  readonly score: PredictionScore;
  readonly calibration: CalibrationSummary;
  readonly weeklyCalibration: readonly WeeklyCalibration[];
}

export interface WeeklyCorrectionSummary {
  readonly segmentCount: number;
  readonly playerCorrectionCount: number;
  readonly meanSignedCorrection: number | null;
  readonly meanAbsoluteCorrection: number | null;
  readonly maxAbsoluteCorrection: number | null;
}

export interface EvaluationFailure {
  readonly model: EvaluatedModel["id"];
  readonly message: string;
  readonly matchId?: number;
  readonly segmentId?: string;
}

export type EvaluationExclusionReason =
  | "invalid_date"
  | "future"
  | "invalid_score"
  | "duplicate_player"
  | "unknown_player"
  | "invalid_match_id";

export interface EvaluationExclusion {
  readonly matchId: number | null;
  readonly reason: EvaluationExclusionReason;
}

export interface FinalSegmentCorrectionSummary {
  readonly segmentId: string;
  readonly weekStart: string;
  readonly h: number;
  readonly playerCorrectionCount: number;
  readonly meanSignedCorrection: number | null;
  readonly meanAbsoluteCorrection: number | null;
  readonly p50AbsoluteCorrection: number | null;
  readonly p90AbsoluteCorrection: number | null;
  readonly maxAbsoluteCorrection: number | null;
}

export interface EvaluationReport {
  readonly reportVersion: "rating-backtest-v1";
  readonly status: "evaluated" | "insufficient_evidence" | "failed";
  readonly source: EvaluationSource;
  readonly asOf: string;
  readonly configVersion: string;
  readonly input: {
    readonly playerCount: number;
    readonly rawMatchCount: number;
    readonly effectiveMatchCount: number;
    readonly issueCount: number;
    readonly futureMatchCount: number;
    readonly sampleWeekCount: number;
    readonly exclusions: readonly EvaluationExclusion[];
  };
  readonly evidence: {
    readonly minimumMatches: number;
    readonly minimumWeeks: number;
    readonly observedMatches: number;
    readonly observedWeeks: number;
  };
  readonly models: readonly EvaluatedModel[];
  readonly weeklyFinal: {
    readonly corrections: WeeklyCorrectionSummary;
    readonly segments: readonly FinalSegmentCorrectionSummary[];
  };
  readonly failures: readonly EvaluationFailure[];
}

interface EffectiveMatch {
  readonly match: RatingMatch;
  readonly inputIndex: number;
}

interface ClassifiedMatches {
  readonly effective: readonly EffectiveMatch[];
  readonly issueCount: number;
  readonly futureMatchCount: number;
  readonly exclusions: readonly EvaluationExclusion[];
}

/** Scores bounded pre-match probabilities; no clipping is applied to Brier loss. */
export function scorePredictions(predictions: readonly Prediction[]): PredictionScore {
  if (predictions.length === 0) return { count: 0, brier: null, logLoss: null };

  let brierTotal = 0;
  let logLossTotal = 0;
  for (const prediction of predictions) {
    assertPrediction(prediction);
    brierTotal += (prediction.probability - prediction.result) ** 2;
    const probabilityOfOutcome = prediction.result === 1 ? prediction.probability : 1 - prediction.probability;
    logLossTotal += -Math.log(Math.max(LOG_LOSS_EPSILON, probabilityOfOutcome));
  }
  return {
    count: predictions.length,
    brier: brierTotal / predictions.length,
    logLoss: logLossTotal / predictions.length,
  };
}

/** Places probabilities into explicit, inclusive-last calibration bins. */
export function calibratePredictions(
  predictions: readonly Prediction[],
  edges: readonly number[] = DEFAULT_CALIBRATION_EDGES
): CalibrationSummary {
  assertCalibrationEdges(edges);
  const totals = edges.slice(0, -1).map(() => ({ count: 0, probability: 0, result: 0 }));
  for (const prediction of predictions) {
    assertPrediction(prediction);
    const index = calibrationBinIndex(prediction.probability, edges);
    const total = totals[index];
    total.count += 1;
    total.probability += prediction.probability;
    total.result += prediction.result;
  }
  return {
    count: predictions.length,
    bins: totals.map((total, index) => ({
      lower: edges[index],
      upper: edges[index + 1],
      count: total.count,
      meanPrediction: total.count === 0 ? null : total.probability / total.count,
      observedRate: total.count === 0 ? null : total.result / total.count,
    })),
  };
}

/** Summarizes only the correction emitted by T5's weekly Final event. */
export function summarizeWeeklyCorrections(events: readonly RatingEvent[]): WeeklyCorrectionSummary {
  const finalEvents = weeklyFinalEvents(events);
  const corrections = finalEvents.flatMap((event) => Object.values(event.correction));
  if (corrections.length === 0) {
    return {
      segmentCount: 0,
      playerCorrectionCount: 0,
      meanSignedCorrection: null,
      meanAbsoluteCorrection: null,
      maxAbsoluteCorrection: null,
    };
  }
  const absolute = corrections.map(Math.abs);
  return {
    segmentCount: finalEvents.length,
    playerCorrectionCount: corrections.length,
    meanSignedCorrection: sum(corrections) / corrections.length,
    meanAbsoluteCorrection: sum(absolute) / absolute.length,
    maxAbsoluteCorrection: Math.max(...absolute),
  };
}

/** Returns correction distributions for each already-closed weekly Final segment. */
export function summarizeFinalSegmentCorrections(
  events: readonly RatingEvent[]
): readonly FinalSegmentCorrectionSummary[] {
  return weeklyFinalEvents(events).map((event) => {
    const corrections = Object.values(event.correction);
    const absolute = corrections.map(Math.abs).sort((first, second) => first - second);
    return {
      segmentId: event.segment.id,
      weekStart: event.segment.weekStart,
      h: event.segment.h,
      playerCorrectionCount: corrections.length,
      meanSignedCorrection: mean(corrections),
      meanAbsoluteCorrection: mean(absolute),
      p50AbsoluteCorrection: percentile(absolute, 0.5),
      p90AbsoluteCorrection: percentile(absolute, 0.9),
      maxAbsoluteCorrection: absolute.length === 0 ? null : absolute.at(-1)!,
    };
  });
}

/**
 * Runs three prequential, deterministic rating comparisons from raw numeric input.
 * Each model records a probability before it is allowed to observe that match result.
 */
export function evaluateRatings(input: EvaluationInput, options: EvaluationOptions): EvaluationReport {
  const config = validateRatingConfig(options.config);
  validateSource(input.source);
  const thresholds = normalizeThresholds(options.evidence);
  const classified = classifyMatches(input.matches, input.playerIds, options.asOf);
  const effective = [...classified.effective].sort(compareEffectiveMatches);
  const failures: EvaluationFailure[] = [];

  const legacy = evaluateSequentialModel(
    "legacy-elo-k16-v1",
    effective,
    createLegacyPredictor(),
    failures
  );
  const naive = evaluateSequentialModel(
    "naive-glicko2-no-partner-v1",
    effective,
    createNaivePredictor(config),
    failures,
    "Comparison baseline only: this model ignores a teammate in every virtual opponent and omits process variance and weekly Final settlement; it is a frozen-sigma sequential baseline."
  );
  const doubles = evaluateDoublesModel(input.playerIds, effective, options, failures);

  const observedWeeks = new Set(effective.map(({ match }) => weekStart(match.playedAt))).size;
  const insufficient = effective.length < thresholds.minimumMatches || observedWeeks < thresholds.minimumWeeks;
  const doublesReplay = doubles.replay;
  const models = [legacy.model, naive.model, doubles.model];
  return {
    reportVersion: "rating-backtest-v1",
    status: failures.length > 0 ? "failed" : insufficient ? "insufficient_evidence" : "evaluated",
    source: { ...input.source },
    asOf: options.asOf,
    configVersion: ratingConfigVersion(config),
    input: {
      playerCount: new Set(input.playerIds).size,
      rawMatchCount: input.matches.length,
      effectiveMatchCount: effective.length,
      issueCount: classified.issueCount,
      futureMatchCount: classified.futureMatchCount,
      sampleWeekCount: observedWeeks,
      exclusions: classified.exclusions,
    },
    evidence: {
      minimumMatches: thresholds.minimumMatches,
      minimumWeeks: thresholds.minimumWeeks,
      observedMatches: effective.length,
      observedWeeks,
    },
    models,
    weeklyFinal: {
      corrections: doublesReplay === undefined ? emptyWeeklyCorrections() : summarizeWeeklyCorrections(doublesReplay.events),
      segments: doublesReplay === undefined ? [] : summarizeFinalSegmentCorrections(doublesReplay.events),
    },
    failures,
  };
}

function evaluateSequentialModel(
  id: "legacy-elo-k16-v1" | "naive-glicko2-no-partner-v1",
  effective: readonly EffectiveMatch[],
  predictor: SequentialPredictor,
  failures: EvaluationFailure[],
  comparisonNote?: string
): { model: EvaluatedModel } {
  const samples: PredictionSample[] = [];
  for (const { match } of effective) {
    try {
      const probability = predictor.predict(match);
      samples.push(toPredictionSample(match, probability));
      predictor.observe(match);
    } catch (error) {
      failures.push({ model: id, matchId: match.id, segmentId: weekStart(match.playedAt), message: messageOf(error) });
      break;
    }
  }
  return { model: modelReport(id, samples, comparisonNote) };
}

function evaluateDoublesModel(
  playerIds: readonly PlayerId[],
  effective: readonly EffectiveMatch[],
  options: EvaluationOptions,
  failures: EvaluationFailure[]
): { model: EvaluatedModel; replay: ReturnType<typeof replayRatings> | undefined } {
  const matches = effective.map(({ match }) => match);
  try {
    const replay = replayRatings(matches, playerIds, options);
    const samples = effective.map(({ match }) => {
      const estimate = replay.matchEstimates[String(match.id)];
      if (estimate === undefined) {
        throw new Error(`T6 did not emit a pre-match Estimated probability for match ${match.id}`);
      }
      return toPredictionSample(match, estimate.preWinA);
    });
    return { model: modelReport("glicko2-doubles-v1", samples), replay };
  } catch (error) {
    const failed = locateDoublesFailure(playerIds, effective, options);
    failures.push({
      model: "glicko2-doubles-v1",
      matchId: failed?.match.id,
      segmentId: failed === undefined ? undefined : weekStart(failed.match.playedAt),
      message: messageOf(error),
    });
    return { model: modelReport("glicko2-doubles-v1", []), replay: undefined };
  }
}

function locateDoublesFailure(
  playerIds: readonly PlayerId[],
  effective: readonly EffectiveMatch[],
  options: EvaluationOptions
): EffectiveMatch | undefined {
  for (let index = 0; index < effective.length; index += 1) {
    const prefix = effective.slice(0, index + 1).map(({ match }) => match);
    try {
      replayRatings(prefix, playerIds, options);
    } catch {
      return effective[index];
    }
  }
  return undefined;
}

function modelReport(
  id: EvaluatedModel["id"],
  samples: readonly PredictionSample[],
  comparisonNote?: string
): EvaluatedModel {
  return {
    id,
    ...(comparisonNote === undefined ? {} : { comparisonNote }),
    samples,
    score: scorePredictions(samples),
    calibration: calibratePredictions(samples),
    weeklyCalibration: weeklyCalibration(samples),
  };
}

function weeklyCalibration(samples: readonly PredictionSample[]): WeeklyCalibration[] {
  const byWeek = new Map<string, PredictionSample[]>();
  for (const sample of samples) {
    const week = byWeek.get(sample.weekStart);
    if (week === undefined) byWeek.set(sample.weekStart, [sample]);
    else week.push(sample);
  }
  return [...byWeek.entries()]
    .sort(([first], [second]) => compareString(first, second))
    .map(([weekStart, weekSamples]) => ({
      weekStart,
      score: scorePredictions(weekSamples),
      calibration: calibratePredictions(weekSamples),
    }));
}

interface SequentialPredictor {
  predict(match: RatingMatch): number;
  observe(match: RatingMatch): void;
}

function createLegacyPredictor(): SequentialPredictor {
  const ratings: Record<string, number> = {};
  return {
    predict(match) {
      return predictElo(
        String(match.teamA[0]),
        String(match.teamA[1]),
        String(match.teamB[0]),
        String(match.teamB[1]),
        ratings
      ).teamAWin;
    },
    observe(match) {
      const ids = [match.teamA[0], match.teamA[1], match.teamB[0], match.teamB[1]];
      for (const playerId of ids) ratings[String(playerId)] ??= INITIAL_RATING;
      const resultA = resultFor(match);
      const teamA = [match.teamA[0], match.teamA[1]];
      const teamB = [match.teamB[0], match.teamB[1]];
      const before = { ...ratings };
      for (const playerId of teamA) {
        const expected = individualEloExpected(playerId, teamB, before);
        ratings[String(playerId)] = before[String(playerId)] + K_DOUBLES * (resultA - expected);
      }
      for (const playerId of teamB) {
        const expected = individualEloExpected(playerId, teamA, before);
        ratings[String(playerId)] = before[String(playerId)] + K_DOUBLES * ((1 - resultA) - expected);
      }
    },
  };
}

function individualEloExpected(playerId: PlayerId, opponents: readonly PlayerId[], ratings: Record<string, number>): number {
  const opponentAverage = (ratings[String(opponents[0])] + ratings[String(opponents[1])]) / 2;
  return 1 / (1 + 10 ** ((opponentAverage - ratings[String(playerId)]) / 400));
}

function createNaivePredictor(config: RatingConfig): SequentialPredictor {
  const states: Record<string, RatingState> = {};
  return {
    predict(match) {
      const snapshot = naiveSnapshot(match, states, config);
      return (naiveExpected(snapshot, match.teamA[0], match.teamB) + naiveExpected(snapshot, match.teamA[1], match.teamB)) / 2;
    },
    observe(match) {
      const snapshot = naiveSnapshot(match, states, config);
      const resultA = resultFor(match);
      const updates = new Map<PlayerId, RatingState>();
      for (const playerId of match.teamA) updates.set(playerId, naiveUpdate(snapshot, playerId, match.teamB, resultA, config));
      const resultB: 0 | 1 = resultA === 1 ? 0 : 1;
      for (const playerId of match.teamB) updates.set(playerId, naiveUpdate(snapshot, playerId, match.teamA, resultB, config));
      for (const [playerId, state] of updates) states[String(playerId)] = state;
    },
  };
}

function naiveSnapshot(
  match: RatingMatch,
  states: Readonly<Record<string, RatingState>>,
  config: RatingConfig
): Record<string, RatingState> {
  const snapshot: Record<string, RatingState> = {};
  for (const playerId of [match.teamA[0], match.teamA[1], match.teamB[0], match.teamB[1]]) {
    const state = states[String(playerId)];
    snapshot[String(playerId)] = state === undefined ? initialState(config) : { ...state };
  }
  return snapshot;
}

function naiveExpected(
  snapshot: Readonly<Record<string, RatingState>>,
  playerId: PlayerId,
  opponents: readonly [PlayerId, PlayerId]
): number {
  const player = internal(snapshot[String(playerId)]);
  const opponentOne = internal(snapshot[String(opponents[0])]);
  const opponentTwo = internal(snapshot[String(opponents[1])]);
  const opponentX = opponentOne.x + opponentTwo.x;
  const opponentPhi = Math.hypot(opponentOne.phi, opponentTwo.phi);
  return logistic(glickoG(opponentPhi) * (player.x - opponentX));
}

function naiveUpdate(
  snapshot: Readonly<Record<string, RatingState>>,
  playerId: PlayerId,
  opponents: readonly [PlayerId, PlayerId],
  result: 0 | 1,
  config: RatingConfig
): RatingState {
  const before = snapshot[String(playerId)];
  const player = internal(before);
  const opponentOne = internal(snapshot[String(opponents[0])]);
  const opponentTwo = internal(snapshot[String(opponents[1])]);
  const opponentX = opponentOne.x + opponentTwo.x;
  const opponentPhi = Math.hypot(opponentOne.phi, opponentTwo.phi);
  const g = glickoG(opponentPhi);
  const expected = logistic(g * (player.x - opponentX));
  const information = g ** 2 * expected * (1 - expected);
  const variance = 1 / (1 / player.phi ** 2 + information);
  const x = player.x + variance * g * (result - expected);
  const phi = Math.sqrt(variance);
  const r = 1000 + GLICKO2_INTERNAL_SCALE * x;
  const rd = Math.min(config.maxRd, Math.max(config.minRd, GLICKO2_INTERNAL_SCALE * phi));
  if (!Number.isFinite(r) || !Number.isFinite(rd) || rd <= 0) {
    throw new RangeError("naive Glicko-2 update produced an invalid state");
  }
  return { r, rd, volatility: before.volatility };
}

function classifyMatches(
  matches: readonly RatingMatch[],
  playerIds: readonly PlayerId[],
  asOf: string
): ClassifiedMatches {
  const known = new Set(playerIds);
  const effective: EffectiveMatch[] = [];
  let issueCount = 0;
  let futureMatchCount = 0;
  const exclusions: EvaluationExclusion[] = [];
  const matchIdCounts = new Map<number, number>();
  for (const match of matches) {
    if (Number.isSafeInteger(match.id) && match.id >= 0) {
      matchIdCounts.set(match.id, (matchIdCounts.get(match.id) ?? 0) + 1);
    }
  }
  for (const [inputIndex, match] of matches.entries()) {
    if (!Number.isSafeInteger(match.id) || match.id < 0 || (matchIdCounts.get(match.id) ?? 0) > 1) {
      issueCount += 1;
      exclusions.push({ matchId: safeMatchId(match.id), reason: "invalid_match_id" });
      continue;
    }
    if (!isValidLocalDate(match.playedAt)) {
      issueCount += 1;
      exclusions.push({ matchId: match.id, reason: "invalid_date" });
      continue;
    }
    if (isFutureShanghaiLocalDate(match.playedAt, asOf)) {
      futureMatchCount += 1;
      exclusions.push({ matchId: match.id, reason: "future" });
      continue;
    }
    if (!isValidResult(match)) {
      issueCount += 1;
      exclusions.push({ matchId: match.id, reason: "invalid_score" });
      continue;
    }
    const ids = [match.teamA[0], match.teamA[1], match.teamB[0], match.teamB[1]];
    if (new Set(ids).size !== 4) {
      issueCount += 1;
      exclusions.push({ matchId: match.id, reason: "duplicate_player" });
      continue;
    }
    if (ids.some((playerId) => !known.has(playerId))) {
      issueCount += 1;
      exclusions.push({ matchId: match.id, reason: "unknown_player" });
      continue;
    }
    effective.push({ match, inputIndex });
  }
  return { effective, issueCount, futureMatchCount, exclusions };
}

function normalizeThresholds(thresholds: EvaluationEvidenceThresholds | undefined): Required<EvaluationEvidenceThresholds> {
  const minimumMatches = thresholds?.minimumMatches ?? 20;
  const minimumWeeks = thresholds?.minimumWeeks ?? 4;
  for (const [name, value] of Object.entries({ minimumMatches, minimumWeeks })) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be a positive safe integer`);
  }
  return { minimumMatches, minimumWeeks };
}

function validateSource(source: EvaluationSource): void {
  if (source.kind !== "synthetic" && source.kind !== "database") throw new RangeError("source.kind must be synthetic or database");
  if (source.inputVersion.trim() === "" || source.inputHash.trim() === "") {
    throw new RangeError("source inputVersion and inputHash must be non-empty");
  }
}

function compareEffectiveMatches(first: EffectiveMatch, second: EffectiveMatch): number {
  const playedAt = compareString(first.match.playedAt, second.match.playedAt);
  if (playedAt !== 0) return playedAt;
  const createdAt = compareString(first.match.createdAt, second.match.createdAt);
  if (createdAt !== 0) return createdAt;
  const id = first.match.id - second.match.id;
  return id !== 0 ? id : first.inputIndex - second.inputIndex;
}

function toPredictionSample(match: RatingMatch, probability: number): PredictionSample {
  return { matchId: match.id, weekStart: weekStart(match.playedAt), probability, result: resultFor(match) };
}

function resultFor(match: RatingMatch): 0 | 1 {
  if (!isValidResult(match)) throw new RangeError(`match ${match.id} has an invalid score`);
  return match.scoreA > match.scoreB ? 1 : 0;
}

function isValidResult(match: RatingMatch): boolean {
  return (
    Number.isSafeInteger(match.scoreA) &&
    match.scoreA >= 0 &&
    Number.isSafeInteger(match.scoreB) &&
    match.scoreB >= 0 &&
    match.scoreA !== match.scoreB
  );
}

function assertPrediction(prediction: Prediction): void {
  if (!Number.isFinite(prediction.probability) || prediction.probability < 0 || prediction.probability > 1) {
    throw new RangeError("prediction.probability must be finite and within [0, 1]");
  }
  if (prediction.result !== 0 && prediction.result !== 1) throw new RangeError("prediction.result must be 0 or 1");
}

function assertCalibrationEdges(edges: readonly number[]): void {
  if (edges.length < 2 || edges[0] !== 0 || edges.at(-1) !== 1) {
    throw new RangeError("calibration edges must begin at 0 and end at 1");
  }
  for (let index = 0; index < edges.length; index += 1) {
    if (!Number.isFinite(edges[index]) || (index > 0 && edges[index] <= edges[index - 1])) {
      throw new RangeError("calibration edges must be finite and strictly increasing");
    }
  }
}

function calibrationBinIndex(probability: number, edges: readonly number[]): number {
  for (let index = 0; index < edges.length - 1; index += 1) {
    if (probability < edges[index + 1] || index === edges.length - 2) return index;
  }
  throw new Error("unreachable calibration bin");
}

function initialState(config: RatingConfig): RatingState {
  return { r: config.initialRating, rd: config.initialRd, volatility: config.initialVolatility };
}

function internal(state: RatingState): { x: number; phi: number } {
  return { x: displayRatingToInternalX(state.r), phi: displayRdToInternalPhi(state.rd) };
}

function glickoG(phi: number): number {
  return 1 / Math.hypot(1, (Math.sqrt(3) / Math.PI) * phi);
}

function logistic(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponent = Math.exp(value);
  return exponent / (1 + exponent);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function emptyWeeklyCorrections(): WeeklyCorrectionSummary {
  return {
    segmentCount: 0,
    playerCorrectionCount: 0,
    meanSignedCorrection: null,
    meanAbsoluteCorrection: null,
    maxAbsoluteCorrection: null,
  };
}

function weeklyFinalEvents(
  events: readonly RatingEvent[]
): readonly Extract<RatingEvent, { kind: "weekly_final" }>[] {
  return events.filter((event): event is Extract<RatingEvent, { kind: "weekly_final" }> => event.kind === "weekly_final");
}

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : sum(values) / values.length;
}

function percentile(sorted: readonly number[], quantile: number): number | null {
  if (sorted.length === 0) return null;
  const index = Math.ceil(quantile * sorted.length) - 1;
  return sorted[index];
}

function safeMatchId(value: number): number | null {
  return Number.isSafeInteger(value) ? value : null;
}

function compareString(first: string, second: string): number {
  return first < second ? -1 : first > second ? 1 : 0;
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
