import {
  GLICKO2_INTERNAL_SCALE,
  displayRatingToInternalX,
  displayRdToInternalPhi,
  validateRatingConfig,
} from "./config";
import {
  buildDoublesMatchEvidence,
  type DoublesPlayerEvidence,
  type DoublesRatingStates,
} from "./doubles";
import { updateGlicko2 } from "./glicko2-core";
import type {
  MatchEstimate,
  PeriodFinal,
  PlayerChange,
  PlayerId,
  RatingConfig,
  RatingMatch,
  RatingSegment,
  RatingState,
} from "./types";

export type SegmentRatingStates = DoublesRatingStates;

/** An Estimated event together with its explicit next working snapshot. */
export interface EstimatedMatch extends MatchEstimate {
  readonly states: Record<string, RatingState>;
}

/**
 * Starts an Estimated segment by adding its process variance exactly once.
 * The returned map is a working snapshot; callers must pass it to subsequent
 * `estimateNextMatch` calls rather than prepare it again.
 */
export function prepareEstimated(
  baseStates: SegmentRatingStates,
  segment: RatingSegment,
  config: RatingConfig
): Record<string, RatingState> {
  validateRatingConfig(config);
  const sqrtH = assertSegment(segment);
  const prepared: Record<string, RatingState> = {};

  for (const [playerId, state] of stateEntries(baseStates)) {
    prepared[playerId] = prepareState(state, sqrtH, config);
  }
  return prepared;
}

/**
 * Applies one synchronous, intra-segment Estimated update. Volatility remains
 * frozen because segment process variance was already added by preparation.
 */
export function estimateNextMatch(
  match: RatingMatch,
  workingStates: SegmentRatingStates,
  segment: RatingSegment,
  config: RatingConfig
): EstimatedMatch {
  validateRatingConfig(config);
  const sqrtH = assertSegment(segment);
  const resultA = matchResult(match);
  const nextStates = cloneStates(workingStates);

  for (const playerId of matchPlayerIds(match)) {
    if (nextStates[playerId] === undefined) {
      nextStates[playerId] = prepareState(initialState(config), sqrtH, config);
    }
  }

  const evidence = buildDoublesMatchEvidence({
    teamA: match.teamA,
    teamB: match.teamB,
    states: nextStates,
    config,
    resultA,
  });
  const changes = evidence.players.map((player) => frozenObservationUpdate(player, config));
  for (const change of changes) {
    nextStates[change.playerId] = cloneState(change.after);
  }

  return {
    kind: "match_estimated",
    eventId: `match_estimated:${segment.id}:${match.id}`,
    matchId: match.id,
    segmentId: segment.id,
    playedAt: match.playedAt,
    preWinA: evidence.preWinA,
    changes,
    states: nextStates,
  };
}

/**
 * Calculates the segment's official batch result from its formal start state.
 * Estimated snapshots are deliberately never accepted as a Final input.
 */
export function finalizeSegment(
  baseStates: SegmentRatingStates,
  segmentMatches: readonly RatingMatch[],
  segment: RatingSegment,
  config: RatingConfig
): PeriodFinal {
  validateRatingConfig(config);
  const sqrtH = assertSegment(segment);
  const matches = segmentMatches.map((match) => ({ match, resultA: matchResult(match) }));
  assertUniqueMatchIds(matches);
  const start = cloneStates(baseStates);

  for (const { match } of matches) {
    for (const playerId of matchPlayerIds(match)) {
      if (start[playerId] === undefined) start[playerId] = initialState(config);
    }
  }

  const observationsByPlayer = collectFinalObservations(start, matches, config);
  const final = finalizeAllPlayers(start, observationsByPlayer, sqrtH, config);
  const estimatedEnd = estimateSegmentEnd(start, matches.map(({ match }) => match), segment, config);
  const correction: Record<string, number> = {};
  for (const playerId of sortedPlayerIds(final)) {
    correction[playerId] = final[playerId].r - estimatedEnd[playerId].r;
  }

  return {
    kind: "weekly_final",
    eventId: `weekly_final:${segment.id}`,
    segment: { ...segment },
    start,
    estimatedEnd,
    final,
    correction,
  };
}

function collectFinalObservations(
  start: Record<string, RatingState>,
  matches: readonly { readonly match: RatingMatch; readonly resultA: 0 | 1 }[],
  config: RatingConfig
): Map<PlayerId, DoublesPlayerEvidence[]> {
  const observationsByPlayer = new Map<PlayerId, DoublesPlayerEvidence[]>();
  for (const playerId of sortedPlayerIds(start)) observationsByPlayer.set(Number(playerId), []);

  const sortedMatches = [...matches].sort((first, second) => first.match.id - second.match.id);
  for (const { match, resultA } of sortedMatches) {
    const evidence = buildDoublesMatchEvidence({
      teamA: match.teamA,
      teamB: match.teamB,
      states: start,
      config,
      resultA,
    });
    for (const player of evidence.players) {
      const observations = observationsByPlayer.get(player.playerId);
      if (observations === undefined) {
        throw new RangeError(`missing final baseline for player ${player.playerId}`);
      }
      observations.push(player);
    }
  }
  return observationsByPlayer;
}

function finalizeAllPlayers(
  start: Record<string, RatingState>,
  observationsByPlayer: ReadonlyMap<PlayerId, readonly DoublesPlayerEvidence[]>,
  sqrtH: number,
  config: RatingConfig
): Record<string, RatingState> {
  const final: Record<string, RatingState> = {};
  for (const playerId of sortedPlayerIds(start)) {
    const state = start[playerId];
    const observations = observationsByPlayer.get(Number(playerId));
    if (observations === undefined) throw new RangeError(`missing observations for player ${playerId}`);
    final[playerId] = fullPeriodUpdate(state, observations, sqrtH, config);
  }
  return final;
}

function estimateSegmentEnd(
  start: Record<string, RatingState>,
  matches: readonly RatingMatch[],
  segment: RatingSegment,
  config: RatingConfig
): Record<string, RatingState> {
  let working = prepareEstimated(start, segment, config);
  for (const match of matches) {
    working = estimateNextMatch(match, working, segment, config).states;
  }
  return working;
}

function fullPeriodUpdate(
  before: RatingState,
  observations: readonly DoublesPlayerEvidence[],
  sqrtH: number,
  config: RatingConfig
): RatingState {
  assertRatingState(before);
  const updated = updateGlicko2(
    {
      x: displayRatingToInternalX(before.r),
      phi: displayRdToInternalPhi(before.rd),
      sigma: before.volatility * sqrtH,
    },
    observations.map((observation) => ({
      opponentX: observation.opponentX,
      opponentPhi: observation.opponentPhi,
      result: observation.result,
    })),
    config.tau
  );
  const volatility = updated.sigma / sqrtH;
  const rd = clampRd(GLICKO2_INTERNAL_SCALE * updated.phi, config);
  if (!Number.isFinite(volatility) || volatility <= 0 || !Number.isFinite(rd)) {
    throw new RangeError("Glicko-2 Final update produced an invalid display state");
  }
  return {
    r: 1000 + GLICKO2_INTERNAL_SCALE * updated.x,
    rd,
    volatility,
  };
}

function frozenObservationUpdate(
  evidence: DoublesPlayerEvidence,
  config: RatingConfig
): PlayerChange {
  const { playerX, playerPhi, opponentX, opponentPhi, result } = evidence;
  if (!Number.isFinite(playerX) || !Number.isFinite(opponentX) || playerPhi <= 0 || opponentPhi < 0) {
    throw new RangeError("Glicko-2 Estimated evidence is invalid");
  }
  const g = 1 / Math.hypot(1, (Math.sqrt(3) / Math.PI) * opponentPhi);
  const expectedScore = logistic(g * (playerX - opponentX));
  const information = g ** 2 * expectedScore * (1 - expectedScore);
  const scoreDifference = g * (result - expectedScore);
  const posteriorVariance = 1 / (1 / playerPhi ** 2 + information);
  const afterX = playerX + posteriorVariance * scoreDifference;
  const afterPhi = Math.sqrt(posteriorVariance);
  if (!Number.isFinite(afterX) || !Number.isFinite(afterPhi) || afterPhi <= 0) {
    throw new RangeError("Glicko-2 Estimated update produced an invalid state");
  }
  const before = cloneState(evidence.before);
  const after = {
    r: 1000 + GLICKO2_INTERNAL_SCALE * afterX,
    rd: clampRd(GLICKO2_INTERNAL_SCALE * afterPhi, config),
    volatility: before.volatility,
  };
  return {
    playerId: evidence.playerId,
    before,
    after,
    delta: after.r - before.r,
  };
}

function prepareState(state: RatingState, sqrtH: number, config: RatingConfig): RatingState {
  assertRatingState(state);
  const phi = displayRdToInternalPhi(state.rd);
  const rd = clampRd(GLICKO2_INTERNAL_SCALE * Math.hypot(phi, sqrtH * state.volatility), config);
  if (!Number.isFinite(rd)) throw new RangeError("Glicko-2 Estimated preparation produced an invalid RD");
  return { r: state.r, rd, volatility: state.volatility };
}

function matchResult(match: RatingMatch): 0 | 1 {
  if (
    !Number.isSafeInteger(match.scoreA) ||
    match.scoreA < 0 ||
    !Number.isSafeInteger(match.scoreB) ||
    match.scoreB < 0 ||
    match.scoreA === match.scoreB
  ) {
    throw new RangeError("scoreA and scoreB must be non-negative safe integers and unequal");
  }
  if (!Number.isSafeInteger(match.id) || match.id < 0) {
    throw new RangeError("match.id must be a non-negative safe integer");
  }
  return match.scoreA > match.scoreB ? 1 : 0;
}

function assertUniqueMatchIds(
  matches: readonly { readonly match: RatingMatch; readonly resultA: 0 | 1 }[]
): void {
  const ids = new Set<number>();
  for (const { match } of matches) {
    if (ids.has(match.id)) {
      throw new RangeError("segmentMatches must not contain duplicate match.id values");
    }
    ids.add(match.id);
  }
}

function matchPlayerIds(match: RatingMatch): readonly PlayerId[] {
  return [match.teamA[0], match.teamA[1], match.teamB[0], match.teamB[1]];
}

function initialState(config: RatingConfig): RatingState {
  return {
    r: config.initialRating,
    rd: config.initialRd,
    volatility: config.initialVolatility,
  };
}

function cloneStates(states: SegmentRatingStates): Record<string, RatingState> {
  const clone: Record<string, RatingState> = {};
  for (const [playerId, state] of stateEntries(states)) clone[playerId] = cloneState(state);
  return clone;
}

function stateEntries(states: SegmentRatingStates): readonly [string, RatingState][] {
  return Object.entries(states).flatMap(([playerId, state]) => {
    if (state === undefined) return [];
    return [[playerId, state] as [string, RatingState]];
  });
}

function cloneState(state: RatingState): RatingState {
  assertRatingState(state);
  return { r: state.r, rd: state.rd, volatility: state.volatility };
}

function assertRatingState(state: RatingState): void {
  if (
    !Number.isFinite(state.r) ||
    !Number.isFinite(state.rd) ||
    state.rd < 0 ||
    !Number.isFinite(state.volatility) ||
    state.volatility <= 0
  ) {
    throw new RangeError("rating state must contain finite r, non-negative RD, and positive volatility");
  }
}

function assertSegment(segment: RatingSegment): number {
  if (typeof segment.id !== "string" || segment.id.trim() === "") {
    throw new RangeError("segment.id must be a non-empty string");
  }
  if (!Number.isFinite(segment.h) || segment.h <= 0 || segment.h > 1) {
    throw new RangeError("segment.h must be a finite value greater than zero and at most one");
  }
  return Math.sqrt(segment.h);
}

function clampRd(rd: number, config: RatingConfig): number {
  return Math.min(config.maxRd, Math.max(config.minRd, rd));
}

function sortedPlayerIds(states: Readonly<Record<string, RatingState>>): string[] {
  return Object.keys(states).sort((first, second) => Number(first) - Number(second));
}

function logistic(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponent = Math.exp(value);
  return exponent / (1 + exponent);
}
