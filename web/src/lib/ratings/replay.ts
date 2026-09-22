import { ratingConfigVersion, validateRatingConfig } from "./config";
import {
  isFutureShanghaiLocalDate,
  isValidLocalDate,
  iterateRatingSegments,
  ratingSegmentAt,
  shanghaiLocalDateFromInstant,
  shanghaiMidnightIso,
} from "./calendar";
import { estimateNextMatch, finalizeSegment, prepareEstimated } from "./segment";
import type {
  MatchEstimate,
  PeriodFinal,
  PlayerChange,
  PlayerId,
  RatingConfig,
  RatingEvent,
  RatingIssue,
  RatingMatch,
  RatingReplay,
  RatingSegment,
  RatingState,
  SeasonReset,
} from "./types";

export interface ReplayOptions {
  readonly config: RatingConfig;
  readonly asOf: string;
}

interface EffectiveMatch {
  readonly match: RatingMatch;
  readonly inputIndex: number;
}

/**
 * Rebuilds one deterministic Glicko-2 doubles timeline as it stood on `asOf`.
 * It begins at the first valid, effective game, so players do not gain RD
 * history before they have actually played.
 */
export function replayRatings(
  matches: readonly RatingMatch[],
  playerIds: readonly PlayerId[],
  options: ReplayOptions
): RatingReplay {
  const config = validateRatingConfig(options.config);
  const configVersion = ratingConfigVersion(config);
  const naturalCurrentSegment = ratingSegmentAt(options.asOf, config.firstSeasonStart);

  assertDistinctSafeMatchIds(matches);
  const { effective, issues } = validateEffectiveMatches(matches, playerIds, options.asOf);
  if (effective.length === 0) {
    return {
      asOf: options.asOf,
      configVersion,
      currentSegment: cloneSegment(naturalCurrentSegment),
      current: {},
      lastFinal: {},
      events: [],
      matchEstimates: {},
      issues,
      nextBoundary: naturalCurrentSegment.end,
    };
  }

  const ordered = [...effective].sort(compareEffectiveMatches);
  const firstDate = ordered[0].match.playedAt;
  const segments = iterateRatingSegments({
    start: shanghaiMidnightIso(firstDate),
    end: naturalCurrentSegment.end,
    firstSeasonStart: config.firstSeasonStart,
  });
  const matchesBySegment = groupMatchesBySegment(ordered, segments);

  let lastFinal: Record<string, RatingState> = {};
  let current: Record<string, RatingState> = {};
  const events: RatingEvent[] = [];
  const matchEstimates: Record<string, MatchEstimate> = {};

  for (const segment of segments) {
    const segmentMatches = matchesBySegment.get(segment.id) ?? [];
    initializeSegmentPlayers(lastFinal, segmentMatches, config);

    let working = prepareEstimated(lastFinal, segment, config);
    for (const match of segmentMatches) {
      const estimated = estimateNextMatch(match, working, segment, config);
      const event = copyMatchEstimate(estimated);
      working = estimated.states;
      matchEstimates[String(match.id)] = copyMatchEstimate(event);
      events.push(event);
    }

    if (segment.end <= naturalCurrentSegment.start) {
      const final = copyPeriodFinal(finalizeSegment(lastFinal, segmentMatches, segment, config));
      events.push(final);
      lastFinal = cloneStates(final.final);

      const reset = resetAtSeasonBoundary(lastFinal, segment, config);
      if (reset !== undefined) {
        events.push(reset.event);
        lastFinal = reset.states;
      }
    }
    current = working;
  }

  const currentSegment = segments.at(-1);
  if (currentSegment === undefined) throw new Error("expected an active rating segment");
  return {
    asOf: options.asOf,
    configVersion,
    currentSegment: cloneSegment(currentSegment),
    current: cloneStates(current),
    lastFinal: cloneStates(lastFinal),
    events,
    matchEstimates,
    issues,
    nextBoundary: currentSegment.end,
  };
}

function assertDistinctSafeMatchIds(matches: readonly RatingMatch[]): void {
  const ids = new Set<number>();
  for (const match of matches) {
    if (!Number.isSafeInteger(match.id) || match.id < 0) {
      throw new RangeError("match.id must be a non-negative safe integer");
    }
    if (ids.has(match.id)) throw new RangeError("matches must not contain duplicate match.id values");
    ids.add(match.id);
  }
}

function validateEffectiveMatches(
  matches: readonly RatingMatch[],
  playerIds: readonly PlayerId[],
  asOf: string
): { effective: EffectiveMatch[]; issues: RatingIssue[] } {
  const knownPlayerIds = new Set(playerIds);
  const effective: EffectiveMatch[] = [];
  const issues: RatingIssue[] = [];

  for (const [inputIndex, match] of matches.entries()) {
    if (!isValidLocalDate(match.playedAt)) {
      issues.push({ matchId: match.id, reason: "invalid_date" });
      continue;
    }
    if (isFutureShanghaiLocalDate(match.playedAt, asOf)) continue;
    if (!isValidScore(match)) {
      issues.push({ matchId: match.id, reason: "invalid_score" });
      continue;
    }
    const ids = matchPlayerIds(match);
    if (new Set(ids).size !== 4) {
      issues.push({ matchId: match.id, reason: "duplicate_player" });
      continue;
    }
    if (ids.some((playerId) => !knownPlayerIds.has(playerId))) {
      issues.push({ matchId: match.id, reason: "unknown_player" });
      continue;
    }
    effective.push({ match, inputIndex });
  }

  issues.sort((first, second) => first.matchId - second.matchId);
  return { effective, issues };
}

function compareEffectiveMatches(first: EffectiveMatch, second: EffectiveMatch): number {
  const playedAt = compareString(first.match.playedAt, second.match.playedAt);
  if (playedAt !== 0) return playedAt;

  const createdAt = compareString(first.match.createdAt, second.match.createdAt);
  if (createdAt !== 0) return createdAt;

  const id = first.match.id - second.match.id;
  return id !== 0 ? id : first.inputIndex - second.inputIndex;
}

function groupMatchesBySegment(
  matches: readonly EffectiveMatch[],
  segments: readonly RatingSegment[]
): Map<string, RatingMatch[]> {
  const groups = new Map<string, RatingMatch[]>();
  let segmentIndex = 0;
  for (const { match } of matches) {
    while (
      segmentIndex < segments.length &&
      match.playedAt >= shanghaiLocalDateFromInstant(segments[segmentIndex].end)
    ) {
      segmentIndex += 1;
    }
    const segment = segments[segmentIndex];
    if (
      segment === undefined ||
      match.playedAt < shanghaiLocalDateFromInstant(segment.start) ||
      match.playedAt >= shanghaiLocalDateFromInstant(segment.end)
    ) {
      throw new RangeError(`could not place match ${match.id} in a rating segment`);
    }
    const group = groups.get(segment.id);
    if (group === undefined) groups.set(segment.id, [match]);
    else group.push(match);
  }
  return groups;
}

function initializeSegmentPlayers(
  states: Record<string, RatingState>,
  matches: readonly RatingMatch[],
  config: RatingConfig
): void {
  const ids = new Set<PlayerId>();
  for (const match of matches) {
    for (const playerId of matchPlayerIds(match)) ids.add(playerId);
  }
  for (const playerId of [...ids].sort((first, second) => first - second)) {
    if (states[String(playerId)] === undefined) states[String(playerId)] = initialState(config);
  }
}

function resetAtSeasonBoundary(
  states: Record<string, RatingState>,
  segment: RatingSegment,
  config: RatingConfig
): { event: SeasonReset; states: Record<string, RatingState> } | undefined {
  const seasonId = shanghaiLocalDateFromInstant(segment.end);
  if (!isQuarterStart(seasonId) || seasonId < config.firstSeasonStart || Object.keys(states).length === 0) {
    return undefined;
  }

  const nextStates: Record<string, RatingState> = {};
  const changes: PlayerChange[] = [];
  for (const playerId of sortedPlayerIds(states)) {
    const before = cloneState(states[playerId]);
    const after = {
      r: softResetRating(before.r, config),
      rd: Math.min(config.maxRd, Math.max(before.rd, config.seasonRdFloor)),
      volatility: before.volatility,
    };
    nextStates[playerId] = after;
    changes.push({ playerId: Number(playerId), before, after: cloneState(after), delta: after.r - before.r });
  }
  return {
    event: {
      kind: "season_reset",
      eventId: `season_reset:${seasonId}`,
      at: segment.end,
      seasonId,
      changes,
    },
    states: nextStates,
  };
}

function isValidScore(match: RatingMatch): boolean {
  return (
    Number.isSafeInteger(match.scoreA) &&
    match.scoreA >= 0 &&
    Number.isSafeInteger(match.scoreB) &&
    match.scoreB >= 0 &&
    match.scoreA !== match.scoreB
  );
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

function softResetRating(rating: number, config: RatingConfig): number {
  if (rating < config.seasonLower) {
    return config.seasonLower + config.seasonRetention * (rating - config.seasonLower);
  }
  if (rating > config.seasonUpper) {
    return config.seasonUpper + config.seasonRetention * (rating - config.seasonUpper);
  }
  return rating;
}

function isQuarterStart(localDate: string): boolean {
  return /^\d{4}-(01|04|07|10)-01$/.test(localDate);
}

function copyMatchEstimate(estimate: {
  readonly kind: "match_estimated";
  readonly eventId: string;
  readonly matchId: number;
  readonly segmentId: string;
  readonly playedAt: string;
  readonly preWinA: number;
  readonly changes: readonly PlayerChange[];
}): MatchEstimate {
  return {
    kind: "match_estimated",
    eventId: estimate.eventId,
    matchId: estimate.matchId,
    segmentId: estimate.segmentId,
    playedAt: estimate.playedAt,
    preWinA: estimate.preWinA,
    changes: estimate.changes.map(copyPlayerChange),
  };
}

function copyPeriodFinal(final: PeriodFinal): PeriodFinal {
  return {
    kind: "weekly_final",
    eventId: final.eventId,
    segment: cloneSegment(final.segment),
    start: cloneStates(final.start),
    estimatedEnd: cloneStates(final.estimatedEnd),
    final: cloneStates(final.final),
    correction: { ...final.correction },
  };
}

function copyPlayerChange(change: PlayerChange): PlayerChange {
  return {
    playerId: change.playerId,
    before: cloneState(change.before),
    after: cloneState(change.after),
    delta: change.delta,
  };
}

function cloneStates(states: Readonly<Record<string, RatingState>>): Record<string, RatingState> {
  const clone: Record<string, RatingState> = {};
  for (const playerId of sortedPlayerIds(states)) clone[playerId] = cloneState(states[playerId]);
  return clone;
}

function cloneState(state: RatingState): RatingState {
  return { r: state.r, rd: state.rd, volatility: state.volatility };
}

function cloneSegment(segment: RatingSegment): RatingSegment {
  return { ...segment };
}

function sortedPlayerIds(states: Readonly<Record<string, RatingState>>): string[] {
  return Object.keys(states).sort((first, second) => Number(first) - Number(second));
}

function compareString(first: string, second: string): number {
  return first < second ? -1 : first > second ? 1 : 0;
}
