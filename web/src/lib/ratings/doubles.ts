import {
  GLICKO2_INTERNAL_SCALE,
  displayRatingToInternalX,
  displayRdToInternalPhi,
} from "./config";
import { updateGlicko2 } from "./glicko2-core";
import type { PlayerId, RatingConfig, RatingState } from "./types";

export type DoublesRatingStates = Readonly<
  Record<PlayerId, Readonly<RatingState> | undefined>
>;

export interface DoublesMatchInput {
  readonly teamA: readonly [PlayerId, PlayerId];
  readonly teamB: readonly [PlayerId, PlayerId];
  readonly states: DoublesRatingStates;
  readonly config: RatingConfig;
  readonly resultA: 0 | 1;
}

export interface DoublesPlayerChange {
  readonly playerId: PlayerId;
  readonly before: RatingState;
  readonly after: RatingState;
  readonly expectedScore: number;
  readonly delta: number;
}

export interface DoublesMatchUpdate {
  readonly preWinA: number;
  readonly changes: readonly DoublesPlayerChange[];
}

/** One player's complete, pre-match evidence for a doubles result. */
export interface DoublesPlayerEvidence {
  readonly playerId: PlayerId;
  readonly before: RatingState;
  readonly playerX: number;
  readonly playerPhi: number;
  readonly opponentX: number;
  readonly opponentPhi: number;
  readonly result: 0 | 1;
  readonly expectedScore: number;
}

/** Shared four-player evidence built from a single immutable pre-match snapshot. */
export interface DoublesMatchEvidence {
  readonly preWinA: number;
  readonly players: readonly DoublesPlayerEvidence[];
}

/** Returns the team-A win probability from the public doubles display formula. */
export function predictDoubles(
  teamA: readonly [PlayerId, PlayerId],
  teamB: readonly [PlayerId, PlayerId],
  states: DoublesRatingStates,
  config: RatingConfig
): number {
  assertDistinctPlayers(teamA[0], teamA[1], teamB[0], teamB[1]);
  const [a1, a2] = teamA.map((playerId) => internalState(playerId, states, config));
  const [b1, b2] = teamB.map((playerId) => internalState(playerId, states, config));
  return predictFromInternalStates(a1, a2, b1, b2);
}

/** Resolves an unknown player to a temporary initial state without storing it. */
export function resolveDoublesState(
  playerId: PlayerId,
  states: DoublesRatingStates,
  config: RatingConfig
): RatingState {
  const state = states[playerId];
  if (state !== undefined) {
    return { r: state.r, rd: state.rd, volatility: state.volatility };
  }
  return {
    r: config.initialRating,
    rd: config.initialRd,
    volatility: config.initialVolatility,
  };
}

/**
 * Returns one player's expected score against their virtual opponent:
 * the two opposing players minus the player's teammate.
 */
export function expectedDoublesScore(
  playerId: PlayerId,
  teammateId: PlayerId,
  opponents: readonly [PlayerId, PlayerId],
  states: DoublesRatingStates,
  config: RatingConfig
): number {
  assertDistinctPlayers(playerId, teammateId, opponents[0], opponents[1]);
  const player = internalState(playerId, states, config);
  const teammate = internalState(teammateId, states, config);
  const [opponentOne, opponentTwo] = opponents.map((opponentId) =>
    internalState(opponentId, states, config)
  );
  const virtualX = opponentOne.x + opponentTwo.x - teammate.x;
  const virtualPhi = Math.hypot(opponentOne.phi, opponentTwo.phi, teammate.phi);
  return logistic(glickoG(virtualPhi) * (player.x - virtualX));
}

/**
 * Builds virtual-opponent observations for all four players from one pre-match snapshot.
 * Unknown players are represented only by their temporary configured initial states.
 */
export function buildDoublesMatchEvidence(input: DoublesMatchInput): DoublesMatchEvidence {
  const { teamA, teamB, states, config, resultA } = input;
  assertDistinctPlayers(teamA[0], teamA[1], teamB[0], teamB[1]);
  const playerIds = [teamA[0], teamA[1], teamB[0], teamB[1]] as const;
  const snapshot = new Map<PlayerId, RatingState>(
    playerIds.map((playerId) => [playerId, resolveDoublesState(playerId, states, config)])
  );
  const [a1, a2, b1, b2] = playerIds.map((playerId) =>
    internalStateFromDisplay(snapshot.get(playerId)!)
  );
  const resultB: 0 | 1 = resultA === 1 ? 0 : 1;

  return {
    preWinA: predictFromInternalStates(a1, a2, b1, b2),
    players: [
      buildPlayerEvidence(teamA[0], snapshot.get(teamA[0])!, a1, a2, b1, b2, resultA),
      buildPlayerEvidence(teamA[1], snapshot.get(teamA[1])!, a2, a1, b1, b2, resultA),
      buildPlayerEvidence(teamB[0], snapshot.get(teamB[0])!, b1, b2, a1, a2, resultB),
      buildPlayerEvidence(teamB[1], snapshot.get(teamB[1])!, b2, b1, a1, a2, resultB),
    ],
  };
}

/**
 * Applies a complete Glicko-2 rating-period update, including volatility re-estimation.
 * Do not use this for T5 intra-week Estimated updates, which must freeze volatility.
 */
export function updateDoublesMatch(input: DoublesMatchInput): DoublesMatchUpdate {
  const evidence = buildDoublesMatchEvidence(input);

  return {
    preWinA: evidence.preWinA,
    changes: evidence.players.map((player) => updatePlayerFromEvidence(player, input.config)),
  };
}

interface InternalState {
  readonly x: number;
  readonly phi: number;
}

function buildPlayerEvidence(
  playerId: PlayerId,
  before: RatingState,
  player: InternalState,
  teammate: InternalState,
  opponentOne: InternalState,
  opponentTwo: InternalState,
  result: 0 | 1
): DoublesPlayerEvidence {
  const opponentX = opponentOne.x + opponentTwo.x - teammate.x;
  const opponentPhi = Math.hypot(opponentOne.phi, opponentTwo.phi, teammate.phi);
  return {
    playerId,
    before,
    playerX: player.x,
    playerPhi: player.phi,
    opponentX,
    opponentPhi,
    result,
    expectedScore: logistic(glickoG(opponentPhi) * (player.x - opponentX)),
  };
}

function updatePlayerFromEvidence(
  evidence: DoublesPlayerEvidence,
  config: RatingConfig
): DoublesPlayerChange {
  const afterInternal = updateGlicko2(
    {
      x: evidence.playerX,
      phi: evidence.playerPhi,
      sigma: evidence.before.volatility,
    },
    [
      {
        opponentX: evidence.opponentX,
        opponentPhi: evidence.opponentPhi,
        result: evidence.result,
      },
    ],
    config.tau
  );
  const after = {
    r: 1000 + GLICKO2_INTERNAL_SCALE * afterInternal.x,
    rd: GLICKO2_INTERNAL_SCALE * afterInternal.phi,
    volatility: afterInternal.sigma,
  };

  return {
    playerId: evidence.playerId,
    before: evidence.before,
    after,
    expectedScore: evidence.expectedScore,
    delta: after.r - evidence.before.r,
  };
}

function internalState(
  playerId: PlayerId,
  states: DoublesRatingStates,
  config: RatingConfig
): InternalState {
  return internalStateFromDisplay(resolveDoublesState(playerId, states, config));
}

function internalStateFromDisplay(state: RatingState): InternalState {
  return {
    x: displayRatingToInternalX(state.r),
    phi: displayRdToInternalPhi(state.rd),
  };
}

function predictFromInternalStates(
  a1: InternalState,
  a2: InternalState,
  b1: InternalState,
  b2: InternalState
): number {
  const difference = a1.x + a2.x - b1.x - b2.x;
  const uncertainty = Math.hypot(a1.phi, a2.phi, b1.phi, b2.phi);
  return logistic(glickoG(uncertainty) * difference);
}

function assertDistinctPlayers(...playerIds: readonly PlayerId[]): void {
  if (new Set(playerIds).size !== 4) {
    throw new RangeError("a doubles matchup requires four distinct player IDs");
  }
}

function glickoG(phi: number): number {
  return 1 / Math.hypot(1, (Math.sqrt(3) / Math.PI) * phi);
}

function logistic(value: number): number {
  if (value >= 0) {
    return 1 / (1 + Math.exp(-value));
  }
  const exponent = Math.exp(value);
  return exponent / (1 + exponent);
}
