import { describe, expect, it } from "vitest";
import {
  GLICKO2_INTERNAL_SCALE,
  createRatingConfig,
  displayRatingToInternalX,
  displayRdToInternalPhi,
} from "./config";
import {
  buildDoublesMatchEvidence,
  expectedDoublesScore,
  predictDoubles,
  resolveDoublesState,
  updateDoublesMatch,
} from "./doubles";
import { updateGlicko2 } from "./glicko2-core";
import type { RatingState } from "./types";

const config = createRatingConfig({ firstSeasonStart: "2024-01-01" });

function statesFor(
  values: readonly [number, number, number, number]
): Readonly<Record<number, RatingState>> {
  return Object.fromEntries(
    values.map((rating, index) => [
      index + 1,
      { r: rating, rd: 120, volatility: 0.06 },
    ])
  );
}

describe("Glicko-2 doubles", () => {
  it("predicts fifty percent for equal combined rating totals", () => {
    expect(predictDoubles([1, 2], [3, 4], statesFor([1400, 800, 1100, 1100]), config)).toBeCloseTo(0.5, 15);
    expect(predictDoubles([1, 2], [3, 4], statesFor([1000, 1000, 1000, 1000]), config)).toBeCloseTo(0.5, 15);
  });

  it("keeps public probabilities symmetric under partner swaps, team swaps, and rating shifts", () => {
    const states = {
      1: { r: 1320, rd: 80, volatility: 0.05 },
      2: { r: 980, rd: 140, volatility: 0.06 },
      3: { r: 1180, rd: 100, volatility: 0.04 },
      4: { r: 1050, rd: 160, volatility: 0.07 },
    };
    const winA = predictDoubles([1, 2], [3, 4], states, config);
    const shifted = Object.fromEntries(
      Object.entries(states).map(([playerId, state]) => [
        playerId,
        { ...state, r: state.r + 75 },
      ])
    );

    expect(predictDoubles([2, 1], [3, 4], states, config)).toBeCloseTo(winA, 15);
    expect(predictDoubles([3, 4], [1, 2], states, config)).toBeCloseTo(1 - winA, 15);
    expect(predictDoubles([1, 2], [3, 4], shifted, config)).toBeCloseTo(winA, 15);
  });

  it("uses an explicit temporary initial state for unknown players without changing supplied states", () => {
    const states: Readonly<Record<number, RatingState | undefined>> = Object.freeze({
      1: Object.freeze({ r: 1250, rd: 80, volatility: 0.06 }),
      3: Object.freeze({ r: 1000, rd: 100, volatility: 0.06 }),
      4: Object.freeze({ r: 1000, rd: 100, volatility: 0.06 }),
    });
    const before = structuredClone(states);
    const temporary = resolveDoublesState(2, states, config);
    const withExplicitInitial = {
      ...states,
      2: {
        r: config.initialRating,
        rd: config.initialRd,
        volatility: config.initialVolatility,
      },
    };

    expect(temporary).toEqual(withExplicitInitial[2]);
    expect(predictDoubles([1, 2], [3, 4], states, config)).toBeCloseTo(
      predictDoubles([1, 2], [3, 4], withExplicitInitial, config),
      15
    );
    expect(states).toEqual(before);
    expect(states[2]).toBeUndefined();
  });

  it("keeps an unknown teammate RD in the personal evidence strength", () => {
    const unknownPartnerStates = {
      1: { r: 1250, rd: 80, volatility: 0.06 },
      3: { r: 1000, rd: 100, volatility: 0.06 },
      4: { r: 1000, rd: 100, volatility: 0.06 },
    };
    const knownCertainPartnerStates = {
      ...unknownPartnerStates,
      2: { r: 1000, rd: 60, volatility: 0.06 },
    };

    expect(expectedDoublesScore(1, 2, [3, 4], unknownPartnerStates, config)).toBeLessThan(
      expectedDoublesScore(1, 2, [3, 4], knownCertainPartnerStates, config)
    );
  });

  it("exposes a personal conditional expectation that can differ from the public team probability", () => {
    const states = {
      1: { r: 1300, rd: 80, volatility: 0.06 },
      2: { r: 1000, rd: 240, volatility: 0.06 },
      3: { r: 1150, rd: 100, volatility: 0.06 },
      4: { r: 1000, rd: 100, volatility: 0.06 },
    };
    const publicProbability = predictDoubles([1, 2], [3, 4], states, config);

    expect(expectedDoublesScore(2, 1, [3, 4], states, config)).not.toBeCloseTo(
      publicProbability,
      3
    );
  });

  it("rejects a doubles matchup that repeats a player", () => {
    const states = statesFor([1000, 1000, 1000, 1000]);

    expect(() => predictDoubles([1, 2], [2, 4], states, config)).toThrow(/four distinct player IDs/);
  });

  it("builds four immutable virtual-opponent observations from one pre-match snapshot", () => {
    const states: Readonly<Record<number, RatingState | undefined>> = Object.freeze({
      1: Object.freeze({ r: 1250, rd: 80, volatility: 0.04 }),
      3: Object.freeze({ r: 1100, rd: 100, volatility: 0.06 }),
      4: Object.freeze({ r: 900, rd: 120, volatility: 0.07 }),
    });
    const before = structuredClone(states);
    const evidence = buildDoublesMatchEvidence({
      teamA: [1, 2],
      teamB: [3, 4],
      states,
      config,
      resultA: 1,
    });
    const playerOne = evidence.players.find(({ playerId }) => playerId === 1);
    const temporaryPlayer = evidence.players.find(({ playerId }) => playerId === 2);

    expect(evidence.preWinA).toBeCloseTo(predictDoubles([1, 2], [3, 4], states, config), 15);
    expect(evidence.players.map(({ playerId }) => playerId)).toEqual([1, 2, 3, 4]);
    expect(playerOne).toMatchObject({
      before: states[1],
      opponentX:
        displayRatingToInternalX(states[3]!.r) +
        displayRatingToInternalX(states[4]!.r) -
        displayRatingToInternalX(config.initialRating),
      opponentPhi: Math.hypot(
        displayRdToInternalPhi(states[3]!.rd),
        displayRdToInternalPhi(states[4]!.rd),
        displayRdToInternalPhi(config.initialRd)
      ),
      result: 1,
    });
    expect(temporaryPlayer).toMatchObject({
      before: {
        r: config.initialRating,
        rd: config.initialRd,
        volatility: config.initialVolatility,
      },
      result: 1,
    });
    expect(evidence.players.filter(({ result }) => result === 0)).toHaveLength(2);
    expect(states).toEqual(before);
    expect(states[2]).toBeUndefined();
  });

  it("updates all four players from one snapshot with the un-clamped virtual RD", () => {
    const states = Object.freeze({
      1: Object.freeze({ r: 1250, rd: 250, volatility: 0.04 }),
      2: Object.freeze({ r: 1000, rd: 250, volatility: 0.05 }),
      3: Object.freeze({ r: 1100, rd: 250, volatility: 0.06 }),
      4: Object.freeze({ r: 900, rd: 250, volatility: 0.07 }),
    });
    const before = structuredClone(states);
    const update = updateDoublesMatch({
      teamA: [1, 2],
      teamB: [3, 4],
      states,
      config,
      resultA: 1,
    });
    const a1 = states[1];
    const a2 = states[2];
    const b1 = states[3];
    const b2 = states[4];
    const virtualPhi = Math.hypot(
      displayRdToInternalPhi(a2.rd),
      displayRdToInternalPhi(b1.rd),
      displayRdToInternalPhi(b2.rd)
    );
    const direct = updateGlicko2(
      {
        x: displayRatingToInternalX(a1.r),
        phi: displayRdToInternalPhi(a1.rd),
        sigma: a1.volatility,
      },
      [
        {
          opponentX:
            displayRatingToInternalX(b1.r) +
            displayRatingToInternalX(b2.r) -
            displayRatingToInternalX(a2.r),
          opponentPhi: virtualPhi,
          result: 1,
        },
      ],
      config.tau
    );
    const change = update.changes.find(({ playerId }) => playerId === 1);

    expect(virtualPhi * GLICKO2_INTERNAL_SCALE).toBeGreaterThan(config.maxRd);
    expect(update.preWinA).toBeCloseTo(predictDoubles([1, 2], [3, 4], states, config), 15);
    expect(update.changes).toHaveLength(4);
    expect(change?.before).toEqual(states[1]);
    expect(change?.after.r).toBeCloseTo(1000 + GLICKO2_INTERNAL_SCALE * direct.x, 12);
    expect(change?.after.rd).toBeCloseTo(GLICKO2_INTERNAL_SCALE * direct.phi, 12);
    expect(change?.after.volatility).toBeCloseTo(direct.sigma, 15);
    expect(states).toEqual(before);
  });

  it("allows teammates with different RD to receive different non-zero-sum updates", () => {
    const states = {
      1: { r: 1000, rd: 60, volatility: 0.06 },
      2: { r: 1000, rd: 240, volatility: 0.06 },
      3: { r: 1000, rd: 100, volatility: 0.06 },
      4: { r: 1000, rd: 100, volatility: 0.06 },
    };
    const update = updateDoublesMatch({
      teamA: [1, 2],
      teamB: [3, 4],
      states,
      config,
      resultA: 1,
    });
    const a1 = update.changes.find(({ playerId }) => playerId === 1);
    const a2 = update.changes.find(({ playerId }) => playerId === 2);
    const totalDelta = update.changes.reduce(
      (total, change) => total + change.after.r - change.before.r,
      0
    );

    expect(a1?.after.r).not.toBeCloseTo(a2?.after.r ?? 0, 10);
    expect(totalDelta).not.toBeCloseTo(0, 10);
  });

  it("keeps unknown players temporary when applying a pure match update", () => {
    const states: Readonly<Record<number, RatingState | undefined>> = Object.freeze({
      1: Object.freeze({ r: 1000, rd: 100, volatility: 0.06 }),
      3: Object.freeze({ r: 1000, rd: 100, volatility: 0.06 }),
      4: Object.freeze({ r: 1000, rd: 100, volatility: 0.06 }),
    });
    const update = updateDoublesMatch({
      teamA: [1, 2],
      teamB: [3, 4],
      states,
      config,
      resultA: 0,
    });
    const temporaryPlayer = update.changes.find(({ playerId }) => playerId === 2);

    expect(temporaryPlayer?.before).toEqual({
      r: config.initialRating,
      rd: config.initialRd,
      volatility: config.initialVolatility,
    });
    expect(states[2]).toBeUndefined();
  });

  it("round-trips display states through the doubles internal coordinates without material drift", () => {
    const update = updateDoublesMatch({
      teamA: [1, 2],
      teamB: [3, 4],
      states: {
        1: { r: 1234.56789, rd: 76.54321, volatility: 0.051234 },
        2: { r: 987.65432, rd: 187.65432, volatility: 0.06789 },
        3: { r: 1111.11111, rd: 99.99999, volatility: 0.045678 },
        4: { r: 876.54321, rd: 210.98765, volatility: 0.078901 },
      },
      config,
      resultA: 1,
    });

    for (const { after } of update.changes) {
      expect(1000 + GLICKO2_INTERNAL_SCALE * displayRatingToInternalX(after.r)).toBeCloseTo(
        after.r,
        10
      );
      expect(GLICKO2_INTERNAL_SCALE * displayRdToInternalPhi(after.rd)).toBeCloseTo(after.rd, 10);
    }
  });
});
