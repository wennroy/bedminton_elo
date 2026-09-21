import { describe, expect, it } from "vitest";
import {
  GLICKO2_INTERNAL_SCALE,
  createRatingConfig,
  displayRdToInternalPhi,
} from "./config";
import { predictDoubles } from "./doubles";
import { updateGlicko2 } from "./glicko2-core";
import { estimateNextMatch, finalizeSegment, prepareEstimated } from "./segment";
import type { RatingMatch, RatingSegment, RatingState } from "./types";

const config = createRatingConfig({ firstSeasonStart: "2024-01-01" });

const wholeWeek: RatingSegment = {
  id: "2026-09-28:2026-09-28",
  weekStart: "2026-09-28",
  seasonId: null,
  start: "2026-09-27T16:00:00Z",
  end: "2026-10-04T16:00:00Z",
  h: 1,
};

function match(
  id: number,
  scoreA: number,
  scoreB: number,
  teamA: readonly [number, number] = [1, 2],
  teamB: readonly [number, number] = [3, 4]
): RatingMatch {
  return {
    id,
    playedAt: "2026-09-29",
    createdAt: `2026-09-29T${String(id).padStart(2, "0")}:00:00+08:00`,
    teamA,
    teamB,
    scoreA,
    scoreB,
  };
}

function states(overrides: Partial<Record<number, RatingState>> = {}): Record<number, RatingState> {
  return {
    1: { r: 1040, rd: 80, volatility: 0.06 },
    2: { r: 980, rd: 120, volatility: 0.05 },
    3: { r: 1010, rd: 100, volatility: 0.07 },
    4: { r: 970, rd: 160, volatility: 0.04 },
    ...overrides,
  };
}

describe("single Glicko-2 rating segment", () => {
  it("prepares every existing state exactly once without changing the supplied baseline", () => {
    const base = Object.freeze({
      1: Object.freeze({ r: 1050, rd: 120, volatility: 0.08 }),
      2: Object.freeze({ r: 950, rd: 180, volatility: 0.06 }),
    });
    const before = structuredClone(base);
    const segment = { ...wholeWeek, h: 3 / 7 };

    const prepared = prepareEstimated(base, segment, config);
    const expectedRd =
      GLICKO2_INTERNAL_SCALE *
      Math.hypot(displayRdToInternalPhi(base[1].rd), Math.sqrt(segment.h) * base[1].volatility);

    expect(prepared[1]).toEqual({
      r: base[1].r,
      rd: expectedRd,
      volatility: base[1].volatility,
    });
    expect(prepareEstimated(base, segment, config)).toEqual(prepared);
    expect(base).toEqual(before);
  });

  it("chains Estimated matches from the returned working snapshot while freezing volatility", () => {
    const prepared = prepareEstimated(states(), wholeWeek, config);
    const first = estimateNextMatch(match(1, 21, 17), prepared, wholeWeek, config);
    const secondMatch = match(2, 18, 21, [1, 3], [2, 4]);
    const second = estimateNextMatch(secondMatch, first.states, wholeWeek, config);

    expect(second.preWinA).toBeCloseTo(
      predictDoubles(secondMatch.teamA, secondMatch.teamB, first.states, config),
      15
    );
    for (const change of [...first.changes, ...second.changes]) {
      expect(change.after.volatility).toBe(change.before.volatility);
    }
    for (const playerId of [1, 3]) {
      const firstAfter = first.changes.find((change) => change.playerId === playerId)?.after;
      const secondBefore = second.changes.find((change) => change.playerId === playerId)?.before;
      expect(secondBefore).toEqual(firstAfter);
    }
  });

  it("initializes and prepares a mid-segment newcomer only for their first real match", () => {
    const prepared = prepareEstimated({ 1: states()[1], 3: states()[3], 4: states()[4] }, wholeWeek, config);
    const first = estimateNextMatch(match(1, 21, 17), prepared, wholeWeek, config);
    const second = estimateNextMatch(match(2, 21, 19), first.states, wholeWeek, config);
    const newcomerFirst = first.changes.find((change) => change.playerId === 2);
    const newcomerSecond = second.changes.find((change) => change.playerId === 2);
    const preparedInitialRd =
      GLICKO2_INTERNAL_SCALE *
      Math.hypot(displayRdToInternalPhi(config.initialRd), Math.sqrt(wholeWeek.h) * config.initialVolatility);

    expect(newcomerFirst?.before).toEqual({
      r: config.initialRating,
      rd: preparedInitialRd,
      volatility: config.initialVolatility,
    });
    expect(newcomerSecond?.before).toEqual(newcomerFirst?.after);
    expect(first.states[2]).toEqual(newcomerFirst?.after);
  });

  it("rejects tied or non-finite match scores before producing an Estimated event", () => {
    const prepared = prepareEstimated(states(), wholeWeek, config);

    expect(() => estimateNextMatch(match(1, 21, 21), prepared, wholeWeek, config)).toThrow(
      /scoreA and scoreB must be non-negative safe integers and unequal/
    );
    expect(() => estimateNextMatch(match(2, Number.NaN, 20), prepared, wholeWeek, config)).toThrow(
      /scoreA and scoreB must be non-negative safe integers and unequal/
    );
  });

  it("rejects unsafe scores and match IDs without changing either caller snapshot", () => {
    const working = Object.freeze(prepareEstimated(states(), wholeWeek, config));
    const base = Object.freeze(states());
    const workingBefore = structuredClone(working);
    const baseBefore = structuredClone(base);

    expect(() => estimateNextMatch(match(1, -1, 21), working, wholeWeek, config)).toThrow(
      /scoreA and scoreB must be non-negative safe integers and unequal/
    );
    expect(() => estimateNextMatch(match(1, 21.5, 21), working, wholeWeek, config)).toThrow(
      /scoreA and scoreB must be non-negative safe integers and unequal/
    );
    expect(() => estimateNextMatch(match(1.5, 21, 19), working, wholeWeek, config)).toThrow(
      /match.id must be a non-negative safe integer/
    );
    expect(() => estimateNextMatch(match(Number.NaN, 21, 19), working, wholeWeek, config)).toThrow(
      /match.id must be a non-negative safe integer/
    );
    expect(() =>
      finalizeSegment(base, [match(1, 21, 19), match(2, 17.5, 21)], wholeWeek, config)
    ).toThrow(/scoreA and scoreB must be non-negative safe integers and unequal/);
    expect(() => finalizeSegment(base, [match(1, 21, 19), match(1, 17, 21)], wholeWeek, config)).toThrow(
      /segmentMatches must not contain duplicate match.id values/
    );
    expect(working).toEqual(workingBefore);
    expect(base).toEqual(baseBefore);
  });

  it("rejects an empty segment ID and copies the segment into a Final event", () => {
    const invalidSegment = { ...wholeWeek, id: "" };
    const segment = { ...wholeWeek, id: "segment-before-output" };

    expect(() => prepareEstimated(states(), invalidSegment, config)).toThrow(/segment.id must be a non-empty string/);
    const final = finalizeSegment(states(), [], segment, config);
    segment.id = "mutated-after-output";
    segment.h = 3 / 7;

    expect(final.segment).toEqual({ ...wholeWeek, id: "segment-before-output" });
  });

  it("uses the short-segment volatility conversion for Final, including an empty period", () => {
    const segment = { ...wholeWeek, h: 3 / 7 };
    const base = {
      1: { r: 1020, rd: 150, volatility: 0.08 },
    };
    const final = finalizeSegment(base, [], segment, config);
    const expectedInternal = updateGlicko2(
      {
        x: (base[1].r - 1000) / GLICKO2_INTERNAL_SCALE,
        phi: displayRdToInternalPhi(base[1].rd),
        sigma: base[1].volatility * Math.sqrt(segment.h),
      },
      [],
      config.tau
    );

    expect(final.start).toEqual(base);
    expect(final.estimatedEnd[1]).toEqual(prepareEstimated(base, segment, config)[1]);
    expect(final.final[1]).toMatchObject({
      r: base[1].r,
      rd: GLICKO2_INTERNAL_SCALE * expectedInternal.phi,
      volatility: expectedInternal.sigma / Math.sqrt(segment.h),
    });
    expect(final.correction[1]).toBe(0);
    expect(finalizeSegment({}, [], segment, config)).toMatchObject({
      start: {},
      estimatedEnd: {},
      final: {},
      correction: {},
    });
  });

  it("clamps Estimated preparation and Final output RD for both quarterly split lengths", () => {
    const wideVolatility = { 1: { r: 1000, rd: 250, volatility: 0.5 } };
    const prepared = prepareEstimated(wideVolatility, wholeWeek, config);
    const first = finalizeSegment(wideVolatility, [], { ...wholeWeek, h: 3 / 7 }, config);
    const second = finalizeSegment(wideVolatility, [], { ...wholeWeek, h: 4 / 7 }, config);

    expect(prepared[1].rd).toBe(config.maxRd);
    expect(first.final[1].rd).toBe(config.maxRd);
    expect(second.final[1].rd).toBe(config.maxRd);
    expect(first.final[1].volatility).toBeCloseTo(wideVolatility[1].volatility, 12);
    expect(second.final[1].volatility).toBeCloseTo(wideVolatility[1].volatility, 12);
  });

  it("uses one stable baseline for Final regardless of match array ordering", () => {
    const base = states();
    const matches = [match(20, 21, 19), match(10, 14, 21, [1, 3], [2, 4])];
    const inOrder = finalizeSegment(base, matches, wholeWeek, config);
    const reversed = finalizeSegment(base, [...matches].reverse(), wholeWeek, config);

    expect(reversed.start).toEqual(inOrder.start);
    expect(reversed.final).toEqual(inOrder.final);
  });

  it("is idempotent, reconciles all Estimated rating movement, and retains distinct RD responses", () => {
    const base = states({
      1: { r: 1000, rd: 60, volatility: 0.06 },
      2: { r: 1000, rd: 240, volatility: 0.06 },
    });
    const matches = [match(2, 21, 14), match(1, 18, 21, [1, 3], [2, 4])];
    const first = finalizeSegment(base, matches, wholeWeek, config);
    const repeated = finalizeSegment(base, matches, wholeWeek, config);

    expect(repeated).toEqual(first);
    for (const playerId of Object.keys(first.final)) {
      const id = Number(playerId);
      expect(
        first.start[id].r +
          (first.estimatedEnd[id].r - first.start[id].r) +
          first.correction[id]
      ).toBeCloseTo(first.final[id].r, 12);
      expect(first.final[id].rd).toBeGreaterThanOrEqual(config.minRd);
      expect(first.final[id].rd).toBeLessThanOrEqual(config.maxRd);
    }
    expect(first.final[1].r).not.toBeCloseTo(first.final[2].r, 10);
  });

  it("does not mutate its baseline when a later Final record fails validation", () => {
    const base = Object.freeze(states());
    const before = structuredClone(base);

    expect(() => finalizeSegment(base, [match(1, 21, 19), match(2, 18, 18)], wholeWeek, config)).toThrow(
      /scoreA and scoreB must be non-negative safe integers and unequal/
    );
    expect(base).toEqual(before);
  });
});
