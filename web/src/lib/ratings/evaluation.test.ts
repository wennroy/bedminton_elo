import { describe, expect, it } from "vitest";
import { computeMatchWinProbs } from "../elo";
import { replayRatings } from "./replay";
import {
  calibratePredictions,
  evaluateRatings,
  scorePredictions,
  summarizeWeeklyCorrections,
} from "./evaluation";
import { replayConfig, replayMatch, syntheticRatingScenario } from "../../../test/fixtures/ratings-scenarios";

describe("rating backtest evaluation", () => {
  it("scores Brier and log loss from hand-calculated predictions", () => {
    const score = scorePredictions([
      { matchId: 1, probability: 0.25, result: 1 },
      { matchId: 2, probability: 0.75, result: 0 },
    ]);

    expect(score.count).toBe(2);
    expect(score.brier).toBeCloseTo(0.5625, 12);
    expect(score.logLoss).toBeCloseTo(-Math.log(0.25), 12);
  });

  it("keeps log loss finite at exact zero and one probabilities", () => {
    const score = scorePredictions([
      { matchId: 1, probability: 0, result: 1 },
      { matchId: 2, probability: 1, result: 0 },
    ]);

    expect(Number.isFinite(score.logLoss)).toBe(true);
    expect(score.logLoss).toBeCloseTo(-Math.log(1e-15), 12);
  });

  it("rejects non-finite, out-of-range, and non-binary prediction inputs", () => {
    expect(() => scorePredictions([{ matchId: 1, probability: Number.NaN, result: 1 }])).toThrow(/within \[0, 1\]/);
    expect(() => scorePredictions([{ matchId: 1, probability: 1.01, result: 1 }])).toThrow(/within \[0, 1\]/);
    expect(() => scorePredictions([{ matchId: 1, probability: 0.5, result: 2 as 0 | 1 }])).toThrow(/result must be 0 or 1/);
  });

  it("calibrates fixed bins and represents empty bins safely", () => {
    const calibration = calibratePredictions(
      [
        { matchId: 1, probability: 0.1, result: 0 },
        { matchId: 2, probability: 0.2, result: 1 },
        { matchId: 3, probability: 1, result: 1 },
      ],
      [0, 0.5, 1]
    );

    expect(calibration.bins[0]).toMatchObject({ lower: 0, upper: 0.5, count: 2, observedRate: 0.5 });
    expect(calibration.bins[0].meanPrediction).toBeCloseTo(0.15, 12);
    expect(calibration.bins[1]).toEqual({ lower: 0.5, upper: 1, count: 1, meanPrediction: 1, observedRate: 1 });
    expect(calibratePredictions([], [0, 0.5, 1]).bins).toEqual([
      { lower: 0, upper: 0.5, count: 0, meanPrediction: null, observedRate: null },
      { lower: 0.5, upper: 1, count: 0, meanPrediction: null, observedRate: null },
    ]);
  });

  it("uses only T6 pre-match Estimated values for the doubles model", () => {
    const matches = [
      replayMatch(1, "2026-10-02", 21, 18, [1, 2], [3, 4], "2026-10-02T11:00:00+08:00"),
      replayMatch(2, "2026-10-03", 17, 21, [1, 3], [2, 4], "2026-10-03T11:00:00+08:00"),
    ];
    const options = { config: replayConfig, asOf: "2026-10-05T00:00:00+08:00" };
    const report = evaluateRatings(
      { source: { kind: "synthetic", inputVersion: "test", inputHash: "hash" }, playerIds: [1, 2, 3, 4], matches },
      options
    );
    const replay = replayRatings(matches, [1, 2, 3, 4], options);
    const doubles = report.models.find((model) => model.id === "glicko2-doubles-v1");

    expect(doubles?.samples.map((sample) => sample.probability)).toEqual([
      replay.matchEstimates["1"].preWinA,
      replay.matchEstimates["2"].preWinA,
    ]);
  });

  it("does not let a later result or Final correction alter an earlier prediction", () => {
    const first = replayMatch(1, "2026-09-29", 21, 18, [1, 2], [3, 4], "2026-09-29T11:00:00+08:00");
    const laterWin = replayMatch(2, "2026-10-02", 21, 18, [1, 3], [2, 4], "2026-10-02T11:00:00+08:00");
    const laterLoss = { ...laterWin, scoreA: 18, scoreB: 21 };
    const input = (matches: readonly typeof first[]) => ({
      source: { kind: "synthetic" as const, inputVersion: "test", inputHash: "hash" },
      playerIds: [1, 2, 3, 4],
      matches,
    });
    const options = { config: replayConfig, asOf: "2026-10-05T00:00:00+08:00" };
    const one = evaluateRatings(input([first, laterWin]), options);
    const two = evaluateRatings(input([first, laterLoss]), options);
    const findDoubles = (report: typeof one) =>
      report.models.find((model) => model.id === "glicko2-doubles-v1")?.samples[0].probability;

    expect(findDoubles(two)).toBe(findDoubles(one));
  });

  it("reports all three real models on one effective, future-free sample set", () => {
    const report = evaluateRatings(syntheticRatingScenario, {
      config: replayConfig,
      asOf: "2026-10-05T00:00:00+08:00",
    });

    expect(report.status).toBe("insufficient_evidence");
    expect(report.input.futureMatchCount).toBeGreaterThan(0);
    expect(report.models.map((model) => model.id)).toEqual([
      "legacy-elo-k16-v1",
      "naive-glicko2-no-partner-v1",
      "glicko2-doubles-v1",
    ]);
    expect(new Set(report.models.map((model) => model.samples.length)).size).toBe(1);
    expect(report.models[0].comparisonNote).toBeUndefined();
    expect(report.models[1].comparisonNote).toMatch(/ignores a teammate/i);
    expect(report.models[1].comparisonNote).toMatch(/omits process variance and weekly Final settlement/i);
    expect(report.models[2].samples.map((sample) => sample.matchId)).not.toContain(14);
    expect(report.input.exclusions).toEqual([
      { matchId: 14, reason: "future" },
      { matchId: 15, reason: "invalid_date" },
      { matchId: 16, reason: "invalid_score" },
      { matchId: 17, reason: "duplicate_player" },
      { matchId: 18, reason: "unknown_player" },
    ]);
    expect(JSON.stringify(report)).not.toMatch(/name/i);
  });

  it("excludes all duplicate match IDs before every model sees the same effective sample", () => {
    const first = replayMatch(1, "2026-10-02");
    const duplicateOne = replayMatch(2, "2026-10-03", 18, 21);
    const duplicateTwo = replayMatch(2, "2026-10-04", 21, 18);
    const unsafe = replayMatch(-1, "2026-10-04", 21, 18);
    const report = evaluateRatings(
      {
        source: { kind: "synthetic", inputVersion: "test", inputHash: "hash" },
        playerIds: [1, 2, 3, 4],
        matches: [first, duplicateOne, duplicateTwo, unsafe],
      },
      { config: replayConfig, asOf: "2026-10-05T00:00:00+08:00" }
    );

    expect(report.status).toBe("insufficient_evidence");
    expect(report.input.exclusions).toEqual([
      { matchId: 2, reason: "invalid_match_id" },
      { matchId: 2, reason: "invalid_match_id" },
      { matchId: -1, reason: "invalid_match_id" },
    ]);
    expect(report.models.map((model) => model.samples.map((sample) => sample.matchId))).toEqual([[1], [1], [1]]);
    expect(report.failures).toEqual([]);
  });

  it("matches the legacy public Elo pre-match probability sequence", () => {
    const report = evaluateRatings(syntheticRatingScenario, {
      config: replayConfig,
      asOf: "2026-10-05T00:00:00+08:00",
    });
    const effective = syntheticRatingScenario.matches.slice(0, 13);
    const expected = computeMatchWinProbs(
      effective.map((match) => ({
        date: match.playedAt,
        a1: String(match.teamA[0]),
        a2: String(match.teamA[1]),
        b1: String(match.teamB[0]),
        b2: String(match.teamB[1]),
        scoreA: match.scoreA,
        scoreB: match.scoreB,
      }))
    );

    expect(report.models[0].samples.map((sample) => sample.probability)).toEqual(expected);
  });

  it("summarizes weekly Final corrections without treating match estimates as corrections", () => {
    const replay = replayRatings(
      [replayMatch(1, "2026-09-29"), replayMatch(2, "2026-10-02")],
      [1, 2, 3, 4],
      { config: replayConfig, asOf: "2026-10-05T00:00:00+08:00" }
    );
    const summary = summarizeWeeklyCorrections(replay.events);

    expect(summary.segmentCount).toBeGreaterThan(0);
    expect(summary.playerCorrectionCount).toBeGreaterThan(0);
    expect(Number.isFinite(summary.meanAbsoluteCorrection)).toBe(true);
  });

  it("reports closed Final correction distributions for short segments across the season boundary", () => {
    const report = evaluateRatings(syntheticRatingScenario, {
      config: replayConfig,
      asOf: "2026-10-05T00:00:00+08:00",
    });

    expect(report.weeklyFinal.segments).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ h: 3 / 7, playerCorrectionCount: 6 }),
        expect.objectContaining({ h: 4 / 7, playerCorrectionCount: 6 }),
      ])
    );
    for (const segment of report.weeklyFinal.segments) {
      expect(Number.isFinite(segment.p50AbsoluteCorrection)).toBe(true);
      expect(Number.isFinite(segment.p90AbsoluteCorrection)).toBe(true);
    }
  });
});
