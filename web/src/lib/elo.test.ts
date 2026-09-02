import { describe, expect, it } from "vitest";
import {
  recomputeElos,
  predictElo,
  computeMatchWinProbs,
  type Match,
} from "./elo";
import golden from "../../test/golden/elo.json";
import predictGolden from "../../test/golden/predict.json";

const matches = golden.matches as Match[];

describe("elo", () => {
  it("recomputes doubles ratings aligned with golden", () => {
    const { ratings } = recomputeElos(matches);
    for (const [playerId, expected] of Object.entries(golden.ratings)) {
      expect(ratings[playerId]).toBeCloseTo(expected, 6);
    }
  });

  it("records daily snapshots aligned with golden", () => {
    const { snapshots } = recomputeElos(matches);
    expect(snapshots).toHaveLength(golden.snapshots.length);
    for (let i = 0; i < snapshots.length; i++) {
      expect(snapshots[i].date).toBe(golden.snapshots[i].date);
      expect(snapshots[i].playerId).toBe(golden.snapshots[i].playerId);
      expect(snapshots[i].elo).toBeCloseTo(golden.snapshots[i].elo, 6);
    }
  });

  it("predicts doubles win probabilities", () => {
    for (const c of predictGolden.elo) {
      const result = predictElo(c.a1, c.a2, c.b1, c.b2, c.ratings);
      expect(result.teamAWin).toBeCloseTo(c.expected.teamAWin, 6);
      expect(result.teamBWin).toBeCloseTo(c.expected.teamBWin, 6);
    }
  });

  describe("computeMatchWinProbs", () => {
    const two: Match[] = [
      { date: "2024-01-01", a1: "1", a2: "2", b1: "3", b2: "4", scoreA: 21, scoreB: 18 },
      { date: "2024-01-01", a1: "1", a2: "2", b1: "3", b2: "4", scoreA: 21, scoreB: 19 },
    ];

    it("returns one probability per match in input order", () => {
      const probs = computeMatchWinProbs(two);
      expect(probs).toHaveLength(2);
    });

    it("matches hand computation for the first two matches", () => {
      // Match 1: all players unknown -> both team averages 1000 -> pA = 0.5.
      // A wins, so a1/a2 gain 16*(1-0.5)=8 -> 1008; b1/b2 lose 8 -> 992.
      // Match 2: teamAAvg=1008, teamBAvg=992
      //   -> pA = 1/(1+10^((992-1008)/400)) = 1/(1+10^(-0.04)).
      const probs = computeMatchWinProbs(two);
      expect(probs[0]).toBeCloseTo(0.5, 10);
      expect(probs[1]).toBeCloseTo(1 / (1 + 10 ** (-16 / 400)), 10);
    });

    it("agrees with predictElo on the replayed state before each match", () => {
      const probs = computeMatchWinProbs(matches);
      for (let i = 0; i < matches.length; i++) {
        const { ratings } = recomputeElos(matches.slice(0, i));
        const m = matches[i];
        const expected = predictElo(m.a1, m.a2, m.b1, m.b2, ratings);
        expect(probs[i]).toBeCloseTo(expected.teamAWin, 10);
      }
    });
  });
});
