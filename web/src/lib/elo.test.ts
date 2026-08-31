import { describe, expect, it } from "vitest";
import { recomputeElos, predictElo, type Match } from "./elo";
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
});
