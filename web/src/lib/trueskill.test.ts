import { describe, expect, it } from "vitest";
import {
  predictTeamOutcome,
  recomputeTrueSkills,
  type TrueSkillPlayer,
} from "./trueskill";
import golden from "../../test/golden/trueskill.json";
import predictGolden from "../../test/golden/predict.json";
import type { Match } from "./elo";

const matches = golden.matches as Match[];

describe("trueskill", () => {
  it("recomputes ratings aligned with golden", () => {
    const { players } = recomputeTrueSkills(matches);
    for (const [playerId, expected] of Object.entries(golden.players)) {
      expect(players[playerId].mu).toBeCloseTo(expected.mu, 6);
      expect(players[playerId].sigma).toBeCloseTo(expected.sigma, 6);
    }
  });

  it("records daily snapshots aligned with golden", () => {
    const { snapshots } = recomputeTrueSkills(matches);
    expect(snapshots).toHaveLength(golden.snapshots.length);
    for (let i = 0; i < snapshots.length; i++) {
      expect(snapshots[i].date).toBe(golden.snapshots[i].date);
      expect(snapshots[i].playerId).toBe(golden.snapshots[i].playerId);
      expect(snapshots[i].mu).toBeCloseTo(golden.snapshots[i].mu, 6);
      expect(snapshots[i].sigma).toBeCloseTo(golden.snapshots[i].sigma, 6);
    }
  });

  it("predicts team outcomes", () => {
    for (const c of predictGolden.trueskill) {
      const teamA = c.teamA as TrueSkillPlayer[];
      const teamB = c.teamB as TrueSkillPlayer[];
      const result = predictTeamOutcome(teamA, teamB);
      expect(result.win).toBeCloseTo(c.expected.win, 6);
      expect(result.draw).toBeCloseTo(c.expected.draw, 6);
      expect(result.loss).toBeCloseTo(c.expected.loss, 6);
    }
  });
});
