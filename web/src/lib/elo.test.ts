import { describe, expect, it } from "vitest";
import {
  recomputeElos,
  predictElo,
  predictEloDeltas,
  computeMatchWinProbs,
  computeMatchEloDeltas,
  INITIAL_RATING,
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

  describe("predictEloDeltas", () => {
    it("matches hand computation when all players are unknown", () => {
      // 全员 1000:个人 expected=0.5 -> win=+8, loss=-8
      const predicted = predictEloDeltas("1", "2", "3", "4", {});
      for (const pid of ["1", "2", "3", "4"]) {
        expect(predicted[pid].win).toBeCloseTo(8, 10);
        expect(predicted[pid].loss).toBeCloseTo(-8, 10);
      }
    });

    it("uses per-player rating vs opponent team average, not the team-average line", () => {
      // 关键口径:个人 ELO vs 对方队均分。1 号 1200 分,与 2 号(1000)同队,
      // 队均 1100;但 1 号的 expected 按自己与对方队均 1000 算,不按队均。
      const ratings = { "1": 1200, "2": 1000, "3": 1000, "4": 1000 };
      const predicted = predictEloDeltas("1", "2", "3", "4", ratings);
      const expected1 = 1 / (1 + 10 ** ((1000 - 1200) / 400));
      expect(predicted["1"].win).toBeCloseTo(16 * (1 - expected1), 10);
      expect(predicted["1"].loss).toBeCloseTo(-16 * expected1, 10);
      // 2 号个人分等于对方队均分 -> expected=0.5
      expect(predicted["2"].win).toBeCloseTo(8, 10);
      expect(predicted["2"].loss).toBeCloseTo(-8, 10);
      // 3/4 号在 B 队,对方队均 1100
      const expected3 = 1 / (1 + 10 ** ((1100 - 1000) / 400));
      expect(predicted["3"].win).toBeCloseTo(16 * (1 - expected3), 10);
      expect(predicted["3"].loss).toBeCloseTo(-16 * expected3, 10);
    });

    it("agrees with computeMatchEloDeltas on the replayed state before each match", () => {
      const replayed = computeMatchEloDeltas(matches);
      for (let i = 0; i < matches.length; i++) {
        const { ratings } = recomputeElos(matches.slice(0, i));
        const m = matches[i];
        const predicted = predictEloDeltas(m.a1, m.a2, m.b1, m.b2, ratings);
        const aWins = m.scoreA > m.scoreB;
        for (const pid of [m.a1, m.a2]) {
          const expected = aWins ? predicted[pid].win : predicted[pid].loss;
          expect(replayed[i][pid]).toBeCloseTo(expected, 10);
        }
        for (const pid of [m.b1, m.b2]) {
          const expected = aWins ? predicted[pid].loss : predicted[pid].win;
          expect(replayed[i][pid]).toBeCloseTo(expected, 10);
        }
      }
    });
  });

  describe("computeMatchEloDeltas", () => {
    it("returns one delta record per match in input order", () => {
      const deltas = computeMatchEloDeltas(matches);
      expect(deltas).toHaveLength(matches.length);
      for (let i = 0; i < matches.length; i++) {
        const m = matches[i];
        for (const pid of [m.a1, m.a2, m.b1, m.b2]) {
          expect(deltas[i][pid]).toBeTypeOf("number");
        }
      }
    });

    it("per-player accumulated deltas match recomputeElos finals", () => {
      const deltas = computeMatchEloDeltas(matches);
      const { ratings } = recomputeElos(matches);
      const sums = new Map<string, number>();
      for (const record of deltas) {
        for (const [pid, delta] of Object.entries(record)) {
          sums.set(pid, (sums.get(pid) ?? 0) + delta);
        }
      }
      for (const [pid, final] of Object.entries(ratings)) {
        expect(INITIAL_RATING + (sums.get(pid) ?? 0)).toBeCloseTo(final, 6);
      }
    });

    it("four-player delta sum is ~0 per match (near zero-sum)", () => {
      // 全员同分起步时严格零和；队员评分与队均分的偏差会带来微小非零和
      const two: Match[] = [
        { date: "2024-01-01", a1: "1", a2: "2", b1: "3", b2: "4", scoreA: 21, scoreB: 18 },
        { date: "2024-01-01", a1: "1", a2: "2", b1: "3", b2: "4", scoreA: 21, scoreB: 19 },
      ];
      for (const record of computeMatchEloDeltas(two)) {
        const sum = Object.values(record).reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(0, 10);
      }
      for (const record of computeMatchEloDeltas(matches)) {
        const sum = Object.values(record).reduce((a, b) => a + b, 0);
        expect(Math.abs(sum)).toBeLessThan(2);
      }
    });
  });
});
