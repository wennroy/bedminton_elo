import { describe, expect, it } from "vitest";
import {
  getWeekRange,
  computeWeeklyStats,
  weeklyDataVersion,
  type MatchWithNames,
} from "./weekly";

const players = [
  { id: 1, name: "Alice" },
  { id: 2, name: "Bob" },
  { id: 3, name: "Carol" },
  { id: 4, name: "Dave" },
  { id: 5, name: "Eve" },
  { id: 6, name: "Frank" },
  { id: 7, name: "Grace" },
  { id: 8, name: "Hank" },
];

function match(
  id: number,
  date: string,
  a1: number,
  a2: number,
  b1: number,
  b2: number,
  scoreA: number,
  scoreB: number
): MatchWithNames {
  const nameOf = (id: number) => players.find((p) => p.id === id)?.name ?? "?";
  return {
    id,
    pa1: a1,
    pa2: a2,
    pb1: b1,
    pb2: b2,
    scoreA,
    scoreB,
    playedAt: date,
    enteredBy: null,
    createdAt: `${date}T10:00:00Z`,
    pa1Name: nameOf(a1),
    pa2Name: nameOf(a2),
    pb1Name: nameOf(b1),
    pb2Name: nameOf(b2),
  };
}

const matches = [
  match(1, "2024-01-01", 1, 2, 3, 4, 21, 18), // Mon week 1
  match(2, "2024-01-02", 1, 2, 3, 4, 21, 19), // Tue week 1
  match(3, "2024-01-03", 1, 3, 2, 4, 18, 21), // Wed week 1
  match(4, "2024-01-04", 1, 3, 2, 4, 21, 17), // Thu week 1
  match(5, "2024-01-05", 1, 2, 3, 4, 21, 16), // Fri week 1
  match(6, "2024-01-08", 1, 2, 3, 4, 21, 15), // Mon week 2
];

describe("weekly", () => {
  it("computes week range from any date", () => {
    const range = getWeekRange("2024-01-03"); // Wed
    expect(range.weekStart).toBe("2024-01-01");
    expect(range.weekEnd).toBe("2024-01-07");
    expect(range.weekNumber).toBe(1);
  });

  it("handles Sunday as end of week", () => {
    const range = getWeekRange("2024-01-07"); // Sun
    expect(range.weekStart).toBe("2024-01-01");
    expect(range.weekEnd).toBe("2024-01-07");
  });

  it("aggregates attendance", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    expect(stats.attendance).toHaveLength(4);
    const alice = stats.attendance.find((s) => s.playerId === 1);
    expect(alice?.matches).toBe(5);
  });

  it("finds win king", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    const top = stats.winKing[0];
    expect(top.playerId).toBe(1);
    expect(top.wins).toBe(4);
    expect(top.playerId).toBe(1);
  });

  it("computes elo changes", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    const alice = stats.eloChanges.find((s) => s.playerId === 1);
    expect(alice).toBeDefined();
    expect(alice!.change).not.toBe(0);
  });

  it("finds best pair with at least 3 matches", () => {
    const stats = computeWeeklyStats("2024-01-01", players, matches);
    expect(stats.bestPair).not.toBeNull();
    expect(stats.bestPair!.total).toBe(3);
    expect(stats.bestPair!.winRate).toBe(1);
    expect(stats.bestPair!.playerA).toBe("Alice");
    expect(stats.bestPair!.playerB).toBe("Bob");
  });

  it("returns empty stats for week without matches", () => {
    const stats = computeWeeklyStats("2024-02-05", players, matches);
    expect(stats.attendance).toHaveLength(0);
    expect(stats.winKing).toHaveLength(0);
    expect(stats.eloChanges).toHaveLength(0);
    expect(stats.bestPair).toBeNull();
    expect(stats.fun).toEqual({
      closest: null,
      blowout: null,
      streakKing: null,
      upset: null,
    });
  });

  describe("fun stats", () => {
    it("finds the closest match (smallest score diff)", () => {
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.closest).not.toBeNull();
      expect(stats.fun.closest!.date).toBe("2024-01-02");
      expect(stats.fun.closest!.scoreA).toBe(21);
      expect(stats.fun.closest!.scoreB).toBe(19);
      expect(stats.fun.closest!.teamA).toEqual(["Alice", "Bob"]);
      expect(stats.fun.closest!.teamB).toEqual(["Carol", "Dave"]);
    });

    it("finds the blowout match (largest score diff)", () => {
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.blowout).not.toBeNull();
      expect(stats.fun.blowout!.date).toBe("2024-01-05");
      expect(stats.fun.blowout!.scoreA).toBe(21);
      expect(stats.fun.blowout!.scoreB).toBe(16);
    });

    it("closest tie: higher winner score wins even when later", () => {
      // 22:20 is closer than 21:19 (deuce games need a 2-point margin)
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 19),
        match(2, "2024-03-05", 1, 2, 3, 4, 22, 20),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.closest!.date).toBe("2024-03-05");
      expect(stats.fun.closest!.scoreA).toBe(22);
      expect(stats.fun.closest!.scoreB).toBe(20);
    });

    it("closest full tie: earliest match wins", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 19),
        match(2, "2024-03-05", 1, 2, 3, 4, 21, 19),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.closest!.date).toBe("2024-03-04");
    });

    it("blowout tie: lower loser score wins even when later", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 30, 9), // diff 21, loser 9
        match(2, "2024-03-05", 1, 2, 3, 4, 21, 0), // diff 21, loser 0
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.blowout!.date).toBe("2024-03-05");
      expect(stats.fun.blowout!.scoreB).toBe(0);
    });

    it("blowout full tie: earliest match wins", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 5),
        match(2, "2024-03-05", 1, 2, 3, 4, 21, 5),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.blowout!.date).toBe("2024-03-04");
    });

    it("finds the streak king (longest in-week win streak)", () => {
      // main fixture: Bob wins m1, m2, m3 in a row -> streak 3; Alice max 2
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.streakKing).toEqual({
        playerId: 2,
        name: "Bob",
        streak: 3,
      });
    });

    it("streak king is null when nobody wins twice in a row", () => {
      const ms = [
        match(1, "2024-03-04", 1, 2, 3, 4, 21, 18),
        match(2, "2024-03-05", 3, 4, 1, 2, 21, 18),
      ];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.streakKing).toBeNull();
    });

    it("finds the upset in the main fixture", () => {
      // m4 (2024-01-04): Alice+Carol beat Bob+Dave.
      // Pre-match averages: A=992, B=1008 -> winnerProb = 1/(1+10^(16/400)).
      const stats = computeWeeklyStats("2024-01-01", players, matches);
      expect(stats.fun.upset).not.toBeNull();
      expect(stats.fun.upset!.date).toBe("2024-01-04");
      expect(stats.fun.upset!.teamA).toEqual(["Alice", "Carol"]);
      expect(stats.fun.upset!.teamB).toEqual(["Bob", "Dave"]);
      expect(stats.fun.upset!.winnerWinProb).toBeCloseTo(
        1 / (1 + 10 ** (16 / 400)),
        10
      );
    });

    it("upset is null when every winner was favored (>= 50%)", () => {
      // single match between all-unknown players -> winner prob exactly 0.5
      const ms = [match(1, "2024-03-04", 1, 2, 3, 4, 21, 18)];
      const stats = computeWeeklyStats("2024-03-04", players, ms);
      expect(stats.fun.upset).toBeNull();
      expect(stats.fun.streakKing).toBeNull();
      expect(stats.fun.closest!.scoreA).toBe(21);
      expect(stats.fun.blowout!.scoreA).toBe(21);
    });

    it("upset picks the lowest winner win prob within the week only", () => {
      // Disjoint groups so probabilities stay independent.
      // After h1+h2: Alice/Bob ~= 1015.63, Carol/Dave ~= 984.37.
      // w1: Grace+Hank (992) beat Eve+Frank (1008) -> prob 1/(1+10^0.04) ~= 0.4770
      // w2: Carol+Dave (984.37) beat Alice+Bob (1015.63) -> prob ~= 0.4551
      // w2 comes later but is the bigger upset.
      const ms = [
        match(1, "2024-01-01", 1, 2, 3, 4, 21, 10),
        match(2, "2024-01-02", 1, 2, 3, 4, 21, 10),
        match(3, "2024-01-03", 5, 6, 7, 8, 21, 10),
        match(4, "2024-01-08", 7, 8, 5, 6, 21, 10),
        match(5, "2024-01-09", 3, 4, 1, 2, 21, 10),
      ];
      const week1 = computeWeeklyStats("2024-01-01", players, ms);
      expect(week1.fun.upset).toBeNull(); // week-1 matches were all >= 50%

      const week2 = computeWeeklyStats("2024-01-08", players, ms);
      expect(week2.fun.upset!.date).toBe("2024-01-09");
      expect(week2.fun.upset!.teamA).toEqual(["Carol", "Dave"]);
      expect(week2.fun.upset!.teamB).toEqual(["Alice", "Bob"]);
      expect(week2.fun.upset!.winnerWinProb).toBeCloseTo(0.455129, 6);
    });
  });

  describe("weeklyDataVersion", () => {
    it("is deterministic for identical stats", () => {
      const a = computeWeeklyStats("2024-01-01", players, matches);
      const b = computeWeeklyStats("2024-01-01", players, matches);
      expect(weeklyDataVersion(a)).toBe(weeklyDataVersion(b));
    });

    it("changes when a match is added to the week", () => {
      const before = computeWeeklyStats("2024-01-01", players, matches);
      const after = computeWeeklyStats("2024-01-01", players, [
        ...matches,
        match(7, "2024-01-06", 5, 6, 7, 8, 21, 10),
      ]);
      expect(weeklyDataVersion(after)).not.toBe(weeklyDataVersion(before));
    });

    it("changes when a score changes", () => {
      const before = computeWeeklyStats("2024-01-01", players, matches);
      const modified = matches.map((m) =>
        m.id === 1 ? { ...m, scoreB: 20 } : m
      );
      const after = computeWeeklyStats("2024-01-01", players, modified);
      expect(weeklyDataVersion(after)).not.toBe(weeklyDataVersion(before));
    });

    it("changes when a player is renamed", () => {
      const before = computeWeeklyStats("2024-01-01", players, matches);
      const renamed = players.map((p) =>
        p.id === 1 ? { ...p, name: "Alicia" } : p
      );
      const after = computeWeeklyStats("2024-01-01", renamed, matches);
      expect(weeklyDataVersion(after)).not.toBe(weeklyDataVersion(before));
    });

    it("differs across weeks", () => {
      const week1 = computeWeeklyStats("2024-01-01", players, matches);
      const week2 = computeWeeklyStats("2024-01-08", players, matches);
      expect(weeklyDataVersion(week1)).not.toBe(weeklyDataVersion(week2));
    });
  });
});
