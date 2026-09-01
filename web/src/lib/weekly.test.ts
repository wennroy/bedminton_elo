import { describe, expect, it } from "vitest";
import { getWeekRange, computeWeeklyStats, type MatchWithNames } from "./weekly";

const players = [
  { id: 1, name: "Alice" },
  { id: 2, name: "Bob" },
  { id: 3, name: "Carol" },
  { id: 4, name: "Dave" },
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
  });
});
