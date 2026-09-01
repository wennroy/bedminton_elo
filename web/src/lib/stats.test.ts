import { describe, expect, it } from "vitest";
import {
  headToHead,
  playerMatches,
  playerSummary,
  type StatsData,
} from "./stats";

const players = [
  { id: 1, name: "Alice", createdAt: "2024-01-01T00:00:00Z" },
  { id: 2, name: "Bob", createdAt: "2024-01-01T00:00:00Z" },
  { id: 3, name: "Carol", createdAt: "2024-01-01T00:00:00Z" },
  { id: 4, name: "Dave", createdAt: "2024-01-01T00:00:00Z" },
];

const matches = [
  {
    id: 1,
    pa1: 1,
    pa2: 2,
    pb1: 3,
    pb2: 4,
    scoreA: 21,
    scoreB: 18,
    playedAt: "2024-01-01",
    enteredBy: null,
    createdAt: "2024-01-01T10:00:00Z",
    pa1Name: "Alice",
    pa2Name: "Bob",
    pb1Name: "Carol",
    pb2Name: "Dave",
  },
  {
    id: 2,
    pa1: 1,
    pa2: 3,
    pb1: 2,
    pb2: 4,
    scoreA: 19,
    scoreB: 21,
    playedAt: "2024-01-02",
    enteredBy: null,
    createdAt: "2024-01-02T10:00:00Z",
    pa1Name: "Alice",
    pa2Name: "Carol",
    pb1Name: "Bob",
    pb2Name: "Dave",
  },
  {
    id: 3,
    pa1: 1,
    pa2: 2,
    pb1: 3,
    pb2: 4,
    scoreA: 21,
    scoreB: 15,
    playedAt: "2024-01-03",
    enteredBy: null,
    createdAt: "2024-01-03T10:00:00Z",
    pa1Name: "Alice",
    pa2Name: "Bob",
    pb1Name: "Carol",
    pb2Name: "Dave",
  },
];

const data: StatsData = {
  players,
  matches,
  ratings: new Map(),
  eloHistory: [],
  tsPlayers: {},
};

describe("stats", () => {
  it("computes head-to-head by opponent", () => {
    const h2h = headToHead(1, data);
    expect(h2h).toHaveLength(3);
    const dave = h2h.find((r) => r.opponentId === 4);
    expect(dave).toMatchObject({ opponentName: "Dave", wins: 2, losses: 1, total: 3 });
    const carol = h2h.find((r) => r.opponentId === 3);
    expect(carol).toMatchObject({ opponentName: "Carol", wins: 2, losses: 0, total: 2 });
    const bob = h2h.find((r) => r.opponentId === 2);
    expect(bob).toMatchObject({ opponentName: "Bob", wins: 0, losses: 1, total: 1 });
  });

  it("sorts head-to-head by total matches descending", () => {
    const h2h = headToHead(1, data);
    expect(h2h[0].opponentId).toBe(4);
    for (let i = 1; i < h2h.length; i++) {
      expect(h2h[i - 1].total).toBeGreaterThanOrEqual(h2h[i].total);
    }
  });

  it("lists player matches newest first", () => {
    const records = playerMatches(1, data);
    expect(records).toHaveLength(3);
    expect(records[0].date).toBe("2024-01-03");
    expect(records[0].won).toBe(true);
    expect(records[1].won).toBe(false);
  });

  it("summarizes player record", () => {
    const summary = playerSummary(1, data);
    expect(summary).toMatchObject({
      id: 1,
      name: "Alice",
      totalMatches: 3,
      wins: 2,
      losses: 1,
      winRate: 67,
    });
  });

  it("returns undefined for unknown player", () => {
    expect(playerSummary(999, data)).toBeUndefined();
  });
});
