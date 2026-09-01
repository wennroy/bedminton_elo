import { describe, expect, it } from "vitest";
import {
  headToHead,
  playerFunStats,
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

// 趣味数据 fixture:Alice(1) 的 6 场,搭档 Bob 2胜1负、Eve 3胜0负;
// 对 Carol 5胜1负、对 Dave 3胜1负;末三场连胜
const funPlayers = [
  ...players,
  { id: 5, name: "Eve", createdAt: "2024-01-01T00:00:00Z" },
];

function funMatch(
  id: number,
  pa1: number,
  pa2: number,
  pb1: number,
  pb2: number,
  scoreA: number,
  scoreB: number,
  playedAt: string
) {
  const names = new Map(funPlayers.map((p) => [p.id, p.name]));
  return {
    id,
    pa1,
    pa2,
    pb1,
    pb2,
    scoreA,
    scoreB,
    playedAt,
    enteredBy: null,
    createdAt: `${playedAt}T10:00:00Z`,
    pa1Name: names.get(pa1)!,
    pa2Name: names.get(pa2)!,
    pb1Name: names.get(pb1)!,
    pb2Name: names.get(pb2)!,
  };
}

const funData: StatsData = {
  players: funPlayers,
  matches: [
    funMatch(1, 1, 2, 3, 4, 21, 18, "2024-01-01"),
    funMatch(2, 1, 2, 3, 4, 21, 19, "2024-01-02"),
    funMatch(3, 1, 2, 3, 4, 15, 21, "2024-01-03"),
    funMatch(4, 1, 5, 3, 4, 21, 20, "2024-01-04"),
    funMatch(5, 1, 5, 2, 3, 21, 10, "2024-01-05"),
    funMatch(6, 1, 5, 2, 3, 21, 15, "2024-01-06"),
  ],
  ratings: new Map([[1, { elo: 1030, mu: 25, sigma: 8 }]]),
  eloHistory: [
    { date: "2024-01-02", playerId: "1", playerName: "Alice", elo: 1050 },
    { date: "2024-01-05", playerId: "1", playerName: "Alice", elo: 1080 },
    { date: "2024-01-06", playerId: "1", playerName: "Alice", elo: 1080 },
  ],
  tsPlayers: {},
};

describe("playerFunStats", () => {
  it("computes current and longest win streaks", () => {
    const s = playerFunStats(1, funData);
    expect(s.currentStreakType).toBe("win");
    expect(s.currentStreak).toBe(3);
    expect(s.longestWinStreak).toBe(3);
  });

  it("picks best partner by win rate with >=3 games", () => {
    const s = playerFunStats(1, funData);
    expect(s.bestPartner).toMatchObject({
      id: 5,
      name: "Eve",
      wins: 3,
      total: 3,
      winRate: 100,
    });
  });

  it("picks nemesis as lowest win-rate opponent with >=3 games", () => {
    const s = playerFunStats(1, funData);
    // Dave: 3胜1负=75%,Carol: 5胜1负=83%,Bob 只交手 2 场不计
    expect(s.nemesis).toMatchObject({
      id: 4,
      name: "Dave",
      wins: 3,
      losses: 1,
      total: 4,
      winRate: 75,
    });
  });

  it("reports peak ELO with first date reached", () => {
    const s = playerFunStats(1, funData);
    expect(s.peakElo).toBe(1080);
    expect(s.peakEloDate).toBe("2024-01-05");
  });

  it("averages point difference to 1 decimal", () => {
    // +3, +2, -6, +1, +11, +6 → 17/6 = 2.833… → 2.8
    expect(playerFunStats(1, funData).avgPointDiff).toBe(2.8);
  });

  it("returns nulls and none when games are insufficient", () => {
    const s = playerFunStats(1, data); // 老 fixture:搭档最多 2 场
    expect(s.bestPartner).toBeNull();
    expect(s.nemesis).toMatchObject({ id: 4, total: 3 }); // Dave 交手 3 场
    expect(s.peakElo).toBe(1000); // 无 eloHistory/ratings → 初始分
    expect(s.peakEloDate).toBeNull();
    expect(s.currentStreak).toBe(1);
    expect(s.currentStreakType).toBe("win");
  });

  it("handles a player with no matches", () => {
    const s = playerFunStats(999, funData);
    expect(s.currentStreakType).toBe("none");
    expect(s.bestPartner).toBeNull();
    expect(s.nemesis).toBeNull();
    expect(s.avgPointDiff).toBe(0);
    expect(s.peakElo).toBe(1000);
  });
});
