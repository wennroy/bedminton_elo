import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb } from "@/lib/db";
import { addPlayer, addMatch, recomputeAllRatings } from "@/lib/repo";
import { predictElo } from "@/lib/elo";
import { POST } from "./route";

function createRequest(body: object): Request {
  return new Request("http://localhost/api/schedule", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe.sequential("schedule API", () => {
  let dbPath: string;

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-schedule-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
  });

  afterEach(() => {
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
  });

  it("returns winRate computed from ELO ratings", async () => {
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    // 让 ELO 拉开差距:甲/乙 胜 丙/丁
    addMatch({
      pa1: p1,
      pa2: p2,
      pb1: p3,
      pb2: p4,
      scoreA: 21,
      scoreB: 10,
      playedAt: "2026-09-01",
    });

    const res = await POST(
      createRequest({ playerIds: [p1, p2, p3, p4], matches: 3, seed: 42 })
    );
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.schedule).toHaveLength(3);

    const ratings = recomputeAllRatings();
    const eloRatings: Record<string, number> = Object.fromEntries(
      [...ratings].map(([id, r]) => [String(id), r.elo])
    );

    for (const match of data.schedule) {
      const expected = predictElo(
        match.a1,
        match.a2,
        match.b1,
        match.b2,
        eloRatings
      ).teamAWin;
      expect(match.winRate).toBeCloseTo(expected, 10);
    }
    // ELO 已拉开,至少一场的胜率应显著偏离 50%
    expect(
      data.schedule.some(
        (m: { winRate: number }) => Math.abs(m.winRate - 0.5) > 0.01
      )
    ).toBe(true);
  });

  it("rejects invalid payloads", async () => {
    for (const body of [
      {},
      { playerIds: [1, 2, 3], matches: 4 },
      { playerIds: [1, 1, 2, 3], matches: 4 },
      { playerIds: [1, 2, 3, 4], matches: 0 },
    ]) {
      const res = await POST(createRequest(body));
      expect(res.status).toBe(400);
    }
  });

  it("rejects unknown player ids", async () => {
    const res = await POST(
      createRequest({ playerIds: [9991, 9992, 9993, 9994], matches: 2 })
    );
    expect(res.status).toBe(400);
  });
});
