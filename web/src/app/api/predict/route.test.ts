import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  vi,
  type MockInstance,
} from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addPlayer, addMatch, recomputeAllRatings } from "@/lib/repo";
import { predictElo, predictEloDeltas } from "@/lib/elo";
import { initializeRatingConfig } from "@/lib/rating-config";
import { loadGlickoSnapshot } from "@/lib/rating-service";
import { readRatingConfig } from "@/lib/rating-config";
import { predictDoubles } from "@/lib/ratings/doubles";
import * as replayModule from "@/lib/ratings/replay";
import { POST as POST_SCHEDULE } from "@/app/api/schedule/route";
import { POST } from "./route";

const FIRST_SEASON_START = "2026-07-01";

function createRequest(body: object): Request {
  return new Request("http://localhost/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe.sequential("predict API", () => {
  let dbPath: string;

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-predict-${Date.now()}.db`);
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

  it("legacy：胜率与逐人赢/输变化同 predictElo/predictEloDeltas 逐位一致", async () => {
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
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
      createRequest({ pa1: p1, pa2: p2, pb1: p3, pb2: p4 })
    );
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.model).toBe("legacy");
    expect(data.freshness).toBe("fresh");
    expect(data.version).toBe("legacy");

    const ratings = recomputeAllRatings();
    const eloRatings: Record<string, number> = Object.fromEntries(
      [...ratings].map(([id, r]) => [String(id), r.elo])
    );
    const expectedWin = predictElo(
      String(p1),
      String(p2),
      String(p3),
      String(p4),
      eloRatings
    ).teamAWin;
    const expectedDeltas = predictEloDeltas(
      String(p1),
      String(p2),
      String(p3),
      String(p4),
      eloRatings
    );
    expect(data.preWinA).toBe(expectedWin);
    expect(data.players).toHaveLength(4);
    for (const player of data.players) {
      expect(player.win).toBe(expectedDeltas[String(player.playerId)].win);
      expect(player.loss).toBe(expectedDeltas[String(player.playerId)].loss);
    }
  });

  it("rejects invalid payloads", async () => {
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    for (const body of [
      {},
      { pa1: p1, pa2: p2, pb1: p3 },
      { pa1: p1, pa2: p1, pb1: p3, pb2: p4 },
      { pa1: "x", pa2: p2, pb1: p3, pb2: p4 },
    ]) {
      const res = await POST(createRequest(body));
      expect(res.status).toBe(400);
    }
  });

  it("rejects unknown player ids", async () => {
    const res = await POST(
      createRequest({ pa1: 9991, pa2: 9992, pb1: 9993, pb2: 9994 })
    );
    expect(res.status).toBe(400);
  });

  it("glicko2 fresh：胜率与赢/输模拟同快照的 predictDoubles/estimateNextMatch 一致", async () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    // 制造评分差距，使胜率显著偏离 50%。
    addMatch({
      pa1: p1,
      pa2: p2,
      pb1: p3,
      pb2: p4,
      scoreA: 21,
      scoreB: 10,
      playedAt: "2026-09-14",
    });
    addMatch({
      pa1: p1,
      pa2: p2,
      pb1: p3,
      pb2: p4,
      scoreA: 21,
      scoreB: 12,
      playedAt: "2026-09-15",
    });

    const res = await POST(
      createRequest({
        pa1: p1,
        pa2: p2,
        pb1: p3,
        pb2: p4,
        rating: "glicko2",
      })
    );
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.model).toBe("glicko2");
    expect(data.freshness).toBe("fresh");
    expect(data.version).toBeTypeOf("string");
    expect(data.inputHash).toBeTypeOf("string");
    expect(data.segmentId).toBeTypeOf("string");
    expect(data.players).toHaveLength(4);
    expect(Math.abs(data.preWinA - 0.5)).toBeGreaterThan(0.01);

    // 同一 asOf 重取快照（memo 命中）：API 概率 = predictDoubles(replay.current)。
    const snapshot = loadGlickoSnapshot(getDb(), data.asOf);
    if (snapshot.state !== "ready") throw new Error("expected ready");
    const config = readRatingConfig(getDb())!.config;
    const expectedWin = predictDoubles(
      [p1, p2],
      [p3, p4],
      snapshot.replay.current,
      config
    );
    expect(data.preWinA).toBe(expectedWin);

    const ids = [p1, p2, p3, p4];
    for (const player of data.players) {
      expect(ids).toContain(player.playerId);
      for (const outcome of [player.win, player.loss]) {
        expect(outcome.playerId).toBe(player.playerId);
        expect(outcome.delta).toBeCloseTo(
          outcome.after.r - outcome.before.r,
          10
        );
        expect(outcome.before.rd).toBeTypeOf("number");
        expect(outcome.after.volatility).toBeTypeOf("number");
      }
    }
    // 赢/输按「该球员所在队」视角（与 legacy 分支同约）：四人的 win 都涨、
    // loss 都跌——Glicko-2 里得分 1 恒大于期望，赢方必涨、输方必跌。
    interface Outcome {
      delta: number;
    }
    interface PlayerPrediction {
      playerId: number;
      win: Outcome;
      loss: Outcome;
    }
    const byId = new Map<number, PlayerPrediction>(
      data.players.map((p: PlayerPrediction) => [p.playerId, p])
    );
    for (const id of [p1, p2, p3, p4]) {
      expect(byId.get(id)!.win.delta).toBeGreaterThan(0);
      expect(byId.get(id)!.loss.delta).toBeLessThan(0);
    }
  });

  it("rating 参数与非法值回退 activeModel（无配置默认 legacy）", async () => {
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    const resInvalid = await POST(
      createRequest({
        pa1: p1,
        pa2: p2,
        pb1: p3,
        pb2: p4,
        rating: "trueskill",
      })
    );
    expect(resInvalid.status).toBe(200);
    expect((await resInvalid.json()).model).toBe("legacy");

    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    const resAlias = await POST(
      createRequest({
        pa1: p1,
        pa2: p2,
        pb1: p3,
        pb2: p4,
        model: "glicko2",
      })
    );
    expect(resAlias.status).toBe(200);
    expect((await resAlias.json()).model).toBe("glicko2");
  });

  it("glicko2 服务 unavailable 拒答 409，不伪造胜率", async () => {
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    const res = await POST(
      createRequest({
        pa1: p1,
        pa2: p2,
        pb1: p3,
        pb2: p4,
        rating: "glicko2",
      })
    );
    expect(res.status).toBe(409);
    const data = await res.json();
    expect(data.state).toBe("unavailable");
    expect(data.reason).toBeTypeOf("string");
    expect(data.preWinA).toBeUndefined();
  });

  it("glicko2 服务 stale 拒答 409 并给出旧成功时点", async () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    addMatch({
      pa1: p1,
      pa2: p2,
      pb1: p3,
      pb2: p4,
      scoreA: 21,
      scoreB: 10,
      playedAt: "2026-09-15",
    });
    const ok = await POST(
      createRequest({
        pa1: p1,
        pa2: p2,
        pb1: p3,
        pb2: p4,
        rating: "glicko2",
      })
    );
    expect(ok.status).toBe(200);

    // 改输入 + 注入计算失败 → stale。
    getDb().prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    const spy: MockInstance<typeof replayModule.replayRatings> = vi
      .spyOn(replayModule, "replayRatings")
      .mockImplementation(() => {
        throw new RangeError("numeric failure");
      });
    try {
      const res = await POST(
        createRequest({
          pa1: p1,
          pa2: p2,
          pb1: p3,
          pb2: p4,
          rating: "glicko2",
        })
      );
      expect(res.status).toBe(409);
      const data = await res.json();
      expect(data.state).toBe("stale");
      expect(data.reason).toContain("last good as of");
    } finally {
      spy.mockRestore();
    }
  });

  it("与 /api/schedule 同阵容同快照概率一致", async () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    addMatch({
      pa1: p1,
      pa2: p2,
      pb1: p3,
      pb2: p4,
      scoreA: 21,
      scoreB: 10,
      playedAt: "2026-09-14",
    });

    const predictRes = await POST(
      createRequest({
        pa1: p1,
        pa2: p2,
        pb1: p3,
        pb2: p4,
        rating: "glicko2",
      })
    );
    expect(predictRes.status).toBe(200);
    const predictData = await predictRes.json();
    expect(Math.abs(predictData.preWinA - 0.5)).toBeGreaterThan(0.01);

    // /api/schedule 同快照输出的每个阵容，/api/predict 给同一胜率。
    const scheduleRes = await POST_SCHEDULE(
      new Request("http://localhost/api/schedule", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          playerIds: [p1, p2, p3, p4, addPlayer("戊"), addPlayer("己")],
          matches: 3,
          seed: 42,
          model: "glicko2",
        }),
      })
    );
    expect(scheduleRes.status).toBe(200);
    const scheduleData = await scheduleRes.json();
    expect(scheduleData.schedule.length).toBeGreaterThan(0);
    for (const match of scheduleData.schedule) {
      const res = await POST(
        createRequest({
          pa1: Number(match.a1),
          pa2: Number(match.a2),
          pb1: Number(match.b1),
          pb2: Number(match.b2),
          rating: "glicko2",
        })
      );
      expect(res.status).toBe(200);
      const data = await res.json();
      // 同一 asOf 工作状态：predict preWinA = schedule winRate（同一 predictDoubles）。
      expect(data.preWinA).toBe(match.winRate);
    }
  });
});
