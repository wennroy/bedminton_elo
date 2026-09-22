import { describe, it, expect, beforeEach, afterEach, vi, type MockInstance } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addPlayer, addMatch, recomputeAllRatings } from "@/lib/repo";
import { predictElo } from "@/lib/elo";
import {
  initializeRatingConfig,
  readRatingConfig,
} from "@/lib/rating-config";
import { loadGlickoSnapshot } from "@/lib/rating-service";
import { loadPredictionView } from "@/lib/rating-view";
import { predictDoubles } from "@/lib/ratings/doubles";
import { optimizeSchedule } from "@/lib/scheduler";
import * as replayModule from "@/lib/ratings/replay";
import { POST } from "./route";

const FIRST_SEASON_START = "2026-07-01";

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

  it("legacy 默认响应带显式模型元信息（null 表示无版本化配置）", async () => {
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    const res = await POST(
      createRequest({ playerIds: [p1, p2, p3, p4], matches: 2, seed: 7 })
    );
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.model).toBe("legacy");
    expect(data.configVersion).toBeNull();
    expect(data.asOf).toBeNull();
    expect(data.inputHash).toBeNull();
  });

  it("glicko2：优化回调、API 概率与预测 DTO 三者同快照一致", async () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    const [p1, p2, p3, p4, p5, p6] = ["甲", "乙", "丙", "丁", "戊", "己"].map(
      (name) => addPlayer(name)
    );
    // 制造评分差距：甲/乙 两连胜 丙/丁，戊/己未参赛（目录未知球员走引擎初值）。
    // 日期取真实当前日期之前，保证重放已生效、概率有不对称性。
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
        playerIds: [p1, p2, p3, p4, p5, p6],
        matches: 3,
        seed: 42,
        model: "glicko2",
      })
    );
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.model).toBe("glicko2");
    expect(data.configVersion).toBeTypeOf("string");
    expect(data.asOf).toBeTypeOf("string");
    expect(data.inputHash).toBeTypeOf("string");

    // 同一快照同一 asOf：服务重取（memo 命中）得到同一份 replay.current。
    const snapshot = loadGlickoSnapshot(getDb(), data.asOf);
    if (snapshot.state !== "ready") throw new Error("expected ready");
    expect(snapshot.inputHash).toBe(data.inputHash);

    const config = readRatingConfig(getDb())!.config;
    const winProbability = (match: {
      a1: string;
      a2: string;
      b1: string;
      b2: string;
    }) =>
      predictDoubles(
        [Number(match.a1), Number(match.a2)],
        [Number(match.b1), Number(match.b2)],
        snapshot.replay.current,
        config
      );

    // 优化确实由 glicko2 回调驱动：同 seed 同回调本地重放，阵容逐位一致
    // （若路由内部用 TrueSkill 优化，评分不对称下阵容必然不同）。
    const localReplay = optimizeSchedule({
      playerIds: [p1, p2, p3, p4, p5, p6].map(String),
      matches: 3,
      players: [],
      seed: 42,
      lambda: 0.5,
      winProbability,
    });
    expect(data.schedule.map((m: { a1: string; a2: string; b1: string; b2: string }) => ({
      a1: m.a1,
      a2: m.a2,
      b1: m.b1,
      b2: m.b2,
    }))).toEqual(localReplay.schedule);

    let closenessSum = 0;
    for (const match of data.schedule) {
      const expected = winProbability(match);
      // API 返回概率 = 优化回调 = predictDoubles（同一快照，逐位一致）。
      expect(match.winRate).toBe(expected);
      closenessSum += Math.abs(expected - 0.5);

      // 预测 DTO 准备数据与 API 概率一致（同一 asOf）。
      const prediction = loadPredictionView({
        pa1: Number(match.a1),
        pa2: Number(match.a2),
        pb1: Number(match.b1),
        pb2: Number(match.b2),
        rating: "glicko2",
        asOf: data.asOf,
      });
      if (prediction.model !== "glicko2" || prediction.freshness !== "fresh") {
        throw new Error("expected glicko2 fresh prediction");
      }
      expect(prediction.preWinA).toBe(match.winRate);
      expect(prediction.inputHash).toBe(data.inputHash);
    }

    // metrics.meanCloseness 由 glicko2 回调的概率计算。
    expect(data.metrics.meanCloseness).toBeCloseTo(
      closenessSum / data.schedule.length,
      10
    );
  });

  it("glicko2：rating 别名参数与非法值回退 activeModel", async () => {
    initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, getDb());
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    // 未配置 activeModel（默认 legacy）：非法 model 值回退 legacy。
    const resLegacy = await POST(
      createRequest({
        playerIds: [p1, p2, p3, p4],
        matches: 2,
        model: "trueskill",
      })
    );
    expect(resLegacy.status).toBe(200);
    expect((await resLegacy.json()).model).toBe("legacy");

    const resAlias = await POST(
      createRequest({
        playerIds: [p1, p2, p3, p4],
        matches: 2,
        rating: "glicko2",
      })
    );
    expect(resAlias.status).toBe(200);
    expect((await resAlias.json()).model).toBe("glicko2");
  });

  it("glicko2：服务 unavailable 拒答 409，不伪造胜率", async () => {
    const [p1, p2, p3, p4] = ["甲", "乙", "丙", "丁"].map((name) =>
      addPlayer(name)
    );
    // 未初始化配置 → unavailable。
    const res = await POST(
      createRequest({ playerIds: [p1, p2, p3, p4], matches: 2, model: "glicko2" })
    );
    expect(res.status).toBe(409);
    const data = await res.json();
    expect(data.state).toBe("unavailable");
    expect(data.reason).toBeTypeOf("string");
    expect(data.schedule).toBeUndefined();
  });

  it("glicko2：服务 stale 拒答 409 并给出旧成功时点", async () => {
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
    // 先成功一次（写入最近成功快照）。
    const ok = await POST(
      createRequest({ playerIds: [p1, p2, p3, p4], matches: 2, model: "glicko2" })
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
        createRequest({ playerIds: [p1, p2, p3, p4], matches: 2, model: "glicko2" })
      );
      expect(res.status).toBe(409);
      const data = await res.json();
      expect(data.state).toBe("stale");
      expect(data.reason).toContain("last good as of");
      expect(data.schedule).toBeUndefined();
    } finally {
      spy.mockRestore();
    }
  });
});
