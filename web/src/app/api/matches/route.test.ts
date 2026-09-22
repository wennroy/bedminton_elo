import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("next/cache", () => ({
  revalidatePath: vi.fn(),
}));
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addPlayer, listPlayers, listRawMatches } from "@/lib/repo";
import * as repo from "@/lib/repo";
import * as ratingService from "@/lib/rating-service";
import { initializeRatingConfig, readRatingConfig } from "@/lib/rating-config";
import { replayRatings } from "@/lib/ratings/replay";
import {
  shanghaiLocalDateFromInstant,
  weekStart,
} from "@/lib/ratings/calendar";
import { POST } from "./route";
import { DELETE as deleteById } from "./[id]/route";

function createRequest(body: object): Request {
  return new Request("http://localhost/api/matches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function createDeleteRequest(
  id: number,
  headers?: Record<string, string>
): Request {
  return new Request(`http://localhost/api/matches/${id}`, {
    method: "DELETE",
    headers,
  });
}

function todayString(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** 本次测试运行时的上海当日（与路由评分使用同一日历口径）。 */
function shanghaiToday(): string {
  return shanghaiLocalDateFromInstant(new Date().toISOString());
}

/** 上一周的周三（必定早于当前周，用于补录场景）。 */
function previousWeekWednesday(): string {
  const monday = weekStart(shanghaiToday());
  const date = new Date(`${monday}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() - 7 + 2);
  return date.toISOString().slice(0, 10);
}

/** 用引擎直接重放当前 DB 输入，取该 matchId 的 MatchEstimate 作为基准。 */
function engineEstimateFor(matchId: number, asOf: string) {
  const db = getDb();
  const record = readRatingConfig(db);
  if (record === null) throw new Error("expected rating config in this test");
  const replay = replayRatings(
    listRawMatches(db).map((m) => ({
      id: m.id,
      playedAt: m.playedAt,
      createdAt: m.createdAt,
      teamA: [m.pa1, m.pa2] as const,
      teamB: [m.pb1, m.pb2] as const,
      scoreA: m.scoreA,
      scoreB: m.scoreB,
    })),
    listPlayers(db).map((p) => p.id),
    { config: record.config, asOf }
  );
  return {
    replay,
    estimate: replay.matchEstimates[String(matchId)],
    configVersion: replay.configVersion,
  };
}

function initGlickoConfig() {
  // 首赛季起点固定在过去季度，使当前周属于某赛季（确定性输出）。
  initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, getDb());
}

describe.sequential("matches API", () => {
  let dbPath: string;
  let players: number[] = [];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-badminton-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    process.env.ADMIN_PASSWORD = "admin-secret";
    players = [
      addPlayer("A"),
      addPlayer("B"),
      addPlayer("C"),
      addPlayer("D"),
    ];
  });

  afterEach(() => {
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
  });

  it("POST creates a match and returns elo deltas", async () => {
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
        enteredBy: players[0],
      })
    );

    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.id).toBeTypeOf("number");
    expect(json.before).toHaveLength(4);
    expect(json.after).toHaveLength(4);
    for (const item of json.before) {
      expect(item.elo).toBe(1000);
    }
    const winners = json.after.filter(
      (item: { id: number }) => item.id === players[0] || item.id === players[1]
    );
    const losers = json.after.filter(
      (item: { id: number }) => item.id === players[2] || item.id === players[3]
    );
    for (const w of winners) expect(w.elo).toBeGreaterThan(1000);
    for (const l of losers) expect(l.elo).toBeLessThan(1000);
  });

  it("POST rejects equal scores", async () => {
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 21,
        playedAt: todayString(),
      })
    );

    expect(response.status).toBe(400);
    const json = await response.json();
    expect(json.error).toContain("equal");
  });

  it("DELETE allows retraction within 10 minutes", async () => {
    const postResponse = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
      })
    );
    const { id } = await postResponse.json();

    const deleteResponse = await deleteById(createDeleteRequest(id), {
      params: Promise.resolve({ id: String(id) }),
    });

    expect(deleteResponse.status).toBe(200);
    const json = await deleteResponse.json();
    expect(json.success).toBe(true);
  });

  it("DELETE rejects retraction after 10 minutes", async () => {
    const postResponse = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
      })
    );
    const { id } = await postResponse.json();

    const db = getDb();
    db.prepare(
      "UPDATE matches SET created_at = datetime('now', '-11 minutes') WHERE id = ?"
    ).run(id);

    const deleteResponse = await deleteById(createDeleteRequest(id), {
      params: Promise.resolve({ id: String(id) }),
    });

    expect(deleteResponse.status).toBe(403);
    const json = await deleteResponse.json();
    expect(json.error).toContain("10 minutes");
  });

  it("DELETE bypasses window with admin key", async () => {
    const postResponse = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
      })
    );
    const { id } = await postResponse.json();

    const db = getDb();
    db.prepare(
      "UPDATE matches SET created_at = datetime('now', '-11 minutes') WHERE id = ?"
    ).run(id);

    const deleteResponse = await deleteById(
      createDeleteRequest(id, { "x-admin-key": "admin-secret" }),
      { params: Promise.resolve({ id: String(id) }) }
    );

    expect(deleteResponse.status).toBe(200);
    const json = await deleteResponse.json();
    expect(json.success).toBe(true);
  });

  it("普通追加：rating.ready，estimate 与引擎 replayRatings 的该场 MatchEstimate 一致", async () => {
    initGlickoConfig();
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: shanghaiToday(),
      })
    );

    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.rating.state).toBe("ready");
    expect(json.rating.model).toBe("glicko2");
    expect(json.rating.historyRecomputed).toBe(false);

    const expected = engineEstimateFor(json.id, json.rating.asOf);
    expect(json.rating.estimate).toEqual(expected.estimate);
    expect(json.rating.configVersion).toBe(expected.configVersion);
    expect(json.rating.segmentId).toBe(expected.replay.currentSegment.id);
    expect(json.rating.estimate.changes).toHaveLength(4);
    // 旧 elo 语义字段保持：按 playerId 配对、含 name。
    expect(json.before).toHaveLength(4);
    expect(json.after).toHaveLength(4);
    expect(json.before[0]).toEqual({
      id: expect.any(Number),
      name: expect.any(String),
      elo: 1000,
    });
    expect(json.legacyRatingsAvailable).toBe(true);
  });

  it("补录旧周：ready 且 historyRecomputed=true，estimate 只含该场变化", async () => {
    initGlickoConfig();
    await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: shanghaiToday(),
      })
    );

    const pastDate = previousWeekWednesday();
    const response = await POST(
      createRequest({
        pa1: players[2],
        pa2: players[3],
        pb1: players[0],
        pb2: players[1],
        scoreA: 21,
        scoreB: 12,
        playedAt: pastDate,
      })
    );

    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.rating.state).toBe("ready");
    expect(json.rating.historyRecomputed).toBe(true);
    expect(json.rating.estimate.playedAt).toBe(pastDate);
    expect(json.rating.estimate.matchId).toBe(json.id);
    expect(json.rating.estimate.changes).toHaveLength(4);

    const expected = engineEstimateFor(json.id, json.rating.asOf);
    expect(json.rating.estimate).toEqual(expected.estimate);
    // 该场落在历史区段：其区段不是当前区段。
    expect(json.rating.segmentId).not.toBe(expected.replay.currentSegment.id);
    // 下游当前分已被重算，不冒充该场变化：当前分 ≠ 该场预估终值。
    for (const change of json.rating.estimate.changes) {
      expect(expected.replay.current[String(change.playerId)]).not.toEqual(
        change.after
      );
    }
  });

  it("未来日期：201 + not_effective，比赛事实仍保存", async () => {
    initGlickoConfig();
    const future = shanghaiLocalDateFromInstant(
      new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString()
    );
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: future,
      })
    );

    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.rating.state).toBe("not_effective");
    expect(json.rating.reason).toContain("not yet effective");
    expect(json.id).toBeTypeOf("number");
    // 事实列表保留未来记录（不作为坏数据隐藏）。
    expect(listRawMatches(getDb())).toHaveLength(1);
    expect(json.legacyRatingsAvailable).toBe(true);
    expect(json.after).toHaveLength(4);
  });

  it("未初始化新版配置：rating.not_effective，不冒充 Legacy 也不算评分失败", async () => {
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: shanghaiToday(),
      })
    );

    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.rating).toEqual({
      state: "not_effective",
      model: "glicko2",
      reason: "rating config not initialized",
    });
    // Legacy 字段不受新版配置影响。
    expect(json.before).toHaveLength(4);
    expect(json.after).toHaveLength(4);
  });

  it("评分服务失败：201 + pending，数据库恰好只有这一条新比赛", async () => {
    initGlickoConfig();
    const spy = vi
      .spyOn(ratingService, "loadGlickoSnapshot")
      .mockImplementation(() => {
        throw new Error("glicko boom");
      });
    try {
      const response = await POST(
        createRequest({
          pa1: players[0],
          pa2: players[1],
          pb1: players[2],
          pb2: players[3],
          scoreA: 21,
          scoreB: 15,
          playedAt: shanghaiToday(),
        })
      );

      expect(response.status).toBe(201);
      const json = await response.json();
      expect(json.rating.state).toBe("pending");
      expect(json.rating.reason).toContain("glicko boom");
      expect(json.id).toBeTypeOf("number");
      // 比赛已保存，客户端无需再次提交。
      expect(listRawMatches(getDb())).toHaveLength(1);
      // Legacy 附加计算不受影响。
      expect(json.legacyRatingsAvailable).toBe(true);
      expect(json.after).toHaveLength(4);
    } finally {
      spy.mockRestore();
    }
  });

  it("Legacy 附加计算失败：201 + before/after 空数组 + 显式不可用标记", async () => {
    initGlickoConfig();
    const spy = vi
      .spyOn(repo, "recomputeAllRatings")
      .mockImplementation(() => {
        throw new Error("legacy boom");
      });
    try {
      const response = await POST(
        createRequest({
          pa1: players[0],
          pa2: players[1],
          pb1: players[2],
          pb2: players[3],
          scoreA: 21,
          scoreB: 15,
          playedAt: shanghaiToday(),
        })
      );

      expect(response.status).toBe(201);
      const json = await response.json();
      expect(json.before).toEqual([]);
      expect(json.after).toEqual([]);
      expect(json.legacyRatingsAvailable).toBe(false);
      expect(json.id).toBeTypeOf("number");
      // 写入仍成功。
      expect(listRawMatches(getDb())).toHaveLength(1);
    } finally {
      spy.mockRestore();
    }
  });

  it.each([
    [{ pa1: 1.5 }, "integers"],
    [{ pb2: 99999 }, "Unknown player id: 99999"],
    [{ playedAt: "2026-02-30" }, "valid YYYY-MM-DD calendar date"],
    [{ scoreA: -1 }, "non-negative integers"],
    [{ scoreA: 21, scoreB: 21 }, "not be equal"],
  ])("验证失败返回 400：%o", async (overrides, message) => {
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: shanghaiToday(),
        ...overrides,
      })
    );

    expect(response.status).toBe(400);
    const json = await response.json();
    expect(json.error).toContain(message);
    // 验证失败的请求不落库。
    expect(listRawMatches(getDb())).toHaveLength(0);
  });

  it("数据库写入失败返回 5xx（不再误报 400）", async () => {
    const spy = vi.spyOn(repo, "addMatch").mockImplementation(() => {
      throw new Error("db write failed");
    });
    try {
      const response = await POST(
        createRequest({
          pa1: players[0],
          pa2: players[1],
          pb1: players[2],
          pb2: players[3],
          scoreA: 21,
          scoreB: 15,
          playedAt: shanghaiToday(),
        })
      );

      expect(response.status).toBe(500);
      const json = await response.json();
      expect(json.error).toContain("db write failed");
    } finally {
      spy.mockRestore();
    }
  });
});
