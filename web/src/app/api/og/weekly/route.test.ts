import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addPlayer, addMatch } from "@/lib/repo";
import {
  initializeRatingConfig,
  setActiveModel,
} from "@/lib/rating-config";
import {
  buildWeeklyStats,
  getWeekRange,
  weeklyDataVersion,
  weeklyDataVersionContext,
  OG_DESIGN_VERSION,
} from "@/lib/weekly";
import { GET } from "./route";

function createRequest(week: string, ifNoneMatch?: string): Request {
  return new Request(`http://localhost/api/og/weekly?week=${week}&v=2`, {
    headers: ifNoneMatch ? { "If-None-Match": ifNoneMatch } : {},
  });
}

const PLAYED_AT = "2026-09-02"; // 周三
const WEEK = getWeekRange(PLAYED_AT).weekStart;
// glicko2 注入时点（周一界）与对应周：2026-09-14 为上海周一。
const GLICKO_WEEK = "2026-09-14";
const AS_OF_IN_WEEK = "2026-09-16T12:00:00+08:00";
const AS_OF_NEXT_WEEK = "2026-09-23T12:00:00+08:00";

function glickoEtag(week: string, asOf: string): string {
  const stats = buildWeeklyStats(week, { rating: "glicko2", asOf });
  return `"${weeklyDataVersion(stats, weeklyDataVersionContext(stats))}-${OG_DESIGN_VERSION}"`;
}

function seedFourPlayers() {
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
    playedAt: PLAYED_AT,
  });
  return [p1, p2, p3, p4];
}

function currentEtag(): string {
  return `"${weeklyDataVersion(buildWeeklyStats(WEEK))}-${OG_DESIGN_VERSION}"`;
}

describe.sequential("og/weekly API", () => {
  let dbPath: string;

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-og-weekly-${Date.now()}.db`);
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

  it("rejects invalid week params", async () => {
    for (const bad of ["", "2026-9-2", "not-a-date"]) {
      const res = await GET(
        new Request(`http://localhost/api/og/weekly?week=${bad}`)
      );
      expect(res.status).toBe(400);
    }
  });

  it(
    "renders PNG with no-cache + etag when tag is stale or absent",
    async () => {
      seedFourPlayers();
      for (const ifNoneMatch of [undefined, '"stale-tag"']) {
        const res = await GET(createRequest(WEEK, ifNoneMatch));
        expect(res.status).toBe(200);
        expect(res.headers.get("content-type")).toBe("image/png");
        expect(res.headers.get("cache-control")).toBe("no-cache");
        expect(res.headers.get("cache-control")).not.toContain("immutable");
        expect(res.headers.get("etag")).toBe(currentEtag());
        const png = await res.arrayBuffer();
        expect(png.byteLength).toBeGreaterThan(0);
      }
    },
    // Satori + resvg wasm 冷渲染单次需 2-5s,循环两次,默认 5s 超时不够
    30000
  );

  it("short-circuits with 304 when If-None-Match matches current version", async () => {
    seedFourPlayers();
    const res = await GET(createRequest(WEEK, currentEtag()));
    expect(res.status).toBe(304);
    expect(res.headers.get("etag")).toBe(currentEtag());
    expect(res.headers.get("cache-control")).toBe("no-cache");
    expect(await res.text()).toBe("");
  });

  it(
    "etag changes after new match data",
    async () => {
      const [p1, p2, p3, p4] = seedFourPlayers();
      const before = currentEtag();
      const res1 = await GET(createRequest(WEEK, before));
      expect(res1.status).toBe(304);

      addMatch({
        pa1: p1,
        pa2: p3,
        pb1: p2,
        pb2: p4,
        scoreA: 18,
        scoreB: 21,
        playedAt: PLAYED_AT,
      });

      const after = currentEtag();
      expect(after).not.toBe(before);
      // 旧 etag 失效 → 重新渲染
      const res2 = await GET(createRequest(WEEK, before));
      expect(res2.status).toBe(200);
      expect(res2.headers.get("etag")).toBe(after);
      // 新 etag → 又可以 304
      const res3 = await GET(createRequest(WEEK, after));
      expect(res3.status).toBe(304);
    },
    30000
  );

  it("glicko2 服务不可用时返回 409，不伪造图片", async () => {
    // 未初始化配置 → unavailable。
    seedFourPlayers();
    const res = await GET(
      new Request(
        `http://localhost/api/og/weekly?week=${GLICKO_WEEK}&rating=glicko2&asOf=${encodeURIComponent(AS_OF_IN_WEEK)}`
      )
    );
    expect(res.status).toBe(409);
    const data = await res.json();
    expect(data.reason).toBeTypeOf("string");
  });

  it("glicko2 同内容可 304（协商缓存结构不变）", async () => {
    initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, getDb());
    seedFourPlayersGlickoWeek();
    const etag = glickoEtag(GLICKO_WEEK, AS_OF_IN_WEEK);
    const res = await GET(
      new Request(
        `http://localhost/api/og/weekly?week=${GLICKO_WEEK}&rating=glicko2&asOf=${encodeURIComponent(AS_OF_IN_WEEK)}`,
        { headers: { "If-None-Match": etag } }
      )
    );
    expect(res.status).toBe(304);
    expect(res.headers.get("etag")).toBe(etag);
    expect(res.headers.get("cache-control")).toBe("no-cache");
    expect(await res.text()).toBe("");
  });

  it(
    "glicko2 跨边界即使 DB 未变也不能复用旧表示",
    async () => {
      initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, getDb());
      seedFourPlayersGlickoWeek();
      const etagA = glickoEtag(GLICKO_WEEK, AS_OF_IN_WEEK);
      const etagB = glickoEtag(GLICKO_WEEK, AS_OF_NEXT_WEEK);
      expect(etagB).not.toBe(etagA);

      // 旧边界 etag 对新边界请求失效 → 重新渲染并返回新 etag。
      const res = await GET(
        new Request(
          `http://localhost/api/og/weekly?week=${GLICKO_WEEK}&rating=glicko2&asOf=${encodeURIComponent(AS_OF_NEXT_WEEK)}`,
          { headers: { "If-None-Match": etagA } }
        )
      );
      expect(res.status).toBe(200);
      expect(res.headers.get("content-type")).toBe("image/png");
      expect(res.headers.get("etag")).toBe(etagB);

      // 新 etag → 304。
      const res304 = await GET(
        new Request(
          `http://localhost/api/og/weekly?week=${GLICKO_WEEK}&rating=glicko2&asOf=${encodeURIComponent(AS_OF_NEXT_WEEK)}`,
          { headers: { "If-None-Match": etagB } }
        )
      );
      expect(res304.status).toBe(304);
    },
    30000
  );

  it("activeModel=glicko2 时缺省请求走统一模型解析（不带 rating 参数）", async () => {
    initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, getDb());
    setActiveModel("glicko2", getDb());
    seedFourPlayersGlickoWeek();
    const etag = glickoEtag(GLICKO_WEEK, AS_OF_IN_WEEK);
    const res = await GET(
      new Request(
        `http://localhost/api/og/weekly?week=${GLICKO_WEEK}&asOf=${encodeURIComponent(AS_OF_IN_WEEK)}`,
        { headers: { "If-None-Match": etag } }
      )
    );
    expect(res.status).toBe(304);
    expect(res.headers.get("etag")).toBe(etag);
  });

  it("glicko2 非法 week 日期返回 400", async () => {
    initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, getDb());
    const res = await GET(
      new Request(
        `http://localhost/api/og/weekly?week=2026-13-99&rating=glicko2&asOf=${encodeURIComponent(AS_OF_IN_WEEK)}`
      )
    );
    expect(res.status).toBe(400);
  });
});

function seedFourPlayersGlickoWeek() {
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
    playedAt: GLICKO_WEEK,
  });
  return [p1, p2, p3, p4];
}
