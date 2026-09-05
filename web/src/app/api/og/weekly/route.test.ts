import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb } from "@/lib/db";
import { addPlayer, addMatch } from "@/lib/repo";
import { buildWeeklyStats, getWeekRange, weeklyDataVersion } from "@/lib/weekly";
import { GET } from "./route";

function createRequest(week: string, ifNoneMatch?: string): Request {
  return new Request(`http://localhost/api/og/weekly?week=${week}&v=2`, {
    headers: ifNoneMatch ? { "If-None-Match": ifNoneMatch } : {},
  });
}

const PLAYED_AT = "2026-09-02"; // 周三
const WEEK = getWeekRange(PLAYED_AT).weekStart;

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
  return `"${weeklyDataVersion(buildWeeklyStats(WEEK))}"`;
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

  it("renders PNG with no-cache + etag when tag is stale or absent", async () => {
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
  });

  it("short-circuits with 304 when If-None-Match matches current version", async () => {
    seedFourPlayers();
    const res = await GET(createRequest(WEEK, currentEtag()));
    expect(res.status).toBe(304);
    expect(res.headers.get("etag")).toBe(currentEtag());
    expect(res.headers.get("cache-control")).toBe("no-cache");
    expect(await res.text()).toBe("");
  });

  it("etag changes after new match data", async () => {
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
  });
});
