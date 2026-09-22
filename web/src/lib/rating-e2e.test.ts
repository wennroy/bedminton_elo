/**
 * T7 要点 2 的自动验收：固定测试时钟的端到端全链路走查回归。
 *
 * 走查链（同一临时 DB，评分时钟一律经各入口的 asOf 注入，不 mock 系统时间）：
 *   初始化配置 → 录入 → 逐场 Estimated → 无新写入推进 asOf 过周界（Final+校准）
 *   → 过季界（跨季周双段 + season_reset + 峰值口径）→ 旧周补录 → 撤回
 *   → 改名/合并 → OG 分享图 ETag 按预期变/不变；另有 replay 故障旧快照如实
 *   stale、三处事实源比赛集合一致、同库 Legacy 对照三组独立场景。
 *
 * 每个写步骤后核对：读到的 model/version/asOf、重放 events、事实行数与
 * weeklyDataVersion + OG_DESIGN_VERSION 的 ETag 变化。
 *
 * 已知限制（同步记录给真实回放报告）：比赛事实只有按日粒度（playedAt 为
 * YYYY-MM-DD），同日多场的真实开赛顺序无法从普通按日数据还原，重放按录入
 * 顺序（createdAt、id）排序同日场次。真实回放的同日顺序结论须以生产快照
 * 中带时间戳的录入流水为准（docs/ratings-validation.md，用户提供快照后单独
 * 处理），本文件不重复验证该限制。
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import {
  addMatch,
  addPlayer,
  deleteMatch,
  listMatchesByDate,
  listRawMatches,
  mergePlayers,
  renamePlayer,
} from "@/lib/repo";
import {
  initializeRatingConfig,
  readRatingConfig,
  setActiveModel,
} from "@/lib/rating-config";
import { loadGlickoSnapshot } from "@/lib/rating-service";
import { loadRatingView, loadPredictionView } from "@/lib/rating-view";
import { ratingConfigVersion } from "@/lib/ratings/config";
import {
  nextWeekStart,
  shanghaiLocalDateFromInstant,
  weekStart,
} from "@/lib/ratings/calendar";
import type { RatingConfig } from "@/lib/ratings/types";
import {
  buildWeeklyStats,
  weeklyDataVersion,
  weeklyDataVersionContext,
  OG_DESIGN_VERSION,
} from "@/lib/weekly";
import { GET as ogWeeklyGET } from "@/app/api/og/weekly/route";

/** 场景 9 的故障注入开关：只让 replayRatings 抛错，其余模块保持原样。 */
const replayControl = vi.hoisted(() => ({ fail: false }));

vi.mock("@/lib/ratings/replay", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/ratings/replay")>();
  return {
    ...original,
    replayRatings: (
      ...args: Parameters<typeof original.replayRatings>
    ): ReturnType<typeof original.replayRatings> => {
      if (replayControl.fail) {
        throw new Error("simulated replay failure for stale walkthrough");
      }
      return original.replayRatings(...args);
    },
  };
});

/** 固定时钟（全部 ISO instant，含显式时区）与固定的首赛季起点。 */
const FIRST_SEASON_START = "2026-07-01";
const W1 = "2026-09-07"; // 周一
const W3 = "2026-09-28"; // 周一；本周四 2026-10-01 为季界
const AS_OF_RECORD = "2026-09-09T21:00:00+08:00"; // W1 周三晚（录入后当场读）
const AS_OF_WEEK_LATER = "2026-09-16T12:00:00+08:00"; // W2 周三（无新写入过周界）
const AS_OF_CROSS_SEASON = "2026-10-03T12:00:00+08:00"; // W3 季界后段（周六）
const AS_OF_AFTER_SEASON = "2026-10-07T12:00:00+08:00"; // W4 周三（双段皆 Final）
// 首个区段从「首个有效比赛日」起算（引擎不为首个比赛前积累 RD 历史），
// 故 W1 区段起点是 2026-09-09（首场比赛日）而非周一。
const SEG_W1 = "2026-09-07:2026-09-09";
const SEG_W2 = "2026-09-14:2026-09-14";
const SEG_W3A = "2026-09-28:2026-09-28";
const SEG_W3B = "2026-09-28:2026-10-01";
const SEG_W4 = "2026-10-05:2026-10-05";
const NEXT_BOUNDARY_W1 = "2026-09-13T16:00:00Z"; // 上海周一 2026-09-14 零点
const NEXT_BOUNDARY_W3B = "2026-10-04T16:00:00Z"; // 上海周一 2026-10-05 零点

/** 当前 DB 的持久化配置；未初始化即抛错（不把缺失配置当成默认值）。 */
function mustConfig(): RatingConfig {
  const record = readRatingConfig(getDb());
  if (record === null) throw new Error("rating config not initialized");
  return record.config;
}

/** 引擎赛季软重置公式（与 ratings/replay 的 softResetRating 同式）。 */
function softResetR(r: number, config: RatingConfig): number {
  if (r < config.seasonLower) {
    return config.seasonLower + config.seasonRetention * (r - config.seasonLower);
  }
  if (r > config.seasonUpper) {
    return config.seasonUpper + config.seasonRetention * (r - config.seasonUpper);
  }
  return r;
}

/** 与 OG 路由一致的内容指纹 ETag（weeklyDataVersion + OG_DESIGN_VERSION）。 */
function glickoEtag(week: string, asOf: string): string {
  const stats = buildWeeklyStats(week, { rating: "glicko2", asOf, db: getDb() });
  return `"${weeklyDataVersion(stats, weeklyDataVersionContext(stats))}-${OG_DESIGN_VERSION}"`;
}

function readyReplay(asOf: string) {
  const result = loadGlickoSnapshot(getDb(), asOf);
  if (result.state !== "ready") {
    throw new Error(`expected ready replay at ${asOf}, got ${result.state}`);
  }
  return result;
}

describe.sequential("rating e2e 固定时钟全链路走查（T7 要点 2）", () => {
  let dbPath: string;
  let players: number[] = [];
  let A = 0;
  let B = 0;
  let C = 0;
  let D = 0;
  let E = 0;
  let expectedVersion = "";

  /** 固定时钟下的配置初始化 + activeModel 切 glicko2，并冻结期望版本串。 */
  function initFixedConfig() {
    initializeRatingConfig(
      { firstSeasonStart: FIRST_SEASON_START, todayInstant: AS_OF_RECORD },
      getDb()
    );
    setActiveModel("glicko2", getDb());
    expectedVersion = ratingConfigVersion(mustConfig());
  }

  /** 读视图并核对每次读到的 model/版本/asOf（stale/unavailable 直接抛错）。 */
  function expectFreshView(asOf: string) {
    const result = loadRatingView({ rating: "glicko2", asOf, db: getDb() });
    if (result.model !== "glicko2" || result.freshness !== "fresh") {
      throw new Error(`expected fresh glicko2 view at ${asOf}`);
    }
    expect(result.asOf).toBe(asOf);
    expect(result.version).toBe(expectedVersion);
    return result;
  }

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-badminton-e2e-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    players = [
      addPlayer("A"),
      addPlayer("B"),
      addPlayer("C"),
      addPlayer("D"),
      addPlayer("E"),
    ];
    const [a, b, c, d, e] = players;
    if (a === undefined || b === undefined || c === undefined || d === undefined || e === undefined) {
      throw new Error("player fixture incomplete");
    }
    A = a;
    B = b;
    C = c;
    D = d;
    E = e;
  });

  afterEach(() => {
    replayControl.fail = false;
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
  });

  it(
    "场景 1–8：录入→Estimated→周界 Final→季界双段→旧周补录→撤回→改名/合并→OG ETag",
    async () => {
      const db = getDb();

      // —— 场景 1：初始化配置（首赛季起点固定）→ activeModel 切 glicko2 ——
      initFixedConfig();
      const record = readRatingConfig(db);
      if (record === null) throw new Error("config missing");
      expect(record.config.firstSeasonStart).toBe(FIRST_SEASON_START);
      expect(record.activeModel).toBe("glicko2");
      expect(loadRatingView({ asOf: AS_OF_RECORD, db }).model).toBe("glicko2");

      // —— 场景 2：录入一场 → 同 asOf 读到该场 Estimated，模型/版本/asOf 正确 ——
      const m1 = addMatch(
        { pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: "2026-09-09" },
        db
      );
      const atRecord = expectFreshView(AS_OF_RECORD);
      expect(atRecord.currentSegmentId).toBe(SEG_W1);
      expect(atRecord.nextBoundary).toBe(NEXT_BOUNDARY_W1);
      const estimate1 = atRecord.view.matchEstimatesById[String(m1)];
      expect(estimate1).toBeDefined();
      expect(estimate1?.preWinA).toBeCloseTo(0.5, 6); // 全员初始同分
      expect(estimate1?.changes.find((c) => c.playerId === A)?.delta).toBeGreaterThan(0);
      expect(estimate1?.changes.find((c) => c.playerId === D)?.delta).toBeLessThan(0);
      const estPoints1 = atRecord.view.points.filter((p) => p.kind === "match_estimated");
      expect(estPoints1).toHaveLength(4);
      expect(
        estPoints1.every(
          (p) =>
            p.status === "estimated" &&
            p.at === "2026-09-09" &&
            p.segment === SEG_W1 &&
            p.season === "2026-07-01"
        )
      ).toBe(true);
      expect(atRecord.view.points.some((p) => p.kind === "weekly_final")).toBe(false);
      expect(atRecord.view.weekSegments.find((s) => s.segmentId === SEG_W1)?.matches.map((m) => m.matchId)).toEqual([m1]);
      expect(atRecord.view.players.find((p) => p.playerId === A)?.status).toBe("estimated");
      expect(atRecord.view.players.find((p) => p.playerId === E)?.status).toBe("unrated");
      expect(listRawMatches(db)).toHaveLength(1);

      // —— 场景 3：无新写入推进 asOf 过周界 → 上周 Final + 校准，当前段重置为空 ——
      const atWeekLater = expectFreshView(AS_OF_WEEK_LATER);
      expect(atWeekLater.currentSegmentId).toBe(SEG_W2);
      const rowW1 = atWeekLater.view.weekSegments.find((s) => s.segmentId === SEG_W1);
      if (rowW1 === undefined) throw new Error("W1 segment missing");
      expect(rowW1.matches.map((m) => m.matchId)).toEqual([m1]);
      expect(Object.keys(rowW1.correction).sort()).toEqual(
        [String(A), String(B), String(C), String(D)].sort()
      );
      expect(Object.values(rowW1.correction).some((v) => v !== 0)).toBe(true); // 校准确实出现
      const rowW2 = atWeekLater.view.weekSegments.find((s) => s.segmentId === SEG_W2);
      expect(rowW2?.matches).toEqual([]); // 当前段无逐场变化
      const finalPoints = atWeekLater.view.points.filter((p) => p.kind === "weekly_final");
      expect(finalPoints).toHaveLength(4);
      expect(finalPoints.every((p) => p.status === "final" && p.season === "2026-07-01")).toBe(true);
      expect(atWeekLater.view.players.find((p) => p.playerId === A)?.status).toBe("final");
      const w1FinalA = finalPoints.find((p) => p.playerId === A)?.r;
      expect(w1FinalA).toBeDefined();
      const weeklyW1 = buildWeeklyStats(W1, { rating: "glicko2", asOf: AS_OF_WEEK_LATER, db });
      expect(weeklyW1.ratingReport?.segments).toHaveLength(1);
      expect(weeklyW1.ratingReport?.segments[0]?.status).toBe("final");
      expect(weeklyW1.ratingReport?.resets).toEqual([]);
      // 无新写入仅推进时钟：W1 从 Estimated 变 Final，分享图指纹同样失效。
      expect(glickoEtag(W1, AS_OF_WEEK_LATER)).not.toBe(glickoEtag(W1, AS_OF_RECORD));

      // —— 场景 4a：录入跨季周两场，asOf 落在季界后段（当前段） ——
      const m2 = addMatch(
        { pa1: A, pa2: C, pb1: B, pb2: D, scoreA: 21, scoreB: 10, playedAt: "2026-09-30" },
        db
      );
      const m3 = addMatch(
        { pa1: A, pa2: D, pb1: B, pb2: C, scoreA: 18, scoreB: 21, playedAt: "2026-10-02" },
        db
      );
      const atCross = expectFreshView(AS_OF_CROSS_SEASON);
      expect(atCross.currentSegmentId).toBe(SEG_W3B);
      expect(atCross.nextBoundary).toBe(NEXT_BOUNDARY_W3B);
      const segA = atCross.view.weekSegments.find((s) => s.segmentId === SEG_W3A);
      const segB = atCross.view.weekSegments.find((s) => s.segmentId === SEG_W3B);
      if (segA === undefined || segB === undefined) {
        throw new Error("cross-season segments missing");
      }
      // 跨季周双段：同周两个区段、两个赛季，前段已 Final 且挂重置，后段 Estimated。
      expect(segA.seasonId).toBe("2026-07-01");
      expect(segB.seasonId).toBe("2026-10-01");
      expect(segA.matches.map((m) => m.matchId)).toEqual([m2]);
      expect(segB.matches.map((m) => m.matchId)).toEqual([m3]);
      expect(segA.reset?.seasonId).toBe("2026-10-01");
      expect(segB.reset).toBeNull();
      const resetPoints = atCross.view.points.filter((p) => p.kind === "season_reset");
      expect(resetPoints.length).toBeGreaterThan(0);
      expect(resetPoints.every((p) => p.status === "final" && p.season === "2026-10-01")).toBe(true);
      const resetEvents = readyReplay(AS_OF_CROSS_SEASON).replay.events.filter(
        (e) => e.kind === "season_reset"
      );
      expect(resetEvents).toHaveLength(1);
      const weeklyW3 = buildWeeklyStats(W3, { rating: "glicko2", asOf: AS_OF_CROSS_SEASON, db });
      expect(weeklyW3.ratingReport?.segments.map((s) => s.segmentId)).toEqual([SEG_W3A, SEG_W3B]);
      expect(weeklyW3.ratingReport?.segments.map((s) => s.status)).toEqual(["final", "estimated"]);
      expect(weeklyW3.ratingReport?.resets).toHaveLength(1);

      // —— 场景 4b：无新写入再过季界 → 双段皆 Final、重置数值按软重置公式、峰值只认 Final ——
      const atAfter = expectFreshView(AS_OF_AFTER_SEASON);
      expect(atAfter.currentSegmentId).toBe(SEG_W4);
      const weeklyW3After = buildWeeklyStats(W3, { rating: "glicko2", asOf: AS_OF_AFTER_SEASON, db });
      expect(weeklyW3After.ratingReport?.segments.map((s) => s.status)).toEqual(["final", "final"]);
      expect(weeklyW3After.ratingReport?.resets).toHaveLength(1);
      const replayAfter = readyReplay(AS_OF_AFTER_SEASON);
      const resetEvent = replayAfter.replay.events.find((e) => e.kind === "season_reset");
      if (resetEvent === undefined || resetEvent.kind !== "season_reset") {
        throw new Error("season reset event missing");
      }
      expect(resetEvent.seasonId).toBe("2026-10-01");
      expect(resetEvent.changes.map((c) => c.playerId).sort()).toEqual([A, B, C, D].sort());
      const config = mustConfig();
      for (const change of resetEvent.changes) {
        expect(change.after.r).toBeCloseTo(softResetR(change.before.r, config), 6);
        expect(change.after.rd).toBe(
          Math.min(config.maxRd, Math.max(change.before.rd, config.seasonRdFloor))
        );
      }
      // 正式峰值只取 PeriodFinal.final：峰值 eventId 全部来自 weekly_final 事件，
      // 数值等于各周 Final 赛末值的最大值（不含重置后与 Estimated）。
      const expectedPeak = new Map<number, number>();
      for (const event of replayAfter.replay.events) {
        if (event.kind !== "weekly_final") continue;
        for (const [pid, state] of Object.entries(event.final)) {
          const id = Number(pid);
          expectedPeak.set(id, Math.max(expectedPeak.get(id) ?? Number.NEGATIVE_INFINITY, state.r));
        }
      }
      for (const id of [A, B, C, D]) {
        const peak = atAfter.view.peakFinal[String(id)];
        if (peak === null) throw new Error(`peakFinal missing for player ${id}`);
        expect(peak.eventId.startsWith("weekly_final:")).toBe(true);
        expect(peak.r).toBeCloseTo(expectedPeak.get(id) ?? Number.NaN, 6);
      }
      expect(atAfter.view.peakFinal[String(E)]).toBeNull();
      // OG：仅推进 asOf 过季界，W3 分享内容（双段状态 + 重置）指纹失效。
      expect(glickoEtag(W3, AS_OF_AFTER_SEASON)).not.toBe(glickoEtag(W3, AS_OF_CROSS_SEASON));

      // —— 场景 5：旧周补录 → 历史周重算，补录场 Estimated 落旧段，当前分与历史变化区分 ——
      const beforeBackfill = readyReplay(AS_OF_AFTER_SEASON);
      const etagW1BeforeBackfill = glickoEtag(W1, AS_OF_AFTER_SEASON);
      const m4 = addMatch(
        { pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 12, playedAt: "2026-09-11" },
        db
      );
      const afterBackfill = expectFreshView(AS_OF_AFTER_SEASON);
      const replayBackfill = readyReplay(AS_OF_AFTER_SEASON);
      expect(replayBackfill.inputHash).not.toBe(beforeBackfill.inputHash);
      // 补录场的逐场 Estimated 落在旧周区段，状态仍是 Estimated。
      expect(afterBackfill.view.matchEstimatesById[String(m4)]?.segmentId).toBe(SEG_W1);
      expect(
        afterBackfill.view.points
          .filter((p) => p.kind === "match_estimated" && p.segment === SEG_W1)
          .map((p) => p.eventId)
      ).toHaveLength(8); // M1 + M4 各 4 人
      const rowW1Backfill = afterBackfill.view.weekSegments.find((s) => s.segmentId === SEG_W1);
      expect(rowW1Backfill?.matches.map((m) => m.matchId)).toEqual([m1, m4]);
      // 历史周重算：W1 的 Final 与补录前不同。
      const w1FinalAAfterBackfill = afterBackfill.view.points.find(
        (p) => p.kind === "weekly_final" && p.segment === SEG_W1 && p.playerId === A
      )?.r;
      expect(w1FinalAAfterBackfill).toBeDefined();
      expect(w1FinalAAfterBackfill).not.toBe(w1FinalA);
      // 当前分变化与该场历史变化区分：当前段没有新逐场事件，但 current 已被下游重算。
      expect(afterBackfill.view.weekSegments.find((s) => s.segmentId === SEG_W4)?.matches).toEqual([]);
      expect(replayBackfill.replay.current[String(A)]?.r).not.toBe(
        beforeBackfill.replay.current[String(A)]?.r
      );
      const weeklyW1Backfill = buildWeeklyStats(W1, { rating: "glicko2", asOf: AS_OF_AFTER_SEASON, db });
      const reportW1Seg = weeklyW1Backfill.ratingReport?.segments.find((s) => s.segmentId === SEG_W1);
      expect(reportW1Seg?.players.find((p) => p.playerId === A)?.matchesPlayed).toBe(2);
      const etagW1Backfill = glickoEtag(W1, AS_OF_AFTER_SEASON);
      expect(etagW1Backfill).not.toBe(etagW1BeforeBackfill);

      // —— 场景 6：撤回（删比赛）→ 该场从事实与重放消失 ——
      deleteMatch(m2, db);
      const afterDelete = expectFreshView(AS_OF_AFTER_SEASON);
      const replayDelete = readyReplay(AS_OF_AFTER_SEASON);
      expect(afterDelete.view.matchEstimatesById[String(m2)]).toBeUndefined();
      expect(
        afterDelete.view.points.some((p) => p.eventId === `match_estimated:${SEG_W3A}:${m2}`)
      ).toBe(false);
      expect(
        replayDelete.replay.events.some((e) => e.kind === "match_estimated" && e.matchId === m2)
      ).toBe(false);
      expect(afterDelete.view.weekSegments.find((s) => s.segmentId === SEG_W3A)?.matches).toEqual([]);
      expect(listRawMatches(db)).toHaveLength(3);
      expect(listMatchesByDate(db)).toHaveLength(3);
      // 全库输入指纹进分享图上下文：与 W1 无关的撤回也使 W1 ETag 保守失效。
      const etagW1Delete = glickoEtag(W1, AS_OF_AFTER_SEASON);
      expect(etagW1Delete).not.toBe(etagW1Backfill);

      // —— 场景 7a：改名 → 指纹变化、事实不重写、投影一致 ——
      const rawsBeforeRename = listRawMatches(db);
      const currentBeforeRename = replayDelete.replay.current;
      renamePlayer(C, "C-star", db);
      const afterRename = expectFreshView(AS_OF_AFTER_SEASON);
      expect(afterRename.inputHash).not.toBe(afterDelete.inputHash);
      expect(listRawMatches(db)).toEqual(rawsBeforeRename); // 比赛事实行不重写
      expect(afterRename.view.players.find((p) => p.playerId === C)?.name).toBe("C-star");
      expect(readyReplay(AS_OF_AFTER_SEASON).replay.current).toEqual(currentBeforeRename); // 改名不动分
      const etagW1Rename = glickoEtag(W1, AS_OF_AFTER_SEASON);
      expect(etagW1Rename).not.toBe(etagW1Delete);

      // —— 场景 7b：合并（未参赛 E 并入 D）→ 目录变化、事实与分值不变 ——
      const rawsBeforeMerge = listRawMatches(db);
      mergePlayers(E, D, db);
      const afterMerge = expectFreshView(AS_OF_AFTER_SEASON);
      expect(afterMerge.inputHash).not.toBe(afterRename.inputHash);
      expect(listRawMatches(db)).toEqual(rawsBeforeMerge); // E 无参赛事实，合并不重写比赛行
      expect(afterMerge.view.players.some((p) => p.playerId === E)).toBe(false);
      expect(afterMerge.view.players.find((p) => p.playerId === D)?.name).toBe("D");
      expect(readyReplay(AS_OF_AFTER_SEASON).replay.current).toEqual(currentBeforeRename);
      const etagW1Merge = glickoEtag(W1, AS_OF_AFTER_SEASON);
      expect(etagW1Merge).not.toBe(etagW1Rename);

      // —— 场景 8：OG 链路 ETag 变/不变 ——
      // 不变：同数据同 asOf 重算稳定；同一区段内换毫秒级 asOf 也不变。
      expect(glickoEtag(W1, AS_OF_AFTER_SEASON)).toBe(etagW1Merge);
      expect(glickoEtag(W1, "2026-10-07T18:30:00+08:00")).toBe(etagW1Merge);
      // 路由协商：If-None-Match 命中当前指纹 → 304 短路不渲染。
      const probe = await ogWeeklyGET(
        new Request(
          `http://localhost/api/og/weekly?week=${W1}&rating=glicko2&asOf=${encodeURIComponent(AS_OF_AFTER_SEASON)}`,
          { headers: { "If-None-Match": etagW1Merge } }
        )
      );
      expect(probe.status).toBe(304);
      expect(probe.headers.get("etag")).toBe(etagW1Merge);
      // 全量渲染一次：响应 ETag 头与库侧计算指纹一致（Satori 冷渲染需数秒）。
      const etagW3Final = glickoEtag(W3, AS_OF_AFTER_SEASON);
      const render = await ogWeeklyGET(
        new Request(
          `http://localhost/api/og/weekly?week=${W3}&rating=glicko2&asOf=${encodeURIComponent(AS_OF_AFTER_SEASON)}`
        )
      );
      expect(render.status).toBe(200);
      expect(render.headers.get("etag")).toBe(etagW3Final);
      expect(render.headers.get("content-type")).toBe("image/png");
    },
    30000
  );

  it("场景 9：replay 故障时旧快照如实 stale、预测拒答、恢复后回 ready", () => {
    initFixedConfig();
    const db = getDb();
    addMatch(
      { pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: "2026-09-09" },
      db
    );
    const good = readyReplay(AS_OF_RECORD); // 已写入最近成功快照（asOf = AS_OF_RECORD）

    // 同一输入、无新写入，仅把时钟推进到新周界后触发重算失败。
    replayControl.fail = true;
    const staleView = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK_LATER, db });
    if (staleView.model !== "glicko2" || staleView.freshness !== "stale") {
      throw new Error(`expected stale glicko2 view, got ${staleView.model}/${staleView.freshness}`);
    }
    // 如实报旧成功时点与旧区段，不冒充当前。
    expect(staleView.asOf).toBe(good.replay.asOf);
    expect(staleView.asOf).not.toBe(AS_OF_WEEK_LATER);
    expect(staleView.version).toBe(expectedVersion);
    expect(staleView.inputHash).toBeNull();
    expect(staleView.currentSegmentId).toBe(good.replay.currentSegment.id);
    expect(staleView.view.points.length).toBeGreaterThan(0); // 旧投影仍可读
    // 预测入口不给任何胜率/变化数字。
    const stalePrediction = loadPredictionView({
      pa1: A,
      pa2: B,
      pb1: C,
      pb2: D,
      rating: "glicko2",
      asOf: AS_OF_WEEK_LATER,
      db,
    });
    if (stalePrediction.model !== "glicko2" || stalePrediction.freshness !== "stale") {
      throw new Error("expected stale glicko2 prediction");
    }
    expect(stalePrediction.lastGoodAsOf).toBe(good.replay.asOf);
    expect("preWinA" in stalePrediction).toBe(false);
    expect("players" in stalePrediction).toBe(false);
    // 周报如实带 stale 标记与旧 asOf：可用展示但不冒充当前。
    const staleWeekly = buildWeeklyStats(W1, { rating: "glicko2", asOf: AS_OF_WEEK_LATER, db });
    expect(staleWeekly.ratingReport?.freshness).toBe("stale");
    expect(staleWeekly.ratingReport?.asOf).toBe(good.replay.asOf);

    // 恢复：同一 asOf 重新 ready，asOf 如实为新时点。
    replayControl.fail = false;
    const recovered = readyReplay(AS_OF_WEEK_LATER);
    expect(recovered.replay.asOf).toBe(AS_OF_WEEK_LATER);
    const freshAgain = loadRatingView({ rating: "glicko2", asOf: AS_OF_WEEK_LATER, db });
    expect(freshAgain.model).toBe("glicko2");
    expect(freshAgain.freshness).toBe("fresh");
    expect(freshAgain.asOf).toBe(AS_OF_WEEK_LATER);
  });

  it("场景 10：/matches 数据源、重放 events 与周报区段的每周比赛集合一致", () => {
    initFixedConfig();
    const db = getDb();
    addMatch({ pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: "2026-09-09" }, db);
    addMatch({ pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 12, playedAt: "2026-09-11" }, db);
    addMatch({ pa1: A, pa2: C, pb1: B, pb2: D, scoreA: 21, scoreB: 10, playedAt: "2026-09-30" }, db);
    addMatch({ pa1: A, pa2: D, pb1: B, pb2: C, scoreA: 18, scoreB: 21, playedAt: "2026-10-02" }, db);
    const asOf = AS_OF_AFTER_SEASON;

    const all = listMatchesByDate(db);
    expect(all).toHaveLength(4);
    const replay = readyReplay(asOf);
    const view = expectFreshView(asOf);
    const ids = (set: Iterable<number>) => [...set].sort((x, y) => x - y);
    const mondays = [...new Set(all.map((m) => weekStart(m.playedAt)))].sort();
    expect(mondays).toEqual([W1, W3]);

    for (const monday of mondays) {
      const endExclusive = nextWeekStart(monday);
      const weekMatches = all.filter((m) => m.playedAt >= monday && m.playedAt < endExclusive);
      const fromMatches = new Set(weekMatches.map((m) => m.id));
      const fromEvents = new Set(
        replay.replay.events.flatMap((e) =>
          e.kind === "match_estimated" && e.playedAt >= monday && e.playedAt < endExclusive
            ? [e.matchId]
            : []
        )
      );
      const fromSegments = new Set(
        view.view.weekSegments
          .filter(
            (s) =>
              shanghaiLocalDateFromInstant(s.start) < endExclusive &&
              shanghaiLocalDateFromInstant(s.end) > monday
          )
          .flatMap((s) => s.matches.map((m) => m.matchId))
      );
      // 三处事实源的比赛集合一致，无漏记录、无多余记录。
      expect(ids(fromEvents)).toEqual(ids(fromMatches));
      expect(ids(fromSegments)).toEqual(ids(fromMatches));

      // 周报侧：report 不携带 matchId，按每人场次与参赛者并集核对；
      // 跨季周被拆成两段，每人周总场次 = 各段场次之和。
      const stats = buildWeeklyStats(monday, { rating: "glicko2", asOf, db });
      const report = stats.ratingReport;
      if (report === undefined) throw new Error("expected rating report");
      const playedInWeek = new Map<number, number>();
      for (const m of weekMatches) {
        for (const id of [m.pa1, m.pa2, m.pb1, m.pb2]) {
          playedInWeek.set(id, (playedInWeek.get(id) ?? 0) + 1);
        }
      }
      const reportedWeek = new Map<number, number>();
      for (const segment of report.segments) {
        for (const player of segment.players) {
          reportedWeek.set(
            player.playerId,
            (reportedWeek.get(player.playerId) ?? 0) + player.matchesPlayed
          );
        }
      }
      expect(ids(reportedWeek.keys())).toEqual(ids(playedInWeek.keys()));
      for (const [playerId, matchesPlayed] of reportedWeek) {
        expect(matchesPlayed).toBe(playedInWeek.get(playerId) ?? 0);
      }
    }

    // 全局并集：重放逐场数 === 事实总数 === 区段逐场并集。
    const estimateEvents = replay.replay.events.filter((e) => e.kind === "match_estimated");
    expect(estimateEvents).toHaveLength(all.length);
    const unionSegments = new Set(view.view.weekSegments.flatMap((s) => s.matches.map((m) => m.matchId)));
    expect(ids(unionSegments)).toEqual(ids(all.map((m) => m.id)));
  });

  it("Legacy 对照：同库显式 legacy 走旧 stats 语义，不与新版混", () => {
    initFixedConfig();
    const db = getDb();
    addMatch(
      { pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: "2026-09-09" },
      db
    );
    const asOf = AS_OF_RECORD;

    // 默认入口跟随 activeModel = glicko2；显式 legacy 不被带偏。
    expect(loadRatingView({ asOf, db }).model).toBe("glicko2");
    const legacy = loadRatingView({ rating: "legacy", asOf, db });
    if (legacy.model !== "legacy") throw new Error("expected legacy view");
    expect(legacy.version).toBe("legacy");
    expect(legacy.freshness).toBe("fresh");
    expect(legacy.nextBoundary).toBeNull();
    expect(legacy.inputHash).toBeNull();
    // 旧 stats 语义：无新版区段/峰值结构，ELO/TrueSkill 数据保留。
    expect("weekSegments" in (legacy.view as object)).toBe(false);
    expect("peakFinal" in (legacy.view as object)).toBe(false);
    expect(Array.isArray(legacy.view.players)).toBe(true);
    expect(legacy.view.players).toHaveLength(5);
    expect(legacy.view.ratings instanceof Map).toBe(true);
    expect(legacy.view.eloHistory.length).toBeGreaterThan(0);
    const glicko = expectFreshView(asOf);
    expect("weekSegments" in (glicko.view as object)).toBe(true);
    expect("peakFinal" in (glicko.view as object)).toBe(true);

    // 预测：legacy 走本地 ELO 口径并给数字。
    const legacyPred = loadPredictionView({ pa1: A, pa2: B, pb1: C, pb2: D, rating: "legacy", asOf, db });
    if (legacyPred.model !== "legacy") throw new Error("expected legacy prediction");
    expect(legacyPred.version).toBe("legacy");
    expect(legacyPred.preWinA).toBeGreaterThan(0);
    expect(legacyPred.preWinA).toBeLessThan(1);
    for (const outcome of legacyPred.players) {
      expect(typeof outcome.win).toBe("number");
      expect(typeof outcome.loss).toBe("number");
    }

    // 周报：legacy 无 ratingReport；同库显式 glicko2 仍有，互不混用。
    const legacyWeekly = buildWeeklyStats(W1, { rating: "legacy", asOf, db });
    expect(legacyWeekly.ratingReport).toBeUndefined();
    const glickoWeekly = buildWeeklyStats(W1, { rating: "glicko2", asOf, db });
    expect(glickoWeekly.ratingReport?.model).toBe("glicko2");
  });
});
