import { describe, expect, it } from "vitest";
import { projectRatingView } from "./projections";
import { replayRatings } from "./replay";
import {
  appendCurrentEstimateRow,
  buildTrendRows,
  currentSegmentOverlay,
  displayRankRows,
  displayRatingRows,
  eventLocalDate,
  formatTrendSeasonLabel,
  listTrendSeasons,
  CURRENT_ESTIMATE_ROW_KEY,
} from "./chart-data";
import type { RatingViewPoint } from "./view-types";
import { replayConfig, replayMatch } from "../../../test/fixtures/ratings-scenarios";

const DIRECTORY = [
  { id: 1, name: "P1" },
  { id: 2, name: "P2" },
  { id: 3, name: "P3" },
  { id: 4, name: "P4" },
];

/** 跨季周夹具：2026-09-28 周被 10-01 季界拆成两段，Final → 重置 → 新段。 */
function crossSeasonView() {
  const replay = replayRatings(
    [
      replayMatch(1, "2026-09-28"),
      replayMatch(2, "2026-10-02", 17, 21),
      replayMatch(3, "2026-10-05"),
    ],
    [1, 2, 3, 4],
    { config: replayConfig, asOf: "2026-10-06T10:00:00+08:00" }
  );
  return projectRatingView(replay, DIRECTORY);
}

function point(partial: Partial<RatingViewPoint> & Pick<RatingViewPoint, "eventId" | "playerId" | "order">): RatingViewPoint {
  return {
    at: "2026-09-22",
    kind: "match_estimated",
    segment: "seg",
    season: "2026-07-01",
    r: 1000,
    status: "estimated",
    ...partial,
  };
}

describe("buildTrendRows", () => {
  it("跨季整周拆两段：Final 与重置同刻分两行不合并，order 升序", () => {
    const view = crossSeasonView();
    const rows = buildTrendRows(view.points, view.weekSegments);

    expect(rows.map((row) => row.kind)).toEqual([
      "match_estimated",
      "weekly_final",
      "season_reset",
      "match_estimated",
      "weekly_final",
      "match_estimated",
    ]);
    // 季界同一时间的两个值都存在（不合并、不丢行）
    expect(rows[1].at).toBe(rows[2].at);
    expect(rows[1].key).not.toBe(rows[2].key);
    // 同一事件一行：多人同事件合并到一行
    expect(rows[0].r).toEqual(
      Object.fromEntries(
        view.points
          .filter((p) => p.eventId === rows[0].key)
          .map((p) => [p.playerId, p.r])
      )
    );
  });

  it("weekly_final 行附周校准、season_reset 行附重置增量（取整）", () => {
    const view = crossSeasonView();
    const rows = buildTrendRows(view.points, view.weekSegments);

    const finalRow = rows.find((row) => row.kind === "weekly_final")!;
    const seg = view.weekSegments.find(
      (s) => s.segmentId === finalRow.segment
    )!;
    expect(finalRow.correction).toEqual(
      Object.fromEntries(
        Object.entries(seg.correction).map(([id, v]) => [Number(id), Math.round(v)])
      )
    );

    const resetRow = rows.find((row) => row.kind === "season_reset")!;
    const reset = view.weekSegments.find(
      (s) => s.reset?.eventId === resetRow.key
    )!.reset!;
    expect(resetRow.resetDeltas).toEqual(
      Object.fromEntries(reset.changes.map((c) => [c.playerId, Math.round(c.delta)]))
    );
  });

  it("空周仍有 Final 行；按赛季过滤", () => {
    const replay = replayRatings([replayMatch(1, "2026-09-28")], [1, 2, 3, 4], {
      config: replayConfig,
      asOf: "2026-10-12T10:00:00+08:00",
    });
    const view = projectRatingView(replay, DIRECTORY);
    const rows = buildTrendRows(view.points, view.weekSegments);
    const finalRows = rows.filter((row) => row.kind === "weekly_final");
    // 09-28（有比赛）与 10-05（空周）两段都结算
    expect(finalRows.length).toBeGreaterThanOrEqual(2);

    const seasonRows = buildTrendRows(view.points, view.weekSegments, {
      season: "2026-07-01",
    });
    expect(
      seasonRows.every((row) =>
        view.points.find(
          (p) => p.eventId === row.key && p.season === "2026-07-01"
        )
      )
    ).toBe(true);
    expect(seasonRows.length).toBeLessThan(rows.length);
  });

  it("零比赛：无点无行；不传 segments 时校准/重置为空对象", () => {
    const replay = replayRatings([], [1, 2, 3, 4], {
      config: replayConfig,
      asOf: "2026-10-06T10:00:00+08:00",
    });
    const view = projectRatingView(replay, DIRECTORY);
    expect(buildTrendRows(view.points, view.weekSegments)).toEqual([]);

    const withPoints = crossSeasonView();
    const rows = buildTrendRows(withPoints.points);
    expect(
      rows.find((row) => row.kind === "weekly_final")!.correction
    ).toEqual({});
    expect(rows.find((row) => row.kind === "season_reset")!.resetDeltas).toEqual(
      {}
    );
  });

  it("match_estimated 行附逐场每球员变化（取整）；其他 kind 为 {}", () => {
    const view = crossSeasonView();
    const rows = buildTrendRows(view.points, view.weekSegments);
    const estimatesById = new Map(
      view.weekSegments.flatMap((s) => s.matches).map((m) => [m.eventId, m])
    );
    for (const row of rows) {
      if (row.kind === "match_estimated") {
        const estimate = estimatesById.get(row.key)!;
        expect(row.matchDeltas).toEqual(
          Object.fromEntries(
            estimate.changes.map((c) => [c.playerId, Math.round(c.delta)])
          )
        );
      } else {
        expect(row.matchDeltas).toEqual({});
      }
    }
  });

  it("不传 segments 时 match_estimated 行的 matchDeltas 为 {}", () => {
    const view = crossSeasonView();
    const rows = buildTrendRows(view.points);
    expect(
      rows.find((row) => row.kind === "match_estimated")!.matchDeltas
    ).toEqual({});
  });
});

describe("display 行", () => {
  it("积分模式：缺席者沿用最近分值取整；未参赛者不出现（不虚构 1000 分）", () => {
    const rows = buildTrendRows([
      point({ eventId: "m:1", order: 0, playerId: 1, r: 1009.6 }),
      point({ eventId: "f:1", order: 1, kind: "weekly_final", at: "2026-09-28T16:00:00Z", playerId: 1, r: 1012.2, status: "final" }),
      // 下一周：1 缺席，2 首次参赛
      point({ eventId: "m:2", order: 2, playerId: 2, r: 998.4 }),
    ]);
    const display = displayRatingRows(rows);
    expect(display[0].values).toEqual({ 1: 1010 });
    expect(display[1].values).toEqual({ 1: 1012 });
    // 2 在首场比赛前不出现在任何行
    expect(display[0].values[2]).toBeUndefined();
    expect(display[1].values[2]).toBeUndefined();
    expect(display[2].values).toEqual({ 1: 1012, 2: 998 });
  });

  it("排名模式：展示整数并列用竞争排名 1,2,2,4；缺席者沿用分值参与排名", () => {
    const rows = buildTrendRows([
      point({ eventId: "m:1", order: 0, playerId: 1, r: 1009.6 }),
      point({ eventId: "m:1", order: 0, playerId: 2, r: 1009.5 }),
      point({ eventId: "m:1", order: 0, playerId: 3, r: 1008.4 }),
      // 1 缺席本周：2、3 继续参赛
      point({ eventId: "m:2", order: 1, playerId: 2, r: 1010.1 }),
      point({ eventId: "m:2", order: 1, playerId: 3, r: 1008.9 }),
    ]);
    const display = displayRankRows(rows);
    // 1009.6/1009.5 同为展示 1010 → 并列 1；1008.4 → 3
    expect(display[0].values).toEqual({ 1: 1, 2: 1, 3: 3 });
    // 1 沿用 1010 仍并列第一；2 → 1010 并列 1；3 → 1009 为第 3
    expect(display[1].values).toEqual({ 1: 1, 2: 1, 3: 3 });
    // 从未参赛者（4）不入榜
    expect(display[1].values[4]).toBeUndefined();
  });

  it("跨季：重置后的值参与后续排名", () => {
    const view = crossSeasonView();
    const rows = buildTrendRows(view.points, view.weekSegments);
    const display = displayRankRows(rows);
    const resetIndex = rows.findIndex((row) => row.kind === "season_reset");
    const afterReset = display[resetIndex + 1];
    const resetRow = rows[resetIndex];
    for (const [playerId, rank] of Object.entries(afterReset.values)) {
      expect(rank).toBeGreaterThanOrEqual(1);
      expect(resetRow.resetDeltas[Number(playerId)]).not.toBeUndefined();
    }
  });
});

describe("listTrendSeasons / formatTrendSeasonLabel", () => {
  it("赛季去重升序、排除 null", () => {
    const seasons = listTrendSeasons([
      point({ eventId: "a", order: 0, playerId: 1, season: "2026-10-01" }),
      point({ eventId: "b", order: 1, playerId: 1, season: null }),
      point({ eventId: "c", order: 2, playerId: 1, season: "2026-07-01" }),
      point({ eventId: "d", order: 3, playerId: 1, season: "2026-07-01" }),
    ]);
    expect(seasons).toEqual(["2026-07-01", "2026-10-01"]);
  });

  it("标签按自然季度", () => {
    expect(formatTrendSeasonLabel("2026-01-01")).toBe("2026年Q1");
    expect(formatTrendSeasonLabel("2026-04-01")).toBe("2026年Q2");
    expect(formatTrendSeasonLabel("2026-07-01")).toBe("2026年Q3");
    expect(formatTrendSeasonLabel("2026-10-01")).toBe("2026年Q4");
  });
});

describe("eventLocalDate", () => {
  it("比赛事实日期（YYYY-MM-DD）原样返回", () => {
    expect(eventLocalDate("2026-09-22")).toBe("2026-09-22");
  });

  it("ISO 瞬刻转本地日期（本地组件，非 UTC 切片）", () => {
    const at = "2026-10-05T16:00:00.000Z";
    const d = new Date(at);
    const expected = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    expect(eventLocalDate(at)).toBe(expected);
  });

  it("上海周界午夜在上海时区显示当天，而非 UTC 切片的前一天", () => {
    // 2026-10-06T00:00:00+08:00 = UTC 2026-10-05T16:00:00Z；slice(0,10) 会错一天
    const prev = process.env.TZ;
    process.env.TZ = "Asia/Shanghai";
    try {
      expect(eventLocalDate("2026-10-05T16:00:00.000Z")).toBe("2026-10-06");
    } finally {
      if (prev === undefined) delete process.env.TZ;
      else process.env.TZ = prev;
    }
  });
});

describe("currentSegmentOverlay", () => {
  it("当前区段预估行入集合，锚点为区段前最后一行（跨季周：新段预估锚在旧段 Final）", () => {
    const view = crossSeasonView();
    const rows = buildTrendRows(view.points, view.weekSegments);
    const currentSegmentId = view.weekSegments.at(-1)!.segmentId;
    const { estimatedKeys, anchorKey } = currentSegmentOverlay(
      rows,
      currentSegmentId
    );

    const lastRow = rows.at(-1)!;
    expect(lastRow.kind).toBe("match_estimated");
    expect(lastRow.segment).toBe(currentSegmentId);
    expect(estimatedKeys).toEqual(new Set([lastRow.key]));
    // 锚点 = 当前区段前的 weekly_final（实线停在此处，虚线从此接续）
    expect(anchorKey).toBe(rows.at(-2)!.key);
    expect(rows.at(-2)!.kind).toBe("weekly_final");
  });

  it("当前区段无预估行（未赛/无此段）：集合与锚点皆空，实线画到最后一行", () => {
    const view = crossSeasonView();
    const rows = buildTrendRows(view.points, view.weekSegments);
    const { estimatedKeys, anchorKey } = currentSegmentOverlay(
      rows,
      "seg:nonexistent"
    );
    expect(estimatedKeys.size).toBe(0);
    expect(anchorKey).toBeUndefined();
  });

  it("首个区段即当前（无历史）：预估行从第 0 行起，无锚点", () => {
    const rows = buildTrendRows([
      point({ eventId: "m:1", order: 0, playerId: 1, segment: "cur" }),
      point({ eventId: "m:2", order: 1, playerId: 2, segment: "cur" }),
    ]);
    const { estimatedKeys, anchorKey } = currentSegmentOverlay(rows, "cur");
    expect(estimatedKeys).toEqual(new Set(["m:1", "m:2"]));
    expect(anchorKey).toBeUndefined();
  });
});

describe("appendCurrentEstimateRow", () => {
  const NOW = "2026-10-04T12:00:00.000Z";
  const baseRows = buildTrendRows([
    point({ eventId: "m:1", order: 0, playerId: 1, r: 1009.6 }),
    point({
      eventId: "f:1",
      order: 1,
      kind: "weekly_final",
      at: "2026-09-28T16:00:00Z",
      playerId: 1,
      r: 1012.2,
      status: "final",
    }),
  ]);

  it("追加合成现在行：kind/segment/at/order/r 正确，deltas 为空", () => {
    const rows = appendCurrentEstimateRow(baseRows, {
      values: { 1: 1012, 2: 998 },
      currentSegmentId: "seg:cur",
      now: NOW,
    });
    expect(rows.length).toBe(baseRows.length + 1);
    const nowRow = rows.at(-1)!;
    expect(nowRow.key).toBe(`${CURRENT_ESTIMATE_ROW_KEY}seg:cur`);
    expect(nowRow.kind).toBe("match_estimated");
    expect(nowRow.segment).toBe("seg:cur");
    expect(nowRow.at).toBe(NOW);
    expect(nowRow.order).toBe(baseRows.at(-1)!.order + 1);
    expect(nowRow.r).toEqual({ 1: 1012, 2: 998 });
    expect(nowRow.correction).toEqual({});
    expect(nowRow.resetDeltas).toEqual({});
    expect(nowRow.matchDeltas).toEqual({});
    // 展示层函数照常消费：现在行进入沿用口径（与排行榜同口径取整分）
    expect(displayRatingRows(rows).at(-1)!.values).toEqual({ 1: 1012, 2: 998 });
  });

  it("零比赛周平接：现在行是当前区段唯一预估行，锚点为最后一个正式行", () => {
    const rows = appendCurrentEstimateRow(baseRows, {
      values: { 1: 1012 },
      currentSegmentId: "seg:cur",
      now: NOW,
    });
    const { estimatedKeys, anchorKey } = currentSegmentOverlay(rows, "seg:cur");
    expect(estimatedKeys).toEqual(new Set([`${CURRENT_ESTIMATE_ROW_KEY}seg:cur`]));
    expect(anchorKey).toBe("f:1");
  });

  it("空 rows 或空 values 不追加", () => {
    expect(
      appendCurrentEstimateRow([], {
        values: { 1: 1000 },
        currentSegmentId: "seg:cur",
        now: NOW,
      })
    ).toEqual([]);
    expect(
      appendCurrentEstimateRow(baseRows, {
        values: {},
        currentSegmentId: "seg:cur",
        now: NOW,
      })
    ).toEqual(baseRows);
  });

  it("不改输入数组（不可变）", () => {
    const frozen = Object.freeze([...baseRows]);
    const before = [...baseRows];
    appendCurrentEstimateRow(frozen, {
      values: { 1: 1012 },
      currentSegmentId: "seg:cur",
      now: NOW,
    });
    expect(baseRows).toEqual(before);
    expect(frozen).toEqual(before);
  });
});
