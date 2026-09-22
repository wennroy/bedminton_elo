import { describe, expect, it } from "vitest";
import { createRatingConfig } from "./config";
import { projectRatingView } from "./projections";
import { replayRatings } from "./replay";
import type {
  MatchEstimate,
  PlayerChange,
  RatingEvent,
  RatingMatch,
  RatingReplay,
  RatingSegment,
  RatingState,
  SeasonReset,
} from "./types";

const CONFIG = createRatingConfig({ firstSeasonStart: "2026-07-01" });

function state(r: number, rd = 120): RatingState {
  return { r, rd, volatility: 0.06 };
}

function segment(
  id: string,
  weekStart: string,
  startIso: string,
  endIso: string,
  seasonId: string | null,
  h: number
): RatingSegment {
  return { id, weekStart, seasonId, start: startIso, end: endIso, h };
}

function change(playerId: number, beforeR: number, afterR: number): PlayerChange {
  return {
    playerId,
    before: state(beforeR),
    after: state(afterR),
    delta: afterR - beforeR,
  };
}

function matchEvent(
  segmentId: string,
  matchId: number,
  playedAt: string,
  changes: PlayerChange[]
): MatchEstimate {
  return {
    kind: "match_estimated",
    eventId: `match_estimated:${segmentId}:${matchId}`,
    matchId,
    segmentId,
    playedAt,
    preWinA: 0.5,
    changes,
  };
}

function makeReplay(input: {
  currentSegment: RatingSegment;
  current: Record<string, RatingState>;
  lastFinal: Record<string, RatingState>;
  events: RatingEvent[];
  asOf?: string;
}): RatingReplay {
  const matchEstimates: Record<string, MatchEstimate> = {};
  for (const event of input.events) {
    if (event.kind === "match_estimated") {
      matchEstimates[String(event.matchId)] = event;
    }
  }
  return {
    asOf: input.asOf ?? "2026-10-04T20:00:00+08:00",
    configVersion: "test-config-version",
    currentSegment: input.currentSegment,
    current: input.current,
    lastFinal: input.lastFinal,
    events: input.events,
    matchEstimates,
    issues: [],
    nextBoundary: input.currentSegment.end,
  };
}

function realMatch(
  id: number,
  playedAt: string,
  teamA: readonly [number, number],
  teamB: readonly [number, number],
  scoreA = 21,
  scoreB = 15
): RatingMatch {
  return {
    id,
    playedAt,
    createdAt: `${playedAt}T10:00:00Z`,
    teamA,
    teamB,
    scoreA,
    scoreB,
  };
}

const DIRECTORY = [
  { id: 1, name: "P1" },
  { id: 2, name: "P2" },
  { id: 3, name: "P3" },
  { id: 4, name: "P4" },
  { id: 5, name: "P5" },
];

describe("projectRatingView 空历史", () => {
  it("未初始化/空历史：全部 unrated、空分数、无排名、峰值 null", () => {
    const replay = replayRatings([], [1, 2, 3], {
      config: CONFIG,
      asOf: "2026-09-23T20:00:00+08:00",
    });
    const view = projectRatingView(replay, DIRECTORY.slice(0, 3));

    expect(view.players).toHaveLength(3);
    for (const player of view.players) {
      expect(player).toEqual({
        playerId: player.playerId,
        name: player.name,
        r: null,
        displayRating: null,
        rank: null,
        status: "unrated",
        rd: null,
        lastFinal: null,
      });
    }
    expect(view.points).toEqual([]);
    expect(view.matchEstimatesById).toEqual({});
    expect(view.issues).toEqual([]);
    expect(view.peakFinal).toEqual({ "1": null, "2": null, "3": null });
    // 当前开放区段列出但不含任何事件。
    expect(view.weekSegments).toHaveLength(1);
    expect(view.weekSegments[0].matches).toEqual([]);
    expect(view.weekSegments[0].correction).toEqual({});
    expect(view.weekSegments[0].reset).toBeNull();
  });
});

describe("projectRatingView 排名", () => {
  it("并列按 displayRating 分组：竞争排名 1,2,2,4，unrated 无排名", () => {
    const seg = segment(
      "2026-09-21:2026-09-21",
      "2026-09-21",
      "2026-09-21T00:00:00+08:00",
      "2026-09-28T00:00:00+08:00",
      "2026-07-01",
      1
    );
    const replay = makeReplay({
      currentSegment: seg,
      current: {
        "1": state(1010.4),
        "2": state(1009.2),
        "3": state(1008.6),
        "4": state(1000.1),
      },
      lastFinal: {
        "1": state(1000),
        "2": state(1000),
        "3": state(1000),
        "4": state(1000),
      },
      events: [
        matchEvent(seg.id, 1, "2026-09-22", [
          change(1, 1000, 1010.4),
          change(2, 1000, 1009.2),
          change(3, 1000, 1008.6),
          change(4, 1000, 1000.1),
        ]),
      ],
      asOf: "2026-09-23T20:00:00+08:00",
    });
    const view = projectRatingView(replay, DIRECTORY);

    const byId = new Map(view.players.map((p) => [p.playerId, p]));
    expect(byId.get(1)).toMatchObject({ displayRating: 1010, rank: 1, status: "estimated" });
    expect(byId.get(2)).toMatchObject({ displayRating: 1009, rank: 2 });
    expect(byId.get(3)).toMatchObject({ displayRating: 1009, rank: 2 });
    expect(byId.get(4)).toMatchObject({ displayRating: 1000, rank: 4 });
    // 同 displayRating 并列名次相同。
    expect(byId.get(2)?.rank).toBe(byId.get(3)?.rank);
    // unrated 不参与排名。
    expect(byId.get(5)).toMatchObject({
      status: "unrated",
      rank: null,
      displayRating: null,
      r: null,
    });
    // 排名组内顺序：rated 按 displayRating 降序、ID 升序。
    const rated = view.players.filter((p) => p.rank !== null);
    expect(rated.map((p) => p.playerId)).toEqual([1, 2, 3, 4]);
  });
});

describe("projectRatingView 跨季周", () => {
  // 2026-09-28 这一周被 2026-10-01 季界拆成 3/7 与 4/7 两段。
  const seg1 = segment(
    "2026-09-28:2026-09-28",
    "2026-09-28",
    "2026-09-28T00:00:00+08:00",
    "2026-10-01T00:00:00+08:00",
    "2026-07-01",
    3 / 7
  );
  const seg2 = segment(
    "2026-10-01:2026-10-01",
    "2026-09-28",
    "2026-10-01T00:00:00+08:00",
    "2026-10-05T00:00:00+08:00",
    "2026-10-01",
    4 / 7
  );
  const finalEvent = {
    kind: "weekly_final" as const,
    eventId: "weekly_final:2026-09-28:2026-09-28",
    segment: seg1,
    start: { "1": state(1000), "2": state(1000), "3": state(1000), "4": state(1000) },
    estimatedEnd: { "1": state(1180), "2": state(1000), "3": state(1000), "4": state(1000) },
    final: { "1": state(1200), "2": state(1000), "3": state(1000), "4": state(1000) },
    correction: { "1": 20, "2": 0, "3": 0, "4": 0 },
  };
  const resetEvent: SeasonReset = {
    kind: "season_reset",
    eventId: "season_reset:2026-10-01",
    at: seg1.end,
    seasonId: "2026-10-01",
    changes: [change(1, 1200, 1175), change(2, 1000, 1000)],
  };
  const replay = makeReplay({
    currentSegment: seg2,
    current: { "1": state(1190), "2": state(1000), "3": state(1000), "4": state(1000) },
    lastFinal: { "1": state(1175), "2": state(1000), "3": state(1000), "4": state(1000) },
    events: [
      matchEvent(seg1.id, 1, "2026-09-29", [change(1, 1000, 1180)]),
      finalEvent,
      resetEvent,
      matchEvent(seg2.id, 2, "2026-10-02", [change(1, 1175, 1190)]),
    ],
  });

  it("跨季周拆成两段与重置分列：软重置不当战绩下跌", () => {
    const view = projectRatingView(replay, DIRECTORY);

    expect(view.weekSegments).toHaveLength(2);
    const [row1, row2] = view.weekSegments;
    // 两段同一自然周，但区段、赛季分开。
    expect(row1.segmentId).toBe(seg1.id);
    expect(row1.weekStart).toBe("2026-09-28");
    expect(row1.seasonId).toBe("2026-07-01");
    expect(row2.segmentId).toBe(seg2.id);
    expect(row2.weekStart).toBe("2026-09-28");
    expect(row2.seasonId).toBe("2026-10-01");

    // 比赛变化与周校准在各自区段。
    expect(row1.matches).toHaveLength(1);
    expect(row1.matches[0].matchId).toBe(1);
    expect(row1.correction["1"]).toBe(20);
    expect(row2.matches).toHaveLength(1);
    expect(row2.matches[0].matchId).toBe(2);
    expect(row2.correction).toEqual({});

    // 重置只挂在触发它的第一段，delta=-25 不混入比赛变化或校准。
    expect(row1.reset).not.toBeNull();
    expect(row1.reset?.seasonId).toBe("2026-10-01");
    expect(row1.reset?.changes.find((c) => c.playerId === 1)?.delta).toBe(-25);
    expect(row2.reset).toBeNull();
    expect(row1.correction["1"]).not.toBe(20 - 25);

    // 该周比赛表现 = 该段逐场变化 + 周校准；重置单列。
    const weekPerformance =
      row1.matches[0].changes.find((c) => c.playerId === 1)!.delta +
      row1.correction["1"];
    expect(weekPerformance).toBe(180 + 20);
  });

  it("Final→reset 同刻两个图表点共存，不合并、顺序保留", () => {
    const view = projectRatingView(replay, DIRECTORY);

    const p1AtBoundary = view.points.filter(
      (point) => point.playerId === 1 && point.at === seg1.end
    );
    expect(p1AtBoundary).toHaveLength(2);
    const finalPoint = p1AtBoundary.find((p) => p.kind === "weekly_final");
    const resetPoint = p1AtBoundary.find((p) => p.kind === "season_reset");
    expect(finalPoint).toMatchObject({
      r: 1200,
      status: "final",
      eventId: finalEvent.eventId,
    });
    expect(resetPoint).toMatchObject({
      r: 1175,
      status: "final",
      eventId: resetEvent.eventId,
    });
    expect(finalPoint?.eventId).not.toBe(resetPoint?.eventId);
    expect(resetPoint?.order).toBeGreaterThan(finalPoint?.order ?? 0);
  });
});

describe("projectRatingView 历史与新人", () => {
  // 第 1 周（2026-09-21）：1,2,3,4 参赛；第 2 周（2026-09-28）：1,2,3,5 参赛。
  const matches: RatingMatch[] = [
    realMatch(1, "2026-09-21", [1, 2], [3, 4]),
    realMatch(2, "2026-09-28", [1, 2], [3, 5]),
  ];
  const replay = replayRatings(matches, [1, 2, 3, 4, 5], {
    config: CONFIG,
    asOf: "2026-09-30T20:00:00+08:00",
  });

  it("历史比赛变化仍标 Estimated，最后正式值与当前预估都可取", () => {
    const view = projectRatingView(replay, DIRECTORY);

    expect(view.matchEstimatesById["1"]?.kind).toBe("match_estimated");
    expect(view.matchEstimatesById["1"]?.matchId).toBe(1);
    expect(view.matchEstimatesById["1"]?.segmentId).toBe("2026-09-21:2026-09-21");
    expect(view.matchEstimatesById["2"]?.kind).toBe("match_estimated");

    const byId = new Map(view.players.map((p) => [p.playerId, p]));
    // 本周缺席的 4 号：status=final，显示 lastFinal（不自动丢分、不显示本周预估）。
    expect(byId.get(4)?.status).toBe("final");
    expect(byId.get(4)?.r).toBe(replay.lastFinal["4"].r);
    expect(byId.get(4)?.lastFinal).toBe(Math.round(replay.lastFinal["4"].r));
    // 本周参赛的 1 号：显示当前预估。
    expect(byId.get(1)?.status).toBe("estimated");
    const current = replay.current["1"];
    expect(byId.get(1)?.r).toBe(current.r);
    expect(byId.get(1)?.r).not.toBe(byId.get(1)?.lastFinal);

    // 4 号的最近点是第 1 周 Final；1 号的最近点是第 2 周逐场预估。
    const latestPointOf = (playerId: number) =>
      view.points
        .filter((point) => point.playerId === playerId)
        .reduce((a, b) => (b.order > a.order ? b : a));
    expect(latestPointOf(4)).toMatchObject({
      kind: "weekly_final",
      segment: "2026-09-21:2026-09-21",
      status: "final",
    });
    expect(latestPointOf(1)).toMatchObject({
      kind: "match_estimated",
      segment: "2026-09-28:2026-09-28",
      status: "estimated",
    });
  });

  it("新人不补造参赛前历史线：第一个点就是首场事件", () => {
    const view = projectRatingView(replay, DIRECTORY);

    const p5Points = view.points
      .filter((point) => point.playerId === 5)
      .sort((a, b) => a.order - b.order);
    expect(p5Points.length).toBeGreaterThan(0);
    expect(p5Points[0].kind).toBe("match_estimated");
    expect(p5Points[0].eventId).toBe("match_estimated:2026-09-28:2026-09-28:2");
    // 第 1 周的 Final 不含 5 号（其首次比赛之前没有任何点）。
    expect(
      view.points.some(
        (point) => point.playerId === 5 && point.segment === "2026-09-21:2026-09-21"
      )
    ).toBe(false);
    const byId = new Map(view.players.map((p) => [p.playerId, p]));
    // 5 号在本区段首次参赛：引擎已在段初为其建立正式初值基准（1000），
    // 该基准随 lastFinal 透出；其首个图表点仍是首场逐场事件。
    expect(byId.get(5)?.lastFinal).toBe(1000);
    expect(byId.get(5)?.status).toBe("estimated");
  });

  it("一致性：排行榜/档案/图表从同一投影取数，数值与名次一致", () => {
    const view = projectRatingView(replay, DIRECTORY);

    // 图表：每人的最新点 r 与 players 行 r 一致。
    for (const player of view.players) {
      if (player.status === "unrated") continue;
      const latest = view.points
        .filter((point) => point.playerId === player.playerId)
        .reduce((a, b) => (b.order > a.order ? b : a));
      expect(latest.r).toBe(player.r);
    }

    // 并列规则：同 displayRating ⇒ 同 rank；rank 为竞争排名（1,2,2,4）。
    const rated = view.players.filter((p) => p.rank !== null);
    for (const a of rated) {
      for (const b of rated) {
        if (a.displayRating === b.displayRating) expect(a.rank).toBe(b.rank);
      }
    }
    const sorted = [...rated].sort(
      (a, b) => (b.displayRating ?? 0) - (a.displayRating ?? 0) || a.playerId - b.playerId
    );
    for (let i = 0; i < sorted.length; i++) {
      let groupStart = i;
      while (
        groupStart > 0 &&
        sorted[groupStart - 1].displayRating === sorted[i].displayRating
      ) {
        groupStart -= 1;
      }
      expect(sorted[i].rank).toBe(groupStart + 1);
    }
  });
});

describe("projectRatingView 峰值", () => {
  it("正式峰值只取 PeriodFinal.final，不含重置前后值与 Estimated", () => {
    const seg = segment(
      "2026-09-21:2026-09-21",
      "2026-09-21",
      "2026-09-21T00:00:00+08:00",
      "2026-09-28T00:00:00+08:00",
      "2026-07-01",
      1
    );
    const seg2 = segment(
      "2026-09-28:2026-09-28",
      "2026-09-28",
      "2026-09-28T00:00:00+08:00",
      "2026-10-05T00:00:00+08:00",
      "2026-07-01",
      1
    );
    const finalEvent = {
      kind: "weekly_final" as const,
      eventId: "weekly_final:2026-09-21:2026-09-21",
      segment: seg,
      start: { "1": state(1000), "2": state(1000) },
      estimatedEnd: { "1": state(1250), "2": state(1000) },
      final: { "1": state(1300), "2": state(900) },
      correction: { "1": 50, "2": 0 },
    };
    const resetEvent: SeasonReset = {
      kind: "season_reset",
      eventId: "season_reset:2026-10-01",
      at: "2026-10-01T00:00:00+08:00",
      seasonId: "2026-10-01",
      changes: [change(1, 1300, 1225)],
    };
    const replay = makeReplay({
      currentSegment: seg2,
      current: { "1": state(1400), "2": state(1000) },
      lastFinal: { "1": state(1225), "2": state(900) },
      events: [
        matchEvent(seg.id, 1, "2026-09-22", [change(1, 1000, 1250)]),
        finalEvent,
        resetEvent,
        matchEvent(seg2.id, 2, "2026-09-29", [change(1, 1225, 1400)]),
      ],
      asOf: "2026-09-30T20:00:00+08:00",
    });
    const view = projectRatingView(replay, DIRECTORY.slice(0, 3));

    // 峰值 1300 来自 weekly_final.final，不是重置后的 1225，也不是 Estimated 的 1400。
    expect(view.peakFinal["1"]).toEqual({
      r: 1300,
      eventId: finalEvent.eventId,
      at: seg.end,
    });
    expect(view.peakFinal["2"]).toEqual({
      r: 900,
      eventId: finalEvent.eventId,
      at: seg.end,
    });
    // 3 号从未有 Final → null。
    expect(view.peakFinal["3"]).toBeNull();
  });
});
