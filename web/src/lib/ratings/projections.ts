import type {
  RatingReplay,
  RatingSegment,
  SeasonReset,
} from "./types";
import type {
  RatingView,
  RatingViewPlayer,
  RatingViewPoint,
  RatingViewReset,
  RatingViewWeekSegment,
} from "./view-types";

/**
 * 把 RatingReplay 投影成只读展示 DTO。只消费 replay 与球员目录，
 * 不自行重放、不读时间；排行榜、档案与图表从同一份结果取数。
 */
export function projectRatingView(
  replay: RatingReplay,
  players: ReadonlyArray<{ id: number; name: string }>
): RatingView {
  // 区段索引：match_estimated 只带 segmentId，周 Final 事件带完整区段。
  const segmentById = new Map<string, RatingSegment>();
  segmentById.set(replay.currentSegment.id, replay.currentSegment);
  for (const event of replay.events) {
    if (event.kind === "weekly_final") {
      segmentById.set(event.segment.id, event.segment);
    }
  }
  const segmentByEnd = new Map<string, RatingSegment>();
  for (const segment of segmentById.values()) {
    segmentByEnd.set(segment.end, segment);
  }

  // 周区段行：已结束区段来自 weekly_final 事件；当前区段（进行中）补在末尾。
  const segmentRows = new Map<string, RatingViewWeekSegment>();
  const weekSegments: RatingViewWeekSegment[] = [];
  for (const event of replay.events) {
    if (event.kind !== "weekly_final") continue;
    const row: RatingViewWeekSegment = {
      segmentId: event.segment.id,
      weekStart: event.segment.weekStart,
      seasonId: event.segment.seasonId,
      start: event.segment.start,
      end: event.segment.end,
      h: event.segment.h,
      matches: [],
      correction: { ...event.correction },
      reset: null,
    };
    segmentRows.set(event.segment.id, row);
    weekSegments.push(row);
  }
  if (!segmentRows.has(replay.currentSegment.id)) {
    const current = replay.currentSegment;
    const row: RatingViewWeekSegment = {
      segmentId: current.id,
      weekStart: current.weekStart,
      seasonId: current.seasonId,
      start: current.start,
      end: current.end,
      h: current.h,
      matches: [],
      correction: {},
      reset: null,
    };
    segmentRows.set(current.id, row);
    weekSegments.push(row);
  }

  // 本周参赛者：出现在当前区段 match_estimated 事件里的球员。
  const thisWeekPlayerIds = new Set<number>();
  for (const event of replay.events) {
    if (
      event.kind === "match_estimated" &&
      event.segmentId === replay.currentSegment.id
    ) {
      for (const change of event.changes) thisWeekPlayerIds.add(change.playerId);
    }
  }

  const points: RatingViewPoint[] = [];
  replay.events.forEach((event, order) => {
    if (event.kind === "match_estimated") {
      const segment =
        segmentById.get(event.segmentId) ?? replay.currentSegment;
      for (const change of event.changes) {
        points.push({
          eventId: event.eventId,
          playerId: change.playerId,
          at: event.playedAt,
          order,
          kind: "match_estimated",
          segment: event.segmentId,
          season: segment.seasonId,
          r: change.after.r,
          status: "estimated",
        });
      }
      segmentRows.get(event.segmentId)?.matches.push(event);
      return;
    }
    if (event.kind === "weekly_final") {
      for (const [playerId, state] of Object.entries(event.final)) {
        points.push({
          eventId: event.eventId,
          playerId: Number(playerId),
          at: event.segment.end,
          order,
          kind: "weekly_final",
          segment: event.segment.id,
          season: event.segment.seasonId,
          r: state.r,
          status: "final",
        });
      }
      return;
    }
    // season_reset：同刻与周 Final 分列为两个点，重置只挂在触发它的区段行。
    const trigger = segmentByEnd.get(event.at);
    for (const change of event.changes) {
      points.push({
        eventId: event.eventId,
        playerId: change.playerId,
        at: event.at,
        order,
        kind: "season_reset",
        segment: trigger?.id ?? "",
        season: event.seasonId,
        r: change.after.r,
        status: "final",
      });
    }
    const row = trigger === undefined ? undefined : segmentRows.get(trigger.id);
    if (row !== undefined) row.reset = toResetView(event);
  });

  const viewPlayers = projectPlayers(replay, players, thisWeekPlayerIds);

  const peakFinal: Record<string, RatingView["peakFinal"][string]> = {};
  for (const player of players) peakFinal[String(player.id)] = null;
  for (const event of replay.events) {
    if (event.kind !== "weekly_final") continue;
    for (const [playerId, state] of Object.entries(event.final)) {
      const existing = peakFinal[playerId];
      // 正式峰值只取 PeriodFinal.final 的赛末值；重置前后值与 Estimated 不参与。
      if (existing === null || state.r > existing.r) {
        peakFinal[playerId] = { r: state.r, eventId: event.eventId, at: event.segment.end };
      }
    }
  }

  return {
    players: viewPlayers,
    points,
    weekSegments,
    matchEstimatesById: { ...replay.matchEstimates },
    issues: replay.issues,
    peakFinal,
  };
}

function projectPlayers(
  replay: RatingReplay,
  players: ReadonlyArray<{ id: number; name: string }>,
  thisWeekPlayerIds: ReadonlySet<number>
): RatingViewPlayer[] {
  const rows: RatingViewPlayer[] = players.map((player) => {
    const key = String(player.id);
    const current = replay.current[key];
    if (current === undefined) {
      // 未参赛：不补造 1000 分历史线，空分数、无排名。
      return {
        playerId: player.id,
        name: player.name,
        r: null,
        displayRating: null,
        rank: null,
        status: "unrated",
        rd: null,
        lastFinal: null,
      };
    }
    const participatedThisWeek = thisWeekPlayerIds.has(player.id);
    // 缺席者不自动丢分：显示最近一次 Final；本周参赛者显示当前预估。
    const lastFinal = replay.lastFinal[key];
    const r = participatedThisWeek ? current.r : (lastFinal?.r ?? current.r);
    const rd = participatedThisWeek ? current.rd : (lastFinal?.rd ?? current.rd);
    return {
      playerId: player.id,
      name: player.name,
      r,
      displayRating: Math.round(r),
      rank: null,
      status: participatedThisWeek ? "estimated" : "final",
      rd,
      lastFinal: lastFinal === undefined ? null : Math.round(lastFinal.r),
    };
  });

  // 排名：按 displayRating 分组并列、竞争排名 1,2,2,4；
  // 组内仅按 ID 稳定排序，不用出勤破同分；unrated 不参与。
  const rated = rows
    .filter((row) => row.status !== "unrated")
    .sort(
      (a, b) =>
        (b.displayRating ?? 0) - (a.displayRating ?? 0) ||
        a.playerId - b.playerId
    );
  let index = 0;
  while (index < rated.length) {
    let end = index + 1;
    while (
      end < rated.length &&
      rated[end].displayRating === rated[index].displayRating
    ) {
      end += 1;
    }
    const rank = index + 1;
    for (let i = index; i < end; i++) rated[i].rank = rank;
    index = end;
  }
  return rows;
}

function toResetView(event: SeasonReset): RatingViewReset {
  return {
    eventId: event.eventId,
    at: event.at,
    seasonId: event.seasonId,
    changes: event.changes.map((change) => ({
      playerId: change.playerId,
      before: { ...change.before },
      after: { ...change.after },
      delta: change.delta,
    })),
  };
}
