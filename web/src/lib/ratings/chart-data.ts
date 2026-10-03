import type { LocalDate } from "./types";
import type { RatingViewPoint, RatingViewWeekSegment } from "./view-types";

/**
 * 全员曲线纯函数：消费 projectRatingView 的 points（+ weekSegments 的
 * 校准/重置细节），产出图表行与展示值。不读 DB、不读时间、不再次结算。
 *
 * 口径：
 * - 同一事件一行（eventId 分组，order 排序）；同刻 weekly_final →
 *   season_reset 是两个点，不合并。
 * - 正式线 = weekly_final + season_reset；当前区段预估线 = 当前区段的
 *   match_estimated（由组件按 segment === currentSegmentId 过滤）。
 * - 没有参赛前不虚构 1000 分线：values 只含已有评分事件的球员。
 */

export type TrendRowKind = RatingViewPoint["kind"];

/** 图表行：同一事件一行；values 由 display*Rows 填充。 */
export interface TrendRow {
  /** 事件 ID（eventId），行唯一键。 */
  key: string;
  /** 事件时点：比赛为 YYYY-MM-DD；结算/重置为 ISO。 */
  at: string;
  /** replay.events 内的事件顺序（升序）。 */
  order: number;
  kind: TrendRowKind;
  /** 所属评分区段 ID。 */
  segment: string;
  /** 该事件后有评分的球员原始实力分（key=playerId）。 */
  r: Record<number, number>;
  /** weekly_final：周校准 final - estimatedEnd（展示取整）；其他 kind 为 {}。 */
  correction: Record<number, number>;
  /** season_reset：每球员重置增量（展示取整）；其他 kind 为 {}。 */
  resetDeltas: Record<number, number>;
  /** match_estimated：该场每球员单场预估变化（展示取整）；其他 kind 为 {}。 */
  matchDeltas: Record<number, number>;
}

export interface BuildTrendRowsOptions {
  /**
   * 赛季过滤（points[].season）：undefined = 全部历史；
   * null 只保留无赛季点（首季起点前的区段，正常重放下不存在）。
   */
  season?: LocalDate | null;
}

/**
 * 把投影点整理成按事件分组的图表行（order 升序）。
 * 传入 weekSegments 时给 weekly_final 行附周校准、season_reset 行附重置增量、
 * match_estimated 行附逐场每球员变化，供独立 tooltip 使用。
 */
export function buildTrendRows(
  points: readonly RatingViewPoint[],
  segments?: readonly RatingViewWeekSegment[],
  options: BuildTrendRowsOptions = {}
): TrendRow[] {
  const filtered =
    options.season === undefined
      ? points
      : points.filter((point) => point.season === options.season);
  const sorted = [...filtered].sort(
    (a, b) => a.order - b.order || a.eventId.localeCompare(b.eventId)
  );

  const correctionBySegment = new Map<string, Record<number, number>>();
  const resetByEventId = new Map<string, Record<number, number>>();
  const matchDeltasByEventId = new Map<string, Record<number, number>>();
  for (const segment of segments ?? []) {
    const correction: Record<number, number> = {};
    for (const [playerId, value] of Object.entries(segment.correction)) {
      correction[Number(playerId)] = Math.round(value);
    }
    correctionBySegment.set(segment.segmentId, correction);
    if (segment.reset !== null) {
      const deltas: Record<number, number> = {};
      for (const change of segment.reset.changes) {
        deltas[change.playerId] = Math.round(change.delta);
      }
      resetByEventId.set(segment.reset.eventId, deltas);
    }
    for (const match of segment.matches) {
      const deltas: Record<number, number> = {};
      for (const change of match.changes) {
        deltas[change.playerId] = Math.round(change.delta);
      }
      matchDeltasByEventId.set(match.eventId, deltas);
    }
  }

  const rows = new Map<string, TrendRow>();
  for (const point of sorted) {
    let row = rows.get(point.eventId);
    if (row === undefined) {
      row = {
        key: point.eventId,
        at: point.at,
        order: point.order,
        kind: point.kind,
        segment: point.segment,
        r: {},
        correction:
          point.kind === "weekly_final"
            ? (correctionBySegment.get(point.segment) ?? {})
            : {},
        resetDeltas:
          point.kind === "season_reset"
            ? (resetByEventId.get(point.eventId) ?? {})
            : {},
        matchDeltas:
          point.kind === "match_estimated"
            ? (matchDeltasByEventId.get(point.eventId) ?? {})
            : {},
      };
      rows.set(point.eventId, row);
    }
    row.r[point.playerId] = point.r;
  }
  return [...rows.values()].sort((a, b) => a.order - b.order);
}

/**
 * 当前区段（未结算）虚线平接的合成「现在」行的 key 前缀。
 * replay 事件 eventId 形如 "match_estimated:<段>:<场>"/"weekly_final:<段>"/
 * "season_reset:<季>"（段/季 ID 只含日期、时间与 ":"），不含 "@"，故不会碰撞。
 */
export const CURRENT_ESTIMATE_ROW_KEY = "current-estimate@";

/**
 * 展示层平接：向行尾追加合成的「现在」行，让当前区段预估虚线总是接到今天
 * （本周已打时也从最后一场延伸到今天；本周零比赛时从锚点行平接）。
 * 纯展示合成——不写入投影 DTO、不进 OG/周报/指纹。kind 复用
 * match_estimated 是为沿用 currentSegmentOverlay 的锚点接续与读数
 * 「预估」语义（currentSegmentOverlay 不需要改）；matchDeltas 为空，
 * tooltip 自然不弹出。
 *
 * rows 为空（赛季无比赛/全部未评级）或 values 为空（未评级者不出现）
 * 时不追加。不改输入数组。
 */
export function appendCurrentEstimateRow(
  rows: readonly TrendRow[],
  options: {
    /** 与排行榜同口径的当前展示分（key=playerId，displayRating；未评级者不出现）。 */
    values: Readonly<Record<number, number>>;
    currentSegmentId: string;
    /** 服务端注入的当前时点（ISO，stale 时如实为旧成功时点）。 */
    now: string;
  }
): TrendRow[] {
  const { values, currentSegmentId, now } = options;
  const last = rows.at(-1);
  if (last === undefined || Object.keys(values).length === 0) return [...rows];
  return [
    ...rows,
    {
      key: `${CURRENT_ESTIMATE_ROW_KEY}${currentSegmentId}`,
      at: now,
      order: last.order + 1,
      kind: "match_estimated",
      segment: currentSegmentId,
      r: { ...values },
      correction: {},
      resetDeltas: {},
      matchDeltas: {},
    },
  ];
}

/** 图表展示行：values 为每球员展示值（积分=取整分；排名=并列名次）。 */
export interface DisplayTrendRow extends TrendRow {
  values: Record<number, number>;
}

/**
 * 积分模式：缺席者沿用最近一次已知实力分参与连线（与「缺席不丢分」一致），
 * 展示取整；从未有评分事件者不出现（不虚构 1000 分）。
 */
export function displayRatingRows(
  rows: readonly TrendRow[]
): DisplayTrendRow[] {
  const carried = new Map<number, number>();
  return rows.map((row) => {
    for (const [playerId, r] of Object.entries(row.r)) {
      carried.set(Number(playerId), r);
    }
    const values: Record<number, number> = {};
    for (const [playerId, r] of carried) values[playerId] = Math.round(r);
    return { ...row, values };
  });
}

/**
 * 排名模式：按展示整数分分组并列、竞争排名（1,2,2,4）；缺席者沿用上次
 * 分值参与排名；从未有评分事件者不入榜。与排行榜并列口径一致。
 */
export function displayRankRows(rows: readonly TrendRow[]): DisplayTrendRow[] {
  const carried = new Map<number, number>();
  return rows.map((row) => {
    for (const [playerId, r] of Object.entries(row.r)) {
      carried.set(Number(playerId), r);
    }
    const ordered = [...carried.entries()]
      .map(([playerId, r]) => ({ playerId, display: Math.round(r) }))
      .sort((a, b) => b.display - a.display || a.playerId - b.playerId);
    const values: Record<number, number> = {};
    let index = 0;
    while (index < ordered.length) {
      let end = index + 1;
      while (
        end < ordered.length &&
        ordered[end].display === ordered[index].display
      ) {
        end += 1;
      }
      const rank = index + 1;
      for (let i = index; i < end; i++) values[ordered[i].playerId] = rank;
      index = end;
    }
    return { ...row, values };
  });
}

/** 点中出现的赛季列表（去重升序，排除 null）。 */
export function listTrendSeasons(
  points: readonly RatingViewPoint[]
): LocalDate[] {
  const seasons = new Set<LocalDate>();
  for (const point of points) {
    if (point.season !== null) seasons.add(point.season);
  }
  return [...seasons].sort();
}

/** 赛季标签："2026-10-01" → "2026年Q4"。 */
export function formatTrendSeasonLabel(season: LocalDate): string {
  const quarter = Math.ceil(Number(season.slice(5, 7)) / 3);
  return `${season.slice(0, 4)}年Q${quarter}`;
}

/**
 * 事件时点 → 本地日期：比赛事实日期（YYYY-MM-DD）原样返回；ISO 瞬刻
 * 转本地日期——周界 Final/赛季重置是上海午夜，取 UTC 日期（slice(0,10)）
 * 在上海时区会显示成前一天。
 */
export function eventLocalDate(at: string): LocalDate {
  if (at.length === 10) return at;
  const d = new Date(at);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/**
 * 当前区段（尚未结算）的实/虚线口径：当前区段的逐场预估行上实线停笔
 * （正式线只到最后一个 weekly_final/season_reset，不与预估虚线同点重叠），
 * 虚线从锚点行——当前区段第一个预估行之前的最后一行——接续，视觉不断线。
 * 当前区段没有预估行时两者皆空，实线画到最后一行。
 */
export function currentSegmentOverlay(
  rows: readonly TrendRow[],
  currentSegmentId: string
): { estimatedKeys: ReadonlySet<string>; anchorKey: string | undefined } {
  const estimatedKeys = new Set<string>();
  let firstIndex: number | undefined;
  rows.forEach((row, index) => {
    if (row.kind === "match_estimated" && row.segment === currentSegmentId) {
      estimatedKeys.add(row.key);
      firstIndex = firstIndex ?? index;
    }
  });
  const anchorKey =
    firstIndex !== undefined && firstIndex > 0
      ? rows[firstIndex - 1].key
      : undefined;
  return { estimatedKeys, anchorKey };
}

/**
 * 当前区段预估虚线（`:est`）的逐行取值：虚线是每人「本周预估」的连续线——
 * 参赛场次按本场值、缺席场次沿用最近分值平线贯穿（与 displayRatingRows/
 * displayRankRows 的沿用口径、读数栏「缺席沿用最近分值」一致），Recharts
 * 不因 undefined 断线。
 *
 * - 预估行（estimatedKeys 含该行 key）：整行 `{...row.values}`——display 行
 *   已把沿用值带进 values，参赛与未参赛者都有；
 * - 锚点行（anchorKey）：只取首个预估行 values 的键集在锚点行的沿用值，让
 *   虚线从实线停笔处接续不断线；本周首次参赛者在锚点行无沿用值自然跳过
 *   （不虚构早期历史，其虚线从首个事件才开始）；
 * - 其他行：`{}`（实线行不掺 `:est`）。
 *
 * 没有预估行时返回空 Map（无锚点可接）。不改输入数组与行。
 */
export function estimatedLineValues(
  rows: readonly DisplayTrendRow[],
  estimatedKeys: ReadonlySet<string>,
  anchorKey: string | undefined
): Map<string, Record<number, number>> {
  const byRowKey = new Map<string, Record<number, number>>();
  const firstEstimated = rows.find((row) => estimatedKeys.has(row.key));
  if (firstEstimated === undefined) return byRowKey;
  const anchorIds = Object.keys(firstEstimated.values).map(Number);
  for (const row of rows) {
    if (estimatedKeys.has(row.key)) {
      byRowKey.set(row.key, { ...row.values });
    } else if (row.key === anchorKey) {
      const values: Record<number, number> = {};
      for (const playerId of anchorIds) {
        const value = row.values[playerId];
        if (value !== undefined) values[playerId] = value;
      }
      byRowKey.set(row.key, values);
    } else {
      byRowKey.set(row.key, {});
    }
  }
  return byRowKey;
}
