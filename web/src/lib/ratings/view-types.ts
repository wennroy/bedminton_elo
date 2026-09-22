import type {
  LocalDate,
  MatchEstimate,
  RatingIssue,
  RatingStatus,
} from "./types";

/**
 * 展示投影 DTO（可序列化，不 import DB/Next/环境）。
 * projectRatingView 的输出形状：排行榜、个人档案、图表与周区段
 * 全部从同一份 RatingView 取数，保证数值与名次一致。
 */

/** 排行榜/档案行：每人一条。未参赛者为 unrated、空分数、无排名。 */
export interface RatingViewPlayer {
  playerId: number;
  name: string;
  /** 原始浮点实力分；unrated 为 null。 */
  r: number | null;
  /** 展示取整分；unrated 为 null。并列排名按此值分组。 */
  displayRating: number | null;
  /** 竞争排名（1,2,2,4）；unrated 不参与排名，为 null。 */
  rank: number | null;
  status: RatingStatus;
  /** 当前状态对应的不确定性；unrated 为 null。 */
  rd: number | null;
  /** 最近一次周 Final 的展示值（取整）；从未结算过为 null。 */
  lastFinal: number | null;
}

/** 图表点状态：逐场变化是预估，周结算与季重置是正式值。 */
export type RatingPointStatus = "estimated" | "final";

export type RatingPointKind = "match_estimated" | "weekly_final" | "season_reset";

/**
 * 图表点（eventId + playerId 唯一）。同一时刻的 weekly_final →
 * season_reset 是两个点，不合并；order 保留 replay.events 顺序。
 */
export interface RatingViewPoint {
  eventId: string;
  playerId: number;
  /** 比赛事件用比赛事实日期（YYYY-MM-DD）；结算/重置用事件 ISO 时点。 */
  at: string;
  /** replay.events 内的事件顺序。 */
  order: number;
  kind: RatingPointKind;
  /** 所属评分区段 ID。 */
  segment: string;
  /** 所属赛季（季度起点）；区段在首赛季起点之前为 null。 */
  season: LocalDate | null;
  /** 该事件后的实力分（原始值）。 */
  r: number;
  status: RatingPointStatus;
}

/** 季界重置事件在周区段列表中的独立呈现，不混入比赛表现。 */
export interface RatingViewReset {
  eventId: string;
  at: string;
  seasonId: LocalDate;
  changes: ReadonlyArray<{
    playerId: number;
    before: { r: number; rd: number; volatility: number };
    after: { r: number; rd: number; volatility: number };
    delta: number;
  }>;
}

/**
 * 周区段投影：跨季周被拆成的两段与重置分开列出——
 * matches 是该段逐场 Estimated 变化，correction 是周结算校准
 * （PeriodFinal.final - estimatedEnd），reset 只挂在触发重置的那一段。
 */
export interface RatingViewWeekSegment {
  segmentId: string;
  weekStart: LocalDate;
  seasonId: LocalDate | null;
  start: string;
  end: string;
  h: number;
  /** 该段逐场 Estimated 变化（按重放顺序）。 */
  matches: MatchEstimate[];
  correction: Record<string, number>;
  reset: RatingViewReset | null;
}

/** 正式峰值：只取 PeriodFinal.final 的赛末值，不含重置前后值与 Estimated。 */
export interface RatingViewPeak {
  /** 峰值原始实力分。 */
  r: number;
  eventId: string;
  /** 达成峰值的周 Final 时点（区段 end）。 */
  at: string;
}

/** 统一展示投影。 */
export interface RatingView {
  players: RatingViewPlayer[];
  points: RatingViewPoint[];
  weekSegments: RatingViewWeekSegment[];
  /** 按 matchId 传递；历史比赛变化仍标 match_estimated（Estimated）。 */
  matchEstimatesById: Record<string, MatchEstimate>;
  issues: readonly RatingIssue[];
  /**  keyed by String(playerId)；无 Final 者为 null。 */
  peakFinal: Record<string, RatingViewPeak | null>;
}
