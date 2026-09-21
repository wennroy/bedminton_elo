export type PlayerId = number;
export type LocalDate = string;
export type RatingModel = "glicko2" | "legacy";
export type RatingStatus = "estimated" | "final" | "unrated";

export interface RatingState {
  r: number;
  rd: number;
  volatility: number;
}

export interface RatingMatch {
  id: number;
  playedAt: LocalDate;
  createdAt: string;
  teamA: readonly [PlayerId, PlayerId];
  teamB: readonly [PlayerId, PlayerId];
  scoreA: number;
  scoreB: number;
}

export interface RatingConfig {
  algorithmVersion: "glicko2-doubles-v1";
  paramsVersion: string;
  firstSeasonStart: LocalDate;
  timeZone: "Asia/Shanghai";
  initialRating: number;
  initialRd: number;
  minRd: number;
  maxRd: number;
  initialVolatility: number;
  tau: number;
  seasonLower: number;
  seasonUpper: number;
  seasonRetention: number;
  seasonRdFloor: number;
}

export interface RatingSegment {
  id: string;
  weekStart: LocalDate;
  seasonId: string | null;
  start: string;
  end: string;
  h: number;
}

export interface PlayerChange {
  playerId: PlayerId;
  before: RatingState;
  after: RatingState;
  delta: number;
}

export interface MatchEstimate {
  kind: "match_estimated";
  eventId: string;
  matchId: number;
  segmentId: string;
  playedAt: LocalDate;
  preWinA: number;
  changes: readonly PlayerChange[];
}

export interface PeriodFinal {
  kind: "weekly_final";
  eventId: string;
  segment: RatingSegment;
  start: Record<string, RatingState>;
  estimatedEnd: Record<string, RatingState>;
  final: Record<string, RatingState>;
  correction: Record<string, number>;
}

export interface SeasonReset {
  kind: "season_reset";
  eventId: string;
  at: string;
  seasonId: string;
  changes: readonly PlayerChange[];
}

export type RatingEvent = MatchEstimate | PeriodFinal | SeasonReset;

export interface RatingIssue {
  matchId: number;
  reason: "invalid_date" | "invalid_score" | "duplicate_player" | "unknown_player";
}

export interface RatingReplay {
  asOf: string;
  configVersion: string;
  currentSegment: RatingSegment;
  current: Record<string, RatingState>;
  lastFinal: Record<string, RatingState>;
  events: readonly RatingEvent[];
  matchEstimates: Record<string, MatchEstimate>;
  issues: readonly RatingIssue[];
  nextBoundary: string;
}
