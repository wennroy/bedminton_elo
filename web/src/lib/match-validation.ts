import { isValidLocalDate } from "./ratings/calendar";
import type { PlayerId } from "./ratings/types";

/**
 * 统一的比赛录入验证：POST /api/matches 与 repo.addMatch 共用同一套规则，
 * 保证「API 拒绝的写入」与「repo 抛错的写入」完全一致。
 * 不额外限制 21 分上限，也不拒绝未来日期（未来场次合法保存、新版评分暂不生效）。
 */
export interface MatchValidationInput {
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
}

export type MatchValidationErrorCode =
  | "invalid_player_id"
  | "duplicate_player"
  | "unknown_player"
  | "invalid_date"
  | "invalid_score";

/** 验证失败抛出的错误；route 层按 code 返回 4xx，repo 层直接向上抛。 */
export class MatchValidationError extends Error {
  readonly code: MatchValidationErrorCode;

  constructor(code: MatchValidationErrorCode, message: string) {
    super(message);
    this.name = "MatchValidationError";
    this.code = code;
  }
}

/**
 * 校验四个互不相同且存在于球员目录的整数 ID、真实日历日期
 * （ratings/calendar 的 isValidLocalDate，2026-02-30 这类日期拒绝）、
 * 非负整数且不相等的比分。存在性校验由调用方注入球员目录集合。
 */
export function assertValidMatchInput(
  input: MatchValidationInput,
  knownPlayerIds: ReadonlySet<PlayerId>
): void {
  const ids = [input.pa1, input.pa2, input.pb1, input.pb2];
  for (const id of ids) {
    if (!Number.isSafeInteger(id)) {
      throw new MatchValidationError(
        "invalid_player_id",
        "Player ids must be integers"
      );
    }
  }
  if (new Set(ids).size !== 4) {
    throw new MatchValidationError(
      "duplicate_player",
      "Four players must be distinct"
    );
  }
  for (const id of ids) {
    if (!knownPlayerIds.has(id)) {
      throw new MatchValidationError(
        "unknown_player",
        `Unknown player id: ${id}`
      );
    }
  }
  if (!isValidLocalDate(input.playedAt)) {
    throw new MatchValidationError(
      "invalid_date",
      "playedAt must be a valid YYYY-MM-DD calendar date"
    );
  }
  if (
    !Number.isSafeInteger(input.scoreA) ||
    input.scoreA < 0 ||
    !Number.isSafeInteger(input.scoreB) ||
    input.scoreB < 0
  ) {
    throw new MatchValidationError(
      "invalid_score",
      "Scores must be non-negative integers"
    );
  }
  if (input.scoreA === input.scoreB) {
    throw new MatchValidationError("invalid_score", "Scores must not be equal");
  }
}
