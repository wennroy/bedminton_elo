import { createRatingConfig } from "../../src/lib/ratings/config";
import type { RatingConfig, RatingMatch } from "../../src/lib/ratings/types";
import type { EvaluationInput } from "../../src/lib/ratings/evaluation";

export const replayConfig: RatingConfig = createRatingConfig({
  firstSeasonStart: "2026-10-01",
});

export function replayMatch(
  id: number,
  playedAt: string,
  scoreA = 21,
  scoreB = 17,
  teamA: readonly [number, number] = [1, 2],
  teamB: readonly [number, number] = [3, 4],
  createdAt = `${playedAt}T12:00:00+08:00`
): RatingMatch {
  return { id, playedAt, createdAt, teamA, teamB, scoreA, scoreB };
}

/** Numeric-only fixture for the standalone rating backtest CLI. */
export const syntheticRatingScenario: EvaluationInput = {
  source: { kind: "synthetic", inputVersion: "synthetic-v1", inputHash: "synthetic-input-v1" },
  playerIds: [101, 102, 103, 104, 105, 106],
  matches: [
    replayMatch(1, "2026-09-22", 21, 17, [101, 102], [103, 104]),
    replayMatch(2, "2026-09-22", 18, 21, [101, 103], [105, 106], "2026-09-22T13:00:00+08:00"),
    replayMatch(3, "2026-09-23", 21, 19, [101, 104], [105, 106]),
    replayMatch(4, "2026-09-24", 16, 21, [102, 103], [104, 105]),
    replayMatch(5, "2026-09-25", 21, 18, [102, 106], [103, 104]),
    replayMatch(6, "2026-09-26", 17, 21, [101, 105], [102, 106]),
    replayMatch(7, "2026-09-27", 21, 15, [103, 106], [104, 105]),
    replayMatch(8, "2026-09-29", 21, 18, [101, 106], [102, 104]),
    replayMatch(9, "2026-09-30", 18, 21, [102, 105], [103, 106]),
    replayMatch(10, "2026-10-01", 21, 19, [101, 103], [104, 106]),
    replayMatch(11, "2026-10-02", 17, 21, [102, 104], [103, 105]),
    replayMatch(12, "2026-10-03", 21, 16, [101, 105], [102, 106]),
    replayMatch(13, "2026-10-04", 18, 21, [101, 104], [103, 106]),
    replayMatch(14, "2026-10-06", 21, 18, [102, 103], [104, 105]),
    replayMatch(15, "2026-02-30", 21, 18, [101, 102], [103, 104]),
    replayMatch(16, "2026-10-03", 21, 21, [101, 102], [103, 104]),
    replayMatch(17, "2026-10-03", 21, 18, [101, 101], [103, 104]),
    replayMatch(18, "2026-10-03", 21, 18, [101, 102], [103, 999]),
  ],
};
