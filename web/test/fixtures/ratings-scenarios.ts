import { createRatingConfig } from "../../src/lib/ratings/config";
import type { RatingConfig, RatingMatch } from "../../src/lib/ratings/types";

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
