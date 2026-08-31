export interface Match {
  date: string;
  a1: string;
  a2: string;
  b1: string;
  b2: string;
  scoreA: number;
  scoreB: number;
}

export interface EloSnapshot {
  date: string;
  playerId: string;
  elo: number;
}

export const INITIAL_RATING = 1000;
export const K_DOUBLES = 16;

export function predictElo(
  a1: string,
  a2: string,
  b1: string,
  b2: string,
  ratings: Record<string, number>
): { teamAWin: number; teamBWin: number } {
  const teamAAvg = (ratingOf(a1, ratings) + ratingOf(a2, ratings)) / 2;
  const teamBAvg = (ratingOf(b1, ratings) + ratingOf(b2, ratings)) / 2;
  const teamAWin = 1 / (1 + 10 ** ((teamBAvg - teamAAvg) / 400));
  return { teamAWin, teamBWin: 1 - teamAWin };
}

function ratingOf(playerId: string, ratings: Record<string, number>): number {
  return ratings[playerId] ?? INITIAL_RATING;
}

export function recomputeElos(matches: Match[]): {
  ratings: Record<string, number>;
  snapshots: EloSnapshot[];
} {
  const ratings: Record<string, number> = {};
  const snapshots: EloSnapshot[] = [];
  let lastDate: string | null = null;

  function recordDayBoundary(date: string | null) {
    if (date === null) return;
    for (const [playerId, elo] of Object.entries(ratings)) {
      snapshots.push({ date, playerId, elo });
    }
  }

  for (const match of matches) {
    if (lastDate === null || lastDate !== match.date) {
      recordDayBoundary(lastDate);
      lastDate = match.date;
    }

    for (const playerId of [match.a1, match.a2, match.b1, match.b2]) {
      if (ratings[playerId] === undefined) {
        ratings[playerId] = INITIAL_RATING;
      }
    }

    const teamAAvg = (ratings[match.a1] + ratings[match.a2]) / 2;
    const teamBAvg = (ratings[match.b1] + ratings[match.b2]) / 2;

    const aWins = match.scoreA > match.scoreB;
    const sA = aWins ? 1 : 0;
    const sB = aWins ? 0 : 1;

    for (const playerId of [match.a1, match.a2]) {
      const expected =
        1 / (1 + 10 ** ((teamBAvg - ratings[playerId]) / 400));
      ratings[playerId] += K_DOUBLES * (sA - expected);
    }

    for (const playerId of [match.b1, match.b2]) {
      const expected =
        1 / (1 + 10 ** ((teamAAvg - ratings[playerId]) / 400));
      ratings[playerId] += K_DOUBLES * (sB - expected);
    }
  }

  recordDayBoundary(lastDate);

  return { ratings, snapshots };
}
