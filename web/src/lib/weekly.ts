import {
  listPlayers,
  listMatchesByDate,
  type MatchWithNames,
} from "@/lib/repo";
import { recomputeElos, INITIAL_RATING, type Match as EloMatch } from "@/lib/elo";

export interface WeeklyPlayerStat {
  playerId: number;
  name: string;
  matches: number;
  wins: number;
  losses: number;
}

export interface EloChangeStat {
  playerId: number;
  name: string;
  eloStart: number;
  eloEnd: number;
  change: number;
}

export interface BestPair {
  playerA: string;
  playerB: string;
  wins: number;
  total: number;
  winRate: number;
}

export interface WeeklyStats {
  weekStart: string;
  weekEnd: string;
  weekNumber: number;
  attendance: WeeklyPlayerStat[];
  winKing: WeeklyPlayerStat[];
  eloChanges: EloChangeStat[];
  bestPair: BestPair | null;
}

function toEloMatch(m: MatchWithNames): EloMatch {
  return {
    date: m.playedAt,
    a1: String(m.pa1),
    a2: String(m.pa2),
    b1: String(m.pb1),
    b2: String(m.pb2),
    scoreA: m.scoreA,
    scoreB: m.scoreB,
  };
}

export function getWeekRange(dateStr: string): {
  weekStart: string;
  weekEnd: string;
  weekNumber: number;
} {
  const [year, month, day] = dateStr.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  const dayOfWeek = date.getDay();
  const mondayOffset = dayOfWeek === 0 ? -6 : 1 - dayOfWeek;
  const monday = new Date(date);
  monday.setDate(date.getDate() + mondayOffset);
  const sunday = new Date(monday);
  sunday.setDate(monday.getDate() + 6);

  const fmt = (d: Date) =>
    `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
      d.getDate()
    ).padStart(2, "0")}`;

  const startOfYear = new Date(monday.getFullYear(), 0, 1);
  const diffMs = monday.getTime() - startOfYear.getTime();
  const weekNumber = Math.floor(diffMs / (7 * 24 * 60 * 60 * 1000)) + 1;

  return { weekStart: fmt(monday), weekEnd: fmt(sunday), weekNumber };
}

export function listWeekStarts(): string[] {
  const matches = listMatchesByDate();
  if (matches.length === 0) return [];
  const first = matches[0].playedAt;
  const last = matches[matches.length - 1].playedAt;
  const { weekStart: firstWeek } = getWeekRange(first);
  const { weekStart: lastWeek } = getWeekRange(last);

  const starts: string[] = [];
  let current = firstWeek;
  while (current <= lastWeek) {
    starts.push(current);
    const [y, m, d] = current.split("-").map(Number);
    const next = new Date(y, m - 1, d + 7);
    current = `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(
      2,
      "0"
    )}-${String(next.getDate()).padStart(2, "0")}`;
  }
  return starts;
}

export function buildWeeklyStats(weekStart: string): WeeklyStats {
  return computeWeeklyStats(weekStart, listPlayers(), listMatchesByDate());
}

export function computeWeeklyStats(
  weekStart: string,
  players: { id: number; name: string }[],
  allMatches: MatchWithNames[]
): WeeklyStats {
  const { weekEnd, weekNumber } = getWeekRange(weekStart);
  const nameMap = new Map(players.map((p) => [p.id, p.name]));

  const weekMatches = allMatches.filter(
    (m) => m.playedAt >= weekStart && m.playedAt <= weekEnd
  );

  const stats = new Map<
    number,
    { playerId: number; name: string; matches: number; wins: number; losses: number }
  >();
  const ensure = (id: number) => {
    if (!stats.has(id)) {
      stats.set(id, {
        playerId: id,
        name: nameMap.get(id) ?? "?",
        matches: 0,
        wins: 0,
        losses: 0,
      });
    }
    return stats.get(id)!;
  };

  const pairStats = new Map<
    string,
    { key: string; names: string[]; wins: number; total: number }
  >();

  for (const m of weekMatches) {
    const aWon = m.scoreA > m.scoreB;

    for (const id of [m.pa1, m.pa2]) {
      const s = ensure(id);
      s.matches++;
      if (aWon) s.wins++;
      else s.losses++;
    }
    for (const id of [m.pb1, m.pb2]) {
      const s = ensure(id);
      s.matches++;
      if (!aWon) s.wins++;
      else s.losses++;
    }

    const teamA = [m.pa1, m.pa2].sort((a, b) => a - b);
    const teamB = [m.pb1, m.pb2].sort((a, b) => a - b);
    const keyA = teamA.join(",");
    const keyB = teamB.join(",");

    for (const [key, ids, won] of [
      [keyA, teamA, aWon],
      [keyB, teamB, !aWon],
    ] as const) {
      if (!pairStats.has(key)) {
        pairStats.set(key, {
          key,
          names: ids.map((id) => nameMap.get(id) ?? "?"),
          wins: 0,
          total: 0,
        });
      }
      const p = pairStats.get(key)!;
      p.total++;
      if (won) p.wins++;
    }
  }

  const allValues = Array.from(stats.values());
  const attendance = [...allValues]
    .filter((s) => s.matches > 0)
    .sort((a, b) => b.matches - a.matches || b.wins - a.wins);
  const winKing = [...allValues]
    .filter((s) => s.wins > 0)
    .sort((a, b) => b.wins - a.wins || b.matches - a.matches);

  const eloChanges = computeEloChanges(weekStart, weekEnd, nameMap, allMatches);

  let bestPair: BestPair | null = null;
  for (const p of pairStats.values()) {
    if (p.total < 3) continue;
    const winRate = p.wins / p.total;
    if (!bestPair || winRate > bestPair.winRate) {
      bestPair = {
        playerA: p.names[0],
        playerB: p.names[1],
        wins: p.wins,
        total: p.total,
        winRate,
      };
    }
  }

  return {
    weekStart,
    weekEnd,
    weekNumber,
    attendance,
    winKing,
    eloChanges,
    bestPair,
  };
}

function computeEloChanges(
  weekStart: string,
  weekEnd: string,
  nameMap: Map<number, string>,
  allMatches: MatchWithNames[]
): EloChangeStat[] {
  const { snapshots } = recomputeElos(allMatches.map(toEloMatch));

  const lastSnapshotBefore = new Map<string, number>();
  const snapshotOnOrBeforeEnd = new Map<string, number>();

  for (const s of snapshots) {
    if (s.date < weekStart) {
      lastSnapshotBefore.set(s.playerId, s.elo);
    }
    if (s.date <= weekEnd) {
      snapshotOnOrBeforeEnd.set(s.playerId, s.elo);
    }
  }

  const playerIds = new Set<string>();
  for (const m of allMatches) {
    if (m.playedAt >= weekStart && m.playedAt <= weekEnd) {
      playerIds.add(String(m.pa1));
      playerIds.add(String(m.pa2));
      playerIds.add(String(m.pb1));
      playerIds.add(String(m.pb2));
    }
  }

  const changes: EloChangeStat[] = [];
  for (const id of playerIds) {
    const numId = Number(id);
    const eloStart = lastSnapshotBefore.get(id) ?? INITIAL_RATING;
    const eloEnd = snapshotOnOrBeforeEnd.get(id) ?? eloStart;
    changes.push({
      playerId: numId,
      name: nameMap.get(numId) ?? "?",
      eloStart: Math.round(eloStart),
      eloEnd: Math.round(eloEnd),
      change: Math.round(eloEnd - eloStart),
    });
  }

  return changes.sort((a, b) => b.change - a.change || a.playerId - b.playerId);
}
