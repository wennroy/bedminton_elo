import {
  listPlayers,
  listMatchesByDate,
  recomputeAllRatings,
  type MatchWithNames,
  type Player,
} from "@/lib/repo";
import { recomputeElos, INITIAL_RATING, type Match as EloMatch } from "@/lib/elo";
import {
  recomputeTrueSkills,
  TS_MU,
  TS_SIGMA,
  type TrueSkillPlayer,
} from "@/lib/trueskill";

export interface HeadToHeadRecord {
  opponentId: number;
  opponentName: string;
  wins: number;
  losses: number;
  total: number;
}

export interface EloHistoryPoint {
  date: string;
  playerId: string;
  playerName: string;
  elo: number;
}

export interface PlayerMatchRecord {
  id: number;
  date: string;
  teammates: string[];
  opponents: string[];
  scoreFor: number;
  scoreAgainst: number;
  won: boolean;
}

export interface PlayerSummary {
  id: number;
  name: string;
  elo: number;
  tsScore: number;
  totalMatches: number;
  wins: number;
  losses: number;
  winRate: number;
}

export interface StatsData {
  players: Player[];
  matches: MatchWithNames[];
  ratings: Map<number, { elo: number; mu: number; sigma: number }>;
  eloHistory: EloHistoryPoint[];
  tsPlayers: Record<string, TrueSkillPlayer>;
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

export function buildStatsData(): StatsData {
  const players = listPlayers();
  const matches = listMatchesByDate();
  const ratings = recomputeAllRatings();

  const eloResult = recomputeElos(matches.map(toEloMatch));
  const tsResult = recomputeTrueSkills(matches.map(toEloMatch));

  const nameMap = new Map(players.map((p) => [p.id, p.name]));
  const eloHistory: EloHistoryPoint[] = eloResult.snapshots.map((s) => ({
    date: s.date,
    playerId: s.playerId,
    playerName: nameMap.get(Number(s.playerId)) ?? "?",
    elo: Math.round(s.elo),
  }));

  return { players, matches, ratings, eloHistory, tsPlayers: tsResult.players };
}

export function headToHead(
  playerId: number,
  data: StatsData
): HeadToHeadRecord[] {
  const records = new Map<number, HeadToHeadRecord>();
  const get = (id: number, name: string) => {
    if (!records.has(id)) {
      records.set(id, {
        opponentId: id,
        opponentName: name,
        wins: 0,
        losses: 0,
        total: 0,
      });
    }
    return records.get(id)!;
  };

  for (const m of data.matches) {
    const teamA = [m.pa1, m.pa2];
    const teamB = [m.pb1, m.pb2];
    const aWon = m.scoreA > m.scoreB;

    if (teamA.includes(playerId)) {
      for (const opp of [m.pb1, m.pb2]) {
        const r = get(opp, opp === m.pb1 ? m.pb1Name : m.pb2Name);
        r.total++;
        if (aWon) r.wins++;
        else r.losses++;
      }
    } else if (teamB.includes(playerId)) {
      for (const opp of [m.pa1, m.pa2]) {
        const r = get(opp, opp === m.pa1 ? m.pa1Name : m.pa2Name);
        r.total++;
        if (!aWon) r.wins++;
        else r.losses++;
      }
    }
  }

  return Array.from(records.values()).sort((a, b) => b.total - a.total);
}

export function playerMatches(
  playerId: number,
  data: StatsData
): PlayerMatchRecord[] {
  const records: PlayerMatchRecord[] = [];
  for (const m of data.matches) {
    const teamA = [m.pa1, m.pa2];
    const teamB = [m.pb1, m.pb2];
    const aWon = m.scoreA > m.scoreB;

    if (teamA.includes(playerId)) {
      records.push({
        id: m.id,
        date: m.playedAt,
        teammates: teamA.filter((id) => id !== playerId).map((id) =>
          id === m.pa1 ? m.pa1Name : m.pa2Name
        ),
        opponents: [m.pb1Name, m.pb2Name],
        scoreFor: m.scoreA,
        scoreAgainst: m.scoreB,
        won: aWon,
      });
    } else if (teamB.includes(playerId)) {
      records.push({
        id: m.id,
        date: m.playedAt,
        teammates: teamB.filter((id) => id !== playerId).map((id) =>
          id === m.pb1 ? m.pb1Name : m.pb2Name
        ),
        opponents: [m.pa1Name, m.pa2Name],
        scoreFor: m.scoreB,
        scoreAgainst: m.scoreA,
        won: !aWon,
      });
    }
  }
  return records.reverse();
}

export function playerSummary(
  playerId: number,
  data: StatsData
): PlayerSummary | undefined {
  const player = data.players.find((p) => p.id === playerId);
  if (!player) return undefined;

  const rating = data.ratings.get(playerId);
  const ts = data.tsPlayers[String(playerId)];

  let total = 0;
  let wins = 0;
  for (const m of data.matches) {
    if ([m.pa1, m.pa2, m.pb1, m.pb2].includes(playerId)) {
      total++;
      const aWon = m.scoreA > m.scoreB;
      if (
        (teamIncludes(m, playerId, "A") && aWon) ||
        (teamIncludes(m, playerId, "B") && !aWon)
      ) {
        wins++;
      }
    }
  }

  return {
    id: playerId,
    name: player.name,
    elo: rating?.elo ?? INITIAL_RATING,
    tsScore: ts ? ts.mu - 3 * ts.sigma : TS_MU - 3 * TS_SIGMA,
    totalMatches: total,
    wins,
    losses: total - wins,
    winRate: total > 0 ? Math.round((wins / total) * 100) : 0,
  };
}

function teamIncludes(
  m: MatchWithNames,
  playerId: number,
  team: "A" | "B"
): boolean {
  if (team === "A") return m.pa1 === playerId || m.pa2 === playerId;
  return m.pb1 === playerId || m.pb2 === playerId;
}
