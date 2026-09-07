import {
  listPlayers,
  listMatchesByDate,
  recomputeAllRatings,
  type MatchWithNames,
  type Player,
} from "@/lib/repo";
import { recomputeElos, computeMatchEloDeltas, INITIAL_RATING, type Match as EloMatch } from "@/lib/elo";
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
  /** 该场 ELO 变化（服务端重放输出，四舍五入到整数） */
  delta: number;
}

export interface PlayerSummary {
  id: number;
  name: string;
  elo: number;
  mu: number;
  sigma: number;
  totalMatches: number;
  wins: number;
  losses: number;
  winRate: number;
}

export interface PlayerFunStats {
  currentStreak: number;
  currentStreakType: "win" | "loss" | "none";
  longestWinStreak: number;
  /** 搭档 ≥3 场中胜率最高者;不足为 null */
  bestPartner: {
    id: number;
    name: string;
    wins: number;
    total: number;
    winRate: number;
  } | null;
  /** 交手 ≥3 场中我方胜率最低者;不足为 null */
  nemesis: {
    id: number;
    name: string;
    wins: number;
    losses: number;
    total: number;
    winRate: number;
  } | null;
  peakElo: number;
  peakEloDate: string | null;
  avgPointDiff: number;
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
  // 每场 ELO delta 由服务端重放输出；data.matches 升序，与 deltas 按下标对齐
  const deltas = computeMatchEloDeltas(data.matches.map(toEloMatch));
  const pid = String(playerId);
  const records: PlayerMatchRecord[] = [];
  data.matches.forEach((m, i) => {
    const teamA = [m.pa1, m.pa2];
    const teamB = [m.pb1, m.pb2];
    const aWon = m.scoreA > m.scoreB;
    const delta = Math.round(deltas[i]?.[pid] ?? 0);

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
        delta,
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
        delta,
      });
    }
  });
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
    mu: ts?.mu ?? TS_MU,
    sigma: ts?.sigma ?? TS_SIGMA,
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

const MIN_PAIR_GAMES = 3;

function nameOf(m: MatchWithNames, id: number): string {
  if (id === m.pa1) return m.pa1Name;
  if (id === m.pa2) return m.pa2Name;
  if (id === m.pb1) return m.pb1Name;
  return m.pb2Name;
}

/** data.matches 按 played_at/created_at/id 升序(listMatchesByDate 保证),可直接顺序扫描。 */
export function playerFunStats(
  playerId: number,
  data: StatsData
): PlayerFunStats {
  let played = 0;
  let diffSum = 0;
  let longestWinStreak = 0;
  let winRun = 0;
  let currentStreak = 0;
  let lastType: "win" | "loss" | "none" = "none";
  const partners = new Map<number, { id: number; name: string; wins: number; total: number }>();
  const opponents = new Map<
    number,
    { id: number; name: string; wins: number; losses: number; total: number }
  >();

  for (const m of data.matches) {
    const inA = teamIncludes(m, playerId, "A");
    const inB = teamIncludes(m, playerId, "B");
    if (!inA && !inB) continue;
    played++;

    const aWon = m.scoreA > m.scoreB;
    const won = (inA && aWon) || (inB && !aWon);
    diffSum += inA ? m.scoreA - m.scoreB : m.scoreB - m.scoreA;

    if (won) {
      winRun++;
      longestWinStreak = Math.max(longestWinStreak, winRun);
    } else {
      winRun = 0;
    }
    const type = won ? ("win" as const) : ("loss" as const);
    currentStreak = type === lastType ? currentStreak + 1 : 1;
    lastType = type;

    const partnerId = inA
      ? m.pa1 === playerId
        ? m.pa2
        : m.pa1
      : m.pb1 === playerId
        ? m.pb2
        : m.pb1;
    const partner = partners.get(partnerId) ?? {
      id: partnerId,
      name: nameOf(m, partnerId),
      wins: 0,
      total: 0,
    };
    partner.total++;
    if (won) partner.wins++;
    partners.set(partnerId, partner);

    for (const oppId of inA ? [m.pb1, m.pb2] : [m.pa1, m.pa2]) {
      const opp = opponents.get(oppId) ?? {
        id: oppId,
        name: nameOf(m, oppId),
        wins: 0,
        losses: 0,
        total: 0,
      };
      opp.total++;
      if (won) opp.wins++;
      else opp.losses++;
      opponents.set(oppId, opp);
    }
  }

  const withRate = <T extends { wins: number; total: number }>(x: T) => ({
    ...x,
    winRate: Math.round((x.wins / x.total) * 100),
  });

  const bestPartner =
    [...partners.values()]
      .filter((p) => p.total >= MIN_PAIR_GAMES)
      .sort((a, b) => b.wins / b.total - a.wins / a.total || b.total - a.total)
      .map(withRate)[0] ?? null;

  const nemesis =
    [...opponents.values()]
      .filter((o) => o.total >= MIN_PAIR_GAMES)
      .sort((a, b) => a.wins / a.total - b.wins / b.total || b.total - a.total)
      .map(withRate)[0] ?? null;

  const history = data.eloHistory.filter(
    (h) => h.playerId === String(playerId)
  );
  // eloHistory 按时间升序,第一个 ">" 即首次达成该峰值
  let peakElo = Math.round(data.ratings.get(playerId)?.elo ?? INITIAL_RATING);
  let peakEloDate: string | null = null;
  if (history.length > 0) {
    peakElo = -Infinity;
    for (const h of history) {
      if (h.elo > peakElo) {
        peakElo = h.elo;
        peakEloDate = h.date;
      }
    }
  }

  return {
    currentStreak,
    currentStreakType: lastType,
    longestWinStreak,
    bestPartner,
    nemesis,
    peakElo,
    peakEloDate,
    avgPointDiff: played > 0 ? Math.round((diffSum / played) * 10) / 10 : 0,
  };
}

export interface RelationRecord {
  id: number;
  name: string;
  wins: number;
  losses: number;
  total: number;
  /** 我方胜率（整数百分比） */
  winRate: number;
}

/**
 * 搭档 / 对手全量列表（含 <3 场样本）。
 * 搭档按胜率降序、对手按我方胜率升序（与 bestPartner / nemesis 口径一致，
 * 因此过滤 ≥3 场后的第一项即黄金搭档 / 值得研究的对手）。
 */
export function playerRelations(
  playerId: number,
  data: StatsData
): { partners: RelationRecord[]; opponents: RelationRecord[] } {
  const partners = new Map<number, RelationRecord>();
  const opponents = new Map<number, RelationRecord>();
  const get = (map: Map<number, RelationRecord>, id: number, name: string) => {
    if (!map.has(id)) {
      map.set(id, { id, name, wins: 0, losses: 0, total: 0, winRate: 0 });
    }
    return map.get(id)!;
  };

  for (const m of data.matches) {
    const inA = teamIncludes(m, playerId, "A");
    const inB = teamIncludes(m, playerId, "B");
    if (!inA && !inB) continue;
    const won = (inA && m.scoreA > m.scoreB) || (inB && m.scoreB > m.scoreA);

    const partnerId = inA
      ? m.pa1 === playerId
        ? m.pa2
        : m.pa1
      : m.pb1 === playerId
        ? m.pb2
        : m.pb1;
    const partner = get(partners, partnerId, nameOf(m, partnerId));
    partner.total++;
    if (won) partner.wins++;
    else partner.losses++;

    for (const oppId of inA ? [m.pb1, m.pb2] : [m.pa1, m.pa2]) {
      const opp = get(opponents, oppId, nameOf(m, oppId));
      opp.total++;
      if (won) opp.wins++;
      else opp.losses++;
    }
  }

  const finish = (r: RelationRecord): RelationRecord => ({
    ...r,
    winRate: Math.round((r.wins / r.total) * 100),
  });
  const byRawRate = (a: RelationRecord, b: RelationRecord) =>
    a.wins / a.total - b.wins / b.total;

  return {
    partners: [...partners.values()]
      .sort((a, b) => byRawRate(b, a) || b.total - a.total || a.id - b.id)
      .map(finish),
    opponents: [...opponents.values()]
      .sort((a, b) => byRawRate(a, b) || b.total - a.total || a.id - b.id)
      .map(finish),
  };
}
