"use client";

import * as React from "react";
import { PlayerAvatar } from "@/components/player-avatar";
import { TodayMatches } from "@/components/today-matches";
import { recomputeElos, INITIAL_RATING } from "@/lib/elo";
import {
  recomputeTrueSkills,
  TS_MU,
  TS_SIGMA,
} from "@/lib/trueskill";

interface Player {
  id: number;
  name: string;
  createdAt: string;
}

interface MatchWithNames {
  id: number;
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
  enteredBy: number | null;
  createdAt: string;
  pa1Name: string;
  pa2Name: string;
  pb1Name: string;
  pb2Name: string;
}

interface LeaderboardProps {
  players: Player[];
  matches: MatchWithNames[];
  todayMatches: MatchWithNames[];
}

type Tab = "elo" | "trueskill";

function toEloMatch(m: MatchWithNames) {
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

export function Leaderboard({
  players,
  matches,
  todayMatches,
}: LeaderboardProps) {
  const [tab, setTab] = React.useState<Tab>("elo");

  const { eloRatings, tsPlayers, stats } = React.useMemo(() => {
    const eloResult = recomputeElos(matches.map(toEloMatch));
    const tsResult = recomputeTrueSkills(matches.map(toEloMatch));

    const stats = new Map<
      number,
      { total: number; wins: number }
    >();
    for (const m of matches) {
      for (const id of [m.pa1, m.pa2]) {
        const s = stats.get(id) ?? { total: 0, wins: 0 };
        s.total++;
        if (m.scoreA > m.scoreB) s.wins++;
        stats.set(id, s);
      }
      for (const id of [m.pb1, m.pb2]) {
        const s = stats.get(id) ?? { total: 0, wins: 0 };
        s.total++;
        if (m.scoreB > m.scoreA) s.wins++;
        stats.set(id, s);
      }
    }

    return {
      eloRatings: eloResult.ratings,
      tsPlayers: tsResult.players,
      stats,
    };
  }, [matches]);

  const rows = React.useMemo(() => {
    return players
      .map((player) => {
        const ts = tsPlayers[String(player.id)];
        const stat = stats.get(player.id) ?? { total: 0, wins: 0 };
        const elo = eloRatings[String(player.id)] ?? INITIAL_RATING;
        const tsScore = ts
          ? ts.mu - 3 * ts.sigma
          : TS_MU - 3 * TS_SIGMA;
        return {
          ...player,
          elo,
          tsScore,
          total: stat.total,
          wins: stat.wins,
          winRate: stat.total > 0 ? Math.round((stat.wins / stat.total) * 100) : 0,
        };
      })
      .sort((a, b) => {
        if (tab === "elo") return b.elo - a.elo;
        return b.tsScore - a.tsScore;
      });
  }, [players, eloRatings, tsPlayers, stats, tab]);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-foreground">排行榜</h1>
      </header>

      <div className="inline-flex rounded-xl bg-muted p-1">
        <button
          onClick={() => setTab("elo")}
          className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
            tab === "elo"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          ELO
        </button>
        <button
          onClick={() => setTab("trueskill")}
          className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
            tab === "trueskill"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          TrueSkill
        </button>
      </div>

      <div className="space-y-2">
        {rows.map((row, index) => (
          <div
            key={row.id}
            className="flex items-center gap-3 rounded-2xl border border-border bg-card p-3 shadow-sm"
          >
            <div className="flex w-8 justify-center text-lg font-bold text-muted-foreground">
              {index + 1}
            </div>
            <PlayerAvatar name={row.name} size="sm" />
            <div className="flex flex-1 flex-col">
              <span className="font-medium text-card-foreground">
                {row.name}
              </span>
              <span className="text-xs text-muted-foreground">
                {row.total} 场 · {row.wins} 胜 · {row.winRate}%
              </span>
            </div>
            <div className="text-right">
              <div className="text-xl font-bold tabular-nums text-card-foreground">
                {Math.round(tab === "elo" ? row.elo : row.tsScore)}
              </div>
              <div className="text-[10px] text-muted-foreground">
                {tab === "trueskill" && "μ-3σ"}
              </div>
            </div>
          </div>
        ))}
      </div>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">当天比赛</h2>
        <TodayMatches matches={todayMatches} />
      </section>
    </div>
  );
}
