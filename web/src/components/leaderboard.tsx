"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { PlayerAvatar } from "@/components/player-avatar";
import { recomputeElos, INITIAL_RATING } from "@/lib/elo";
import { recomputeTrueSkills, TS_MU, TS_SIGMA } from "@/lib/trueskill";
import { getMyPlayerId } from "@/lib/identity";
import { cn } from "@/lib/utils";

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

interface LeaderboardSummary {
  elo: number;
  rank: number;
  weekDelta: number;
}

interface LeaderboardProps {
  players: Player[];
  matches: MatchWithNames[];
  /** key 为球员 id；近一周涨跌取自 weekDelta */
  summaries: Record<number, LeaderboardSummary>;
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

const headerCell =
  "px-3 pb-3.5 text-left text-[10px] font-medium text-muted-foreground";
const bodyCell = "border-t border-border px-3 py-[13px] text-xs";
// 中屏（约 761–1190px）隐藏胜率辅助列
const optionalCell = "min-[761px]:hidden min-[1191px]:table-cell";

export function Leaderboard({ players, matches, summaries }: LeaderboardProps) {
  const router = useRouter();
  const [tab, setTab] = React.useState<Tab>("elo");
  const [myId, setMyId] = React.useState<number | null>(null);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const { eloRatings, tsPlayers, stats } = React.useMemo(() => {
    const eloResult = recomputeElos(matches.map(toEloMatch));
    const tsResult = recomputeTrueSkills(matches.map(toEloMatch));

    const stats = new Map<number, { total: number; wins: number }>();
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
        const mu = ts ? ts.mu : TS_MU;
        const sigma = ts ? ts.sigma : TS_SIGMA;
        return {
          ...player,
          elo,
          mu,
          sigma,
          total: stat.total,
          wins: stat.wins,
          winRate:
            stat.total > 0 ? Math.round((stat.wins / stat.total) * 100) : 0,
        };
      })
      .sort((a, b) => {
        if (tab === "elo") return b.elo - a.elo;
        return b.mu - a.mu;
      });
  }, [players, eloRatings, tsPlayers, stats, tab]);

  return (
    <div className="flex flex-col gap-5">
      <div className="inline-flex self-start rounded-xl bg-muted p-1">
        <button
          onClick={() => setTab("elo")}
          className={cn(
            "rounded-lg px-4 py-1.5 text-xs font-medium transition-all",
            tab === "elo"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          ELO
        </button>
        <button
          onClick={() => setTab("trueskill")}
          className={cn(
            "rounded-lg px-4 py-1.5 text-xs font-medium transition-all",
            tab === "trueskill"
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          TrueSkill
        </button>
      </div>

      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className={cn(headerCell, "w-[34px] pl-0")}>#</th>
            <th className={headerCell}>球员</th>
            <th className={cn(headerCell, optionalCell)}>胜率</th>
            <th className={headerCell}>{tab === "elo" ? "ELO" : "μ"}</th>
            {tab === "elo" && (
              <th className={cn(headerCell, "pr-0 text-right")}>近一周</th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const isMe = row.id === myId;
            const delta = summaries[row.id]?.weekDelta ?? 0;
            return (
              <tr
                key={row.id}
                onClick={() => router.push(`/players/${row.id}`)}
                className={cn(
                  "cursor-pointer",
                  isMe
                    ? "[&>td]:bg-win-bg [&>td:first-child]:rounded-l-[7px] [&>td:first-child]:pl-2 [&>td:last-child]:rounded-r-[7px] [&>td:last-child]:pr-2"
                    : "hover:bg-accent"
                )}
              >
                <td
                  className={cn(
                    bodyCell,
                    "pl-0 font-num text-muted-foreground"
                  )}
                >
                  {String(index + 1).padStart(2, "0")}
                </td>
                <td className={bodyCell}>
                  <Link
                    href={`/players/${row.id}`}
                    onClick={(e) => e.stopPropagation()}
                    className="flex items-center gap-2.5 font-medium text-card-foreground transition-colors hover:text-win"
                  >
                    <PlayerAvatar
                      name={row.name}
                      size="xs"
                      className="size-[29px] text-[11px]"
                    />
                    <span className="truncate">{row.name}</span>
                  </Link>
                </td>
                <td className={cn(bodyCell, optionalCell, "text-muted-foreground")}>
                  {row.winRate}%
                </td>
                <td className={bodyCell}>
                  <span className="font-num text-lg text-card-foreground">
                    {Math.round(tab === "elo" ? row.elo : row.mu)}
                  </span>
                </td>
                {tab === "elo" && (
                  <td className={cn(bodyCell, "pr-0 text-right")}>
                    <span
                      className={cn(
                        "font-num",
                        delta > 0
                          ? "text-win"
                          : delta < 0
                            ? "text-loss"
                            : "text-muted-foreground"
                      )}
                    >
                      {delta > 0 ? `+${delta}` : delta < 0 ? delta : "±0"}
                    </span>
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
