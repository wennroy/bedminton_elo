"use client";

import * as React from "react";
import { PlayerAvatar } from "@/components/player-avatar";
import { Button } from "@/components/ui/button";
import { predictElo } from "@/lib/elo";
import { predictTeamOutcomeWin, createPlayer } from "@/lib/trueskill";
import { RotateCcw } from "lucide-react";

interface PredictFormProps {
  players: { id: number; name: string }[];
  ratings: Map<number, { elo: number; mu: number; sigma: number }>;
}

type Slot = number | null;

export function PredictForm({ players, ratings }: PredictFormProps) {
  const [teamA, setTeamA] = React.useState<[Slot, Slot]>([null, null]);
  const [teamB, setTeamB] = React.useState<[Slot, Slot]>([null, null]);

  const playerMap = React.useMemo(
    () => new Map(players.map((p) => [p.id, p])),
    [players]
  );

  const selected = new Set([
    teamA[0], teamA[1], teamB[0], teamB[1],
  ].filter((id): id is number => id !== null));

  const teamAIds = [teamA[0], teamA[1]].filter((id): id is number => id !== null);
  const teamBIds = [teamB[0], teamB[1]].filter((id): id is number => id !== null);
  const ready = teamAIds.length === 2 && teamBIds.length === 2;

  const eloPrediction = React.useMemo(() => {
    if (!ready) return null;
    const eloRatings: Record<string, number> = {};
    for (const p of players) {
      eloRatings[String(p.id)] = ratings.get(p.id)?.elo ?? 1000;
    }
    return predictElo(
      String(teamAIds[0]),
      String(teamAIds[1]),
      String(teamBIds[0]),
      String(teamBIds[1]),
      eloRatings
    );
  }, [ready, teamAIds, teamBIds, players, ratings]);

  const tsPrediction = React.useMemo(() => {
    if (!ready) return null;
    const teamAPlayers = teamAIds.map((id) => {
      const r = ratings.get(id);
      return createPlayer(r?.mu ?? 25, r?.sigma ?? 8.333);
    });
    const teamBPlayers = teamBIds.map((id) => {
      const r = ratings.get(id);
      return createPlayer(r?.mu ?? 25, r?.sigma ?? 8.333);
    });
    return predictTeamOutcomeWin(teamAPlayers, teamBPlayers);
  }, [ready, teamAIds, teamBIds, ratings]);

  function toggleTeamA(id: number) {
    setTeamA((current) => {
      if (current.includes(id)) return [null, null] as [Slot, Slot];
      const empty = current.indexOf(null);
      if (empty === -1) return current;
      const next: [Slot, Slot] = [...current];
      next[empty] = id;
      return next;
    });
  }

  function toggleTeamB(id: number) {
    setTeamB((current) => {
      if (current.includes(id)) return [null, null] as [Slot, Slot];
      const empty = current.indexOf(null);
      if (empty === -1) return current;
      const next: [Slot, Slot] = [...current];
      next[empty] = id;
      return next;
    });
  }

  function clear() {
    setTeamA([null, null]);
    setTeamB([null, null]);
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-2 gap-3">
        <TeamPanel
          label="A 队"
          accent="bg-blue-50 text-blue-700 ring-blue-200"
          slots={teamA}
          playerMap={playerMap}
        />
        <TeamPanel
          label="B 队"
          accent="bg-orange-50 text-orange-700 ring-orange-200"
          slots={teamB}
          playerMap={playerMap}
        />
      </div>

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold text-card-foreground">选择球员</h2>
          {selected.size > 0 && (
            <button
              onClick={clear}
              className="flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <RotateCcw className="size-3" />
              清空
            </button>
          )}
        </div>

        <div className="space-y-4">
          <div>
            <h3 className="mb-2 text-xs font-medium text-muted-foreground">A 队</h3>
            <div className="grid grid-cols-4 gap-2 sm:grid-cols-5">
              {players.map((player) => {
                const active = teamA.includes(player.id);
                const disabled = !active && selected.has(player.id);
                return (
                  <button
                    key={`a-${player.id}`}
                    onClick={() => toggleTeamA(player.id)}
                    disabled={disabled}
                    className={`flex flex-col items-center gap-1 rounded-xl border p-2 transition-all ${
                      active
                        ? "border-blue-500 bg-blue-50"
                        : disabled
                        ? "border-border bg-muted opacity-40"
                        : "border-border bg-background hover:bg-muted"
                    }`}
                  >
                    <PlayerAvatar name={player.name} size="sm" />
                    <span className="max-w-full truncate text-xs font-medium text-foreground">
                      {player.name}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <h3 className="mb-2 text-xs font-medium text-muted-foreground">B 队</h3>
            <div className="grid grid-cols-4 gap-2 sm:grid-cols-5">
              {players.map((player) => {
                const active = teamB.includes(player.id);
                const disabled = !active && selected.has(player.id);
                return (
                  <button
                    key={`b-${player.id}`}
                    onClick={() => toggleTeamB(player.id)}
                    disabled={disabled}
                    className={`flex flex-col items-center gap-1 rounded-xl border p-2 transition-all ${
                      active
                        ? "border-orange-500 bg-orange-50"
                        : disabled
                        ? "border-border bg-muted opacity-40"
                        : "border-border bg-background hover:bg-muted"
                    }`}
                  >
                    <PlayerAvatar name={player.name} size="sm" />
                    <span className="max-w-full truncate text-xs font-medium text-foreground">
                      {player.name}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {ready && eloPrediction && tsPrediction && (
        <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
          <h2 className="mb-4 font-semibold text-card-foreground">预测结果（A 队胜率）</h2>
          <div className="space-y-4">
            <PredictionBar label="ELO" value={eloPrediction.teamAWin} color="bg-blue-500" />
            <PredictionBar label="TrueSkill" value={tsPrediction} color="bg-orange-500" />
          </div>
          <p className="mt-4 text-xs text-muted-foreground">
            基于当前双评分系统计算，仅供参考。
          </p>
        </div>
      )}

      {!ready && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-center text-sm text-amber-800">
          请为两队各选 2 人
        </div>
      )}
    </div>
  );
}

function TeamPanel({
  label,
  accent,
  slots,
  playerMap,
}: {
  label: string;
  accent: string;
  slots: [Slot, Slot];
  playerMap: Map<number, { id: number; name: string }>;
}) {
  return (
    <div className={`rounded-xl p-3 ring-1 ${accent}`}>
      <div className="mb-2 text-center text-xs font-semibold uppercase tracking-wider opacity-80">
        {label}
      </div>
      <div className="flex justify-center gap-2">
        {slots.map((id, i) => {
          const player = id ? playerMap.get(id) : null;
          return (
            <div
              key={i}
              className="flex h-16 w-16 flex-col items-center justify-center rounded-xl bg-white/70 shadow-sm"
            >
              {player ? (
                <>
                  <PlayerAvatar name={player.name} size="sm" />
                  <span className="mt-1 max-w-[3.5rem] truncate text-[10px] font-medium">
                    {player.name}
                  </span>
                </>
              ) : (
                <span className="text-xs text-muted-foreground">待选</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PredictionBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="font-medium text-card-foreground">{label}</span>
        <span className="tabular-nums font-semibold text-card-foreground">{pct}%</span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
