"use client";

import * as React from "react";
import { PlayerAvatar } from "@/components/player-avatar";
import { predictElo, predictEloDeltas } from "@/lib/elo";
import { RotateCcw } from "lucide-react";

type Slot = number | null;

interface PredictFormProps {
  players: { id: number; name: string }[];
  ratings: Map<number, { elo: number; mu: number; sigma: number }>;
  initialTeamA?: [Slot, Slot];
  initialTeamB?: [Slot, Slot];
}

export function PredictForm({
  players,
  ratings,
  initialTeamA,
  initialTeamB,
}: PredictFormProps) {
  const [teamA, setTeamA] = React.useState<[Slot, Slot]>(
    initialTeamA ?? [null, null]
  );
  const [teamB, setTeamB] = React.useState<[Slot, Slot]>(
    initialTeamB ?? [null, null]
  );

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
    const a1 = String(teamAIds[0]);
    const a2 = String(teamAIds[1]);
    const b1 = String(teamBIds[0]);
    const b2 = String(teamBIds[1]);
    return {
      teamAWin: predictElo(a1, a2, b1, b2, eloRatings).teamAWin,
      deltas: predictEloDeltas(a1, a2, b1, b2, eloRatings),
    };
  }, [ready, teamAIds, teamBIds, players, ratings]);

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
          accent="bg-team-a/15 text-foreground ring-team-a/50"
          slots={teamA}
          playerMap={playerMap}
        />
        <TeamPanel
          label="B 队"
          accent="bg-team-b/15 text-foreground ring-team-b/50"
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
                        ? "border-team-a bg-team-a/15"
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
                        ? "border-team-b bg-team-b/15"
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

      {ready && eloPrediction && (
        <PredictResult
          teamAWin={eloPrediction.teamAWin}
          teamAPlayers={teamAIds.map((id) => playerMap.get(id)!)}
          teamBPlayers={teamBIds.map((id) => playerMap.get(id)!)}
          deltas={eloPrediction.deltas}
        />
      )}

      {!ready && (
        <div className="rounded-xl border border-border bg-secondary p-4 text-center text-sm text-secondary-foreground">
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
              className="flex h-16 w-16 flex-col items-center justify-center rounded-xl bg-card/70 shadow-sm"
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

interface EloDelta {
  win: number;
  loss: number;
}

/** 预测结果面板：顶部双向胜率条 + 按队分列的逐人 ELO 变化（mock predict-elo-change） */
function PredictResult({
  teamAWin,
  teamAPlayers,
  teamBPlayers,
  deltas,
}: {
  teamAWin: number;
  teamAPlayers: { id: number; name: string }[];
  teamBPlayers: { id: number; name: string }[];
  deltas: Record<string, EloDelta>;
}) {
  const pctA = Math.round(teamAWin * 100);
  const pctB = 100 - pctA;
  return (
    <section className="rounded-2xl border border-border bg-card p-5 shadow-sm">
      <div className="mb-[9px] flex items-baseline justify-between">
        <div className="flex min-w-0 items-center gap-[7px]">
          <span className="size-2 shrink-0 rounded-full bg-team-a" />
          <span className="text-[11px] font-semibold text-muted-foreground">
            A 队
          </span>
          <strong className="font-num text-[17px] leading-none font-bold text-primary-foreground">
            {pctA}%
          </strong>
        </div>
        <div className="flex min-w-0 items-center gap-[7px]">
          <strong className="font-num text-[17px] leading-none font-bold text-chart-2">
            {pctB}%
          </strong>
          <span className="text-[11px] font-semibold text-muted-foreground">
            B 队
          </span>
          <span className="size-2 shrink-0 rounded-full bg-team-b" />
        </div>
      </div>

      <div className="flex h-3.5 overflow-hidden rounded-full ring-1 ring-foreground/10 ring-inset">
        <div className="bg-team-a" style={{ width: `${pctA}%` }} />
        <div className="w-[3px] shrink-0 bg-card" />
        <div className="flex-1 bg-team-b" />
      </div>

      <div className="mt-[22px] grid grid-cols-[minmax(0,1fr)_1px_minmax(0,1fr)] gap-[10px]">
        <PredictTeamColumn
          label="A 队"
          side="a"
          players={teamAPlayers}
          deltas={deltas}
        />
        <div className="bg-border" />
        <PredictTeamColumn
          label="B 队"
          side="b"
          players={teamBPlayers}
          deltas={deltas}
        />
      </div>

      <p className="mt-[18px] border-t border-border pt-3 text-center text-[10px] text-muted-foreground">
        数字为赢/输一场的 ELO 变化，基于当前 ELO，仅供参考
      </p>
    </section>
  );
}

function PredictTeamColumn({
  label,
  side,
  players,
  deltas,
}: {
  label: string;
  side: "a" | "b";
  players: { id: number; name: string }[];
  deltas: Record<string, EloDelta>;
}) {
  return (
    <div className="min-w-0">
      <div
        className={`mb-[11px] flex items-center gap-1.5 text-[10px] font-bold tracking-[1px] text-muted-foreground ${
          side === "b" ? "justify-end" : ""
        }`}
      >
        {side === "a" && <span className="size-2 rounded-full bg-team-a" />}
        {label}
        {side === "b" && <span className="size-2 rounded-full bg-team-b" />}
      </div>
      <div className="space-y-3">
        {players.map((player) => (
          <PredictPlayerRow
            key={player.id}
            player={player}
            delta={deltas[String(player.id)]}
          />
        ))}
      </div>
    </div>
  );
}

function PredictPlayerRow({
  player,
  delta,
}: {
  player: { id: number; name: string };
  delta: EloDelta | undefined;
}) {
  const win = delta ? Math.round(delta.win) : 0;
  const loss = delta ? Math.round(delta.loss) : 0;
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <div className="flex min-w-0 flex-1 items-center gap-1.5">
        <PlayerAvatar
          name={player.name}
          size="xs"
          className="size-6 shrink-0 text-[9px]"
        />
        <span className="min-w-0 truncate text-xs font-semibold text-foreground">
          {player.name}
        </span>
      </div>
      <div className="flex shrink-0 flex-col items-end gap-1">
        <span className="inline-flex items-baseline gap-[3px] rounded-[5px] bg-win-bg px-1.5 pt-[2px] pb-[3px] text-[10px] font-semibold whitespace-nowrap text-win">
          赢 <b className="font-num font-bold">+{win}</b>
        </span>
        <span className="inline-flex items-baseline gap-[3px] rounded-[5px] bg-loss-bg px-1.5 pt-[2px] pb-[3px] text-[10px] font-semibold whitespace-nowrap text-loss">
          输 <b className="font-num font-bold">−{Math.abs(loss)}</b>
        </span>
      </div>
    </div>
  );
}
