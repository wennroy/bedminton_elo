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

  const [activeSlot, setActiveSlot] = React.useState(() => {
    const empty = [...(initialTeamA ?? [null, null]), ...(initialTeamB ?? [null, null])].indexOf(null);
    return empty === -1 ? 0 : empty;
  });

  const playerMap = React.useMemo(
    () => new Map(players.map((p) => [p.id, p])),
    [players]
  );

  const selected = new Set([
    teamA[0], teamA[1], teamB[0], teamB[1],
  ].filter((id): id is number => id !== null));

  const teamAIds = React.useMemo(() => teamA.filter((id): id is number => id !== null), [teamA]);
  const teamBIds = React.useMemo(() => teamB.filter((id): id is number => id !== null), [teamB]);
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

  const slots = [...teamA, ...teamB];
  const activeLabel = `${activeSlot < 2 ? "A" : "B"} 队第 ${(activeSlot % 2) + 1} 位`;

  function selectPlayer(id: number) {
    if (selected.has(id)) return;
    const next = [...slots];
    next[activeSlot] = id;
    setTeamA([next[0], next[1]]);
    setTeamB([next[2], next[3]]);
    // 从当前位继续补空位；选满后保留当前位，方便直接换人。
    for (let offset = 1; offset < 4; offset++) {
      const index = (activeSlot + offset) % 4;
      if (next[index] === null) {
        setActiveSlot(index);
        break;
      }
    }
  }

  function clear() {
    setTeamA([null, null]);
    setTeamB([null, null]);
    setActiveSlot(0);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="sticky top-0 z-20 -mx-1 space-y-3 rounded-b-2xl bg-background px-1 py-3 shadow-sm">
        <div className="grid grid-cols-2 gap-3">
          <TeamPanel
            label="A 队"
            side="a"
            slots={teamA}
            activeSlot={activeSlot < 2 ? activeSlot : null}
            onSelectSlot={setActiveSlot}
            playerMap={playerMap}
          />
          <TeamPanel
            label="B 队"
            side="b"
            slots={teamB}
            activeSlot={activeSlot >= 2 ? activeSlot - 2 : null}
            onSelectSlot={(index) => setActiveSlot(index + 2)}
            playerMap={playerMap}
          />
        </div>
        {eloPrediction ? (
          <div role="status" aria-label="预测胜率" className="rounded-xl border border-border bg-card px-3 py-2.5">
            <WinProbability teamAWin={eloPrediction.teamAWin} />
          </div>
        ) : (
          <p role="status" className="text-center text-xs text-muted-foreground">
            请为两队各选 2 人（已选 {selected.size}/4）
          </p>
        )}
      </div>

      {ready && eloPrediction && (
        <PredictResult
          teamAPlayers={teamAIds.map((id) => playerMap.get(id)!)}
          teamBPlayers={teamBIds.map((id) => playerMap.get(id)!)}
          deltas={eloPrediction.deltas}
        />
      )}

      <section aria-labelledby="player-picker-heading" className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-2 flex items-center justify-between">
          <h2 id="player-picker-heading" className="font-semibold text-card-foreground">选择球员</h2>
          {selected.size > 0 && (
            <button
              type="button"
              onClick={clear}
              className="flex min-h-11 items-center gap-1 rounded text-xs text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring"
            >
              <RotateCcw className="size-3" />
              清空
            </button>
          )}
        </div>
        <p className="mb-3 text-xs leading-relaxed text-muted-foreground">
          正在选择 <strong className="text-foreground">{activeLabel}</strong> · 点击上方选手框切换，再点下方球员{slots[activeSlot] !== null ? "替换" : "填入"}
        </p>
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-5">
          {players.map((player) => {
            const slotIndex = slots.indexOf(player.id);
            const assigned = slotIndex !== -1;
            return (
              <button
                type="button"
                key={player.id}
                onClick={() => selectPlayer(player.id)}
                disabled={assigned}
                aria-label={assigned ? `${player.name}，已在 ${slotIndex < 2 ? "A" : "B"} 队第 ${(slotIndex % 2) + 1} 位` : `选择 ${player.name}`}
                title={player.name}
                className={`flex min-w-0 flex-col items-center gap-1 rounded-xl border p-2 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring ${
                  assigned
                    ? slotIndex < 2 ? "border-team-a bg-team-a/15" : "border-team-b bg-team-b/15"
                    : "border-border bg-background hover:bg-muted"
                }`}
              >
                <PlayerAvatar name={player.name} size="sm" />
                <span className="max-w-full truncate text-xs font-medium text-foreground">{player.name}</span>
                {assigned && <span className="text-[10px] text-muted-foreground">{slotIndex < 2 ? "A" : "B"} 队 · 已选</span>}
              </button>
            );
          })}
        </div>
        {players.length === 0 && (
          <p className="py-4 text-center text-sm text-muted-foreground">暂无球员，请先添加球员</p>
        )}
      </section>
    </div>
  );
}

function TeamPanel({
  label,
  side,
  slots,
  activeSlot,
  onSelectSlot,
  playerMap,
}: {
  label: string;
  side: "a" | "b";
  slots: [Slot, Slot];
  activeSlot: number | null;
  onSelectSlot: (index: number) => void;
  playerMap: Map<number, { id: number; name: string }>;
}) {
  return (
    <div className={`min-w-0 rounded-xl p-2.5 ring-1 ${side === "a" ? "bg-team-a/15 ring-team-a/50" : "bg-team-b/15 ring-team-b/50"}`}>
      <div className="mb-2 text-center text-xs font-semibold tracking-wider">{label}</div>
      <div className="grid grid-cols-2 gap-2">
        {slots.map((id, i) => {
          const player = id !== null ? playerMap.get(id) : null;
          const active = activeSlot === i;
          return (
            <button
              type="button"
              key={i}
              onClick={() => onSelectSlot(i)}
              aria-pressed={active}
              aria-label={`${label}第 ${i + 1} 位：${player?.name ?? "待选"}`}
              title={player?.name ?? "点击选择球员"}
              className={`flex h-16 min-w-0 flex-col items-center justify-center rounded-xl border bg-card shadow-sm transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring ${
                active ? "border-foreground ring-2 ring-foreground/70" : "border-transparent hover:border-muted-foreground"
              }`}
            >
              {player ? (
                <>
                  <PlayerAvatar name={player.name} size="sm" />
                  <span className="mt-1 max-w-full truncate px-1 text-[10px] font-medium">{player.name}</span>
                </>
              ) : (
                <span className="text-xs text-muted-foreground">{active ? "选择中" : "待选"}</span>
              )}
            </button>
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

function WinProbability({ teamAWin }: { teamAWin: number }) {
  const pctA = Math.round(teamAWin * 100);
  const pctB = 100 - pctA;
  return (
    <>
      <div className="mb-[9px] flex items-baseline justify-between">
        <div className="flex min-w-0 items-center gap-[7px]">
          <span className="size-2 shrink-0 rounded-full bg-team-a" />
          <span className="text-[11px] font-semibold text-muted-foreground">
            A 队胜率
          </span>
          <strong className="font-num text-[17px] leading-none font-bold text-win">
            {pctA}%
          </strong>
        </div>
        <div className="flex min-w-0 items-center gap-[7px]">
          <strong className="font-num text-[17px] leading-none font-bold text-chart-2">
            {pctB}%
          </strong>
          <span className="text-[11px] font-semibold text-muted-foreground">
            B 队胜率
          </span>
          <span className="size-2 shrink-0 rounded-full bg-team-b" />
        </div>
      </div>

      <div className="flex h-3.5 overflow-hidden rounded-full ring-1 ring-foreground/10 ring-inset">
        <div className="bg-team-a" style={{ width: `${pctA}%` }} />
        <div className="w-[3px] shrink-0 bg-card" />
        <div className="flex-1 bg-team-b" />
      </div>

    </>
  );
}

/** 按队分列的逐人 ELO 变化；胜率随上方选手框吸顶显示。 */
function PredictResult({
  teamAPlayers,
  teamBPlayers,
  deltas,
}: {
  teamAPlayers: { id: number; name: string }[];
  teamBPlayers: { id: number; name: string }[];
  deltas: Record<string, EloDelta>;
}) {
  return (
    <section aria-label="预测 ELO 变化" className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <h2 className="text-sm font-semibold">预测 ELO 变化</h2>
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
