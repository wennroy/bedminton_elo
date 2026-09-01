"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { getMyPlayerId, setMyPlayerId } from "@/lib/identity";
import { PlayerAvatar } from "@/components/player-avatar";
import { IdentityPicker } from "@/components/identity-picker";
import { EloDeltaCard, type EloDeltaPlayer } from "@/components/elo-delta-card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Minus, Plus, RotateCcw, Trophy, UserPlus } from "lucide-react";

interface Player {
  id: number;
  name: string;
}

interface RecordFormProps {
  players: Player[];
  /** 配对页跳转带来的预填阵容(A1,A2,B1,B2),缺省全空 */
  initialSlots?: [Slot, Slot, Slot, Slot];
}

type Slot = number | null;

function todayString(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function RecordForm({ players, initialSlots }: RecordFormProps) {
  const router = useRouter();
  const [myId, setMyId] = React.useState<number | null>(null);
  const [slots, setSlots] = React.useState<[Slot, Slot, Slot, Slot]>(
    initialSlots ?? [null, null, null, null]
  );
  const [scoreA, setScoreA] = React.useState(21);
  const [scoreB, setScoreB] = React.useState(0);
  const [submitting, setSubmitting] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<EloDeltaPlayer[] | null>(null);
  const [addOpen, setAddOpen] = React.useState(false);
  const [addName, setAddName] = React.useState("");
  const [addError, setAddError] = React.useState<string | null>(null);
  const [adding, setAdding] = React.useState(false);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const playerMap = React.useMemo(
    () => new Map(players.map((p) => [p.id, p])),
    [players]
  );

  const selectedIds = slots.filter((id): id is number => id !== null);
  const isDistinct = new Set(selectedIds).size === selectedIds.length;
  const canSubmit =
    selectedIds.length === 4 && isDistinct && scoreA !== scoreB && myId !== null;

  function togglePlayer(id: number) {
    setError(null);
    setSlots((current) => {
      const index = current.indexOf(id);
      if (index !== -1) {
        const next: [Slot, Slot, Slot, Slot] = [...current];
        next[index] = null;
        return next;
      }
      const emptyIndex = current.indexOf(null);
      if (emptyIndex === -1) return current;
      const next: [Slot, Slot, Slot, Slot] = [...current];
      next[emptyIndex] = id;
      return next;
    });
  }

  function clearSelection() {
    setSlots([null, null, null, null]);
  }

  function adjustScore(
    setter: React.Dispatch<React.SetStateAction<number>>,
    delta: number
  ) {
    setter((value) => Math.max(0, Math.min(99, value + delta)));
  }

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch("/api/matches", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pa1: slots[0],
          pa2: slots[1],
          pb1: slots[2],
          pb2: slots[3],
          scoreA,
          scoreB,
          playedAt: todayString(),
          enteredBy: myId,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.error || "提交失败");
        return;
      }
      const deltas: EloDeltaPlayer[] = data.after.map(
        (item: { id: number; name: string; elo: number }, index: number) => ({
          id: item.id,
          name: item.name,
          before: data.before[index].elo,
          after: item.elo,
        })
      );
      setResult(deltas);
    } catch (e) {
      setError(e instanceof Error ? e.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  }

  function handleReset() {
    setSlots([null, null, null, null]);
    setScoreA(21);
    setScoreB(0);
    setResult(null);
    setError(null);
    router.refresh();
  }

  // Courtside case: recorder adds someone ELSE, so identity stays untouched.
  async function handleAddPlayer(event: React.FormEvent) {
    event.preventDefault();
    const name = addName.trim();
    if (!name || adding) return;
    setAdding(true);
    setAddError(null);
    try {
      const res = await fetch("/api/players", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) {
        setAddError(data.error || "添加失败");
        return;
      }
      setAddOpen(false);
      setAddName("");
      // Re-fetch server props so the new player appears in the grid below.
      router.refresh();
    } catch {
      setAddError("网络错误，请重试");
    } finally {
      setAdding(false);
    }
  }

  const teamAScoreColor = scoreA > scoreB ? "text-emerald-600" : scoreA < scoreB ? "text-rose-600" : "text-foreground";
  const teamBScoreColor = scoreB > scoreA ? "text-emerald-600" : scoreB < scoreA ? "text-rose-600" : "text-foreground";

  return (
    <div className="flex flex-col gap-6">
      <IdentityPicker
        players={players}
        onSelect={(id) => {
          setMyPlayerId(id);
          setMyId(id);
        }}
      />

      {!myId && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-center text-sm text-amber-800">
          请先选择你的身份，再录入比赛
        </div>
      )}

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold text-card-foreground">选择阵容</h2>
          {selectedIds.length > 0 && (
            <button
              onClick={clearSelection}
              className="flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <RotateCcw className="size-3" />
              清空
            </button>
          )}
        </div>

        <div className="mb-4 grid grid-cols-2 gap-3">
          <TeamSlots
            label="A 队"
            slots={[slots[0], slots[1]]}
            playerMap={playerMap}
            accent="bg-blue-50 text-blue-700 ring-blue-200"
          />
          <TeamSlots
            label="B 队"
            slots={[slots[2], slots[3]]}
            playerMap={playerMap}
            accent="bg-orange-50 text-orange-700 ring-orange-200"
          />
        </div>

        <div className="grid grid-cols-4 gap-2 sm:grid-cols-5">
          {players.map((player) => {
            const selected = slots.includes(player.id);
            return (
              <button
                key={player.id}
                onClick={() => togglePlayer(player.id)}
                className={`flex flex-col items-center gap-1 rounded-xl border p-2 transition-all ${
                  selected
                    ? "border-primary bg-primary/5 opacity-60"
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
          <button
            onClick={() => setAddOpen(true)}
            className="flex flex-col items-center gap-1 rounded-xl border border-dashed border-border p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-muted">
              <UserPlus className="size-4" />
            </span>
            <span className="text-xs font-medium">新球员</span>
          </button>
        </div>
      </div>

      <Dialog open={addOpen} onOpenChange={setAddOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>添加新球员</DialogTitle>
            <DialogDescription>
              添加后TA就会出现在阵容列表里，不会改变你自己的身份。
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleAddPlayer} className="flex flex-col gap-3 pt-2">
            <input
              autoFocus
              value={addName}
              onChange={(e) => {
                setAddName(e.target.value);
                setAddError(null);
              }}
              maxLength={20}
              placeholder="输入名字"
              className="h-12 w-full rounded-xl border border-border bg-background px-4 text-base outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            {addError && (
              <p className="text-sm text-destructive">{addError}</p>
            )}
            <Button type="submit" disabled={adding || !addName.trim()}>
              {adding ? "添加中…" : "确认添加"}
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <h2 className="mb-4 font-semibold text-card-foreground">比分</h2>
        <div className="flex items-center justify-around">
          <ScoreStepper
            label="A 队"
            value={scoreA}
            onChange={setScoreA}
            onAdjust={(delta) => adjustScore(setScoreA, delta)}
            colorClass={teamAScoreColor}
          />
          <div className="text-2xl font-bold text-muted-foreground">:</div>
          <ScoreStepper
            label="B 队"
            value={scoreB}
            onChange={setScoreB}
            onAdjust={(delta) => adjustScore(setScoreB, delta)}
            colorClass={teamBScoreColor}
          />
        </div>
        {scoreA === scoreB && (
          <p className="mt-3 text-center text-sm text-destructive">
            比分不能相同
          </p>
        )}
      </div>

      {error && (
        <div className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-center text-sm text-destructive">
          {error}
        </div>
      )}

      {!result ? (
        <Button
          size="lg"
          disabled={!canSubmit || submitting}
          onClick={handleSubmit}
          className="h-14 w-full text-lg"
        >
          {submitting ? "提交中…" : "提交比赛"}
        </Button>
      ) : (
        <div className="flex flex-col gap-4">
          <EloDeltaCard players={result} />
          <div className="flex items-center justify-center gap-2 rounded-xl bg-amber-50 p-3 text-sm text-amber-800">
            <Trophy className="size-4" />
            比赛已记录
          </div>
          <Button size="lg" variant="outline" onClick={handleReset} className="h-12 w-full">
            再记一场
          </Button>
        </div>
      )}
    </div>
  );
}

function TeamSlots({
  label,
  slots,
  playerMap,
  accent,
}: {
  label: string;
  slots: [Slot, Slot];
  playerMap: Map<number, Player>;
  accent: string;
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

function ScoreStepper({
  label,
  value,
  onChange,
  onAdjust,
  colorClass,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  onAdjust: (delta: number) => void;
  colorClass: string;
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <button
        onClick={() => onAdjust(1)}
        className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
        aria-label={`${label} 加分`}
      >
        <Plus className="size-6" />
      </button>
      <input
        type="number"
        min={0}
        max={99}
        value={value}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) onChange(Math.max(0, Math.min(99, n)));
        }}
        className={`h-16 w-20 rounded-xl border border-border bg-background text-center text-4xl font-bold outline-none focus-visible:ring-2 focus-visible:ring-ring ${colorClass}`}
      />
      <button
        onClick={() => onAdjust(-1)}
        className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
        aria-label={`${label} 减分`}
      >
        <Minus className="size-6" />
      </button>
    </div>
  );
}
