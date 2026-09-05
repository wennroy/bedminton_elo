"use client";

import * as React from "react";
import Link from "next/link";
import { PlayerAvatar } from "@/components/player-avatar";
import { Button } from "@/components/ui/button";
import { getMyPlayerId } from "@/lib/identity";
import {
  clearSchedule,
  loadSchedule,
  saveSchedule,
  type ScheduleResult,
} from "@/lib/schedule-storage";
import { ChevronDown, ChevronRight, ChevronUp, RotateCcw, Sparkles } from "lucide-react";

interface Player {
  id: number;
  name: string;
}

interface ScheduleFormProps {
  players: Player[];
}

export function ScheduleForm({ players }: ScheduleFormProps) {
  const [selected, setSelected] = React.useState<Set<number>>(new Set());
  const [matches, setMatches] = React.useState(4);
  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  const [seed, setSeed] = React.useState(42);
  const [lambda, setLambda] = React.useState(0.5);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<ScheduleResult | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  // 按当前身份恢复上次生成的配对(只有「清空」或再次「生成配对」才刷新)
  React.useEffect(() => {
    const stored = loadSchedule(getMyPlayerId());
    if (!stored) return;
    setSelected(new Set(stored.playerIds));
    setMatches(stored.matches);
    setSeed(stored.seed);
    setLambda(stored.lambda);
    setResult(stored.result);
  }, []);

  const canGenerate = selected.size >= 4 && matches >= 1;

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    setResult(null);
  }

  function clear() {
    setSelected(new Set());
    setResult(null);
    clearSchedule(getMyPlayerId());
  }

  async function handleGenerate() {
    if (!canGenerate) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch("/api/schedule", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          playerIds: Array.from(selected),
          matches,
          seed,
          lambda,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.error || "生成失败");
        return;
      }
      setResult(data);
      saveSchedule(getMyPlayerId(), {
        playerIds: Array.from(selected),
        matches,
        seed,
        lambda,
        result: data,
        savedAt: new Date().toISOString(),
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "生成失败");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold text-card-foreground">在场球员</h2>
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
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-5">
          {players.map((player) => {
            const active = selected.has(player.id);
            return (
              <button
                key={player.id}
                onClick={() => toggle(player.id)}
                className={`flex flex-col items-center gap-1 rounded-xl border p-2 transition-all ${
                  active
                    ? "border-primary bg-primary/5"
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
        <p className="mt-3 text-xs text-muted-foreground">
          已选 {selected.size} 人{selected.size < 4 && "（至少 4 人）"}
        </p>
      </div>

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <h2 className="mb-4 font-semibold text-card-foreground">总场次</h2>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setMatches((m) => Math.max(1, m - 1))}
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
            aria-label="减少场次"
          >
            −
          </button>
          <input
            type="number"
            min={1}
            max={99}
            value={matches}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n)) setMatches(Math.max(1, Math.min(99, n)));
            }}
            className="h-12 w-20 rounded-xl border border-border bg-background text-center text-xl font-bold outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <button
            onClick={() => setMatches((m) => Math.min(99, m + 1))}
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
            aria-label="增加场次"
          >
            +
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card shadow-sm">
        <button
          onClick={() => setAdvancedOpen((v) => !v)}
          className="flex w-full items-center justify-between p-4 text-left"
        >
          <span className="font-semibold text-card-foreground">高级选项</span>
          {advancedOpen ? (
            <ChevronUp className="size-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="size-4 text-muted-foreground" />
          )}
        </button>
        {advancedOpen && (
          <div className="space-y-4 border-t border-border px-4 pb-4 pt-4">
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">
                随机种子
              </label>
              <input
                type="number"
                value={seed}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n)) setSeed(n);
                }}
                className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">
                温度 λ（{lambda}）
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={lambda}
                onChange={(e) => setLambda(Number(e.target.value))}
                className="w-full"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                越接近 1 越注重阵容多样性，越接近 0 越注重实力均衡
              </p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-center text-sm text-destructive">
          {error}
        </div>
      )}

      <Button
        size="lg"
        disabled={!canGenerate || loading}
        onClick={handleGenerate}
        className="h-14 w-full text-lg"
      >
        {loading ? "生成中…" : "生成配对"}
        {!loading && <Sparkles className="ml-2 size-4" />}
      </Button>

      {result && <ScheduleResultView result={result} />}
    </div>
  );
}

function ScheduleResultView({ result }: { result: ScheduleResult }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between">
        <h2 className="text-lg font-bold text-foreground">生成结果</h2>
        <span className="text-xs text-muted-foreground">
          点击场次卡片直接记分
        </span>
      </div>
      <div className="space-y-3">
        {result.schedule.map((match, index) => (
          <div
            key={index}
            className="rounded-2xl border border-border bg-card p-3 shadow-sm transition-colors hover:border-primary/50"
          >
            <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
              <span>第 {index + 1} 场</span>
              <Link
                href={`/predict?pa1=${match.a1}&pa2=${match.a2}&pb1=${match.b1}&pb2=${match.b2}`}
                className="flex items-center gap-1 rounded-md px-1 py-0.5 transition-colors hover:text-primary"
              >
                A 队胜率 {Math.round(match.winRate * 100)}%
                <ChevronRight className="size-3" />
              </Link>
            </div>
            <Link
              href={`/record?pa1=${match.a1}&pa2=${match.a2}&pb1=${match.b1}&pb2=${match.b2}`}
              className="-mx-1 block rounded-xl px-1 transition-colors hover:bg-muted/30"
            >
              <div className="flex items-center justify-between gap-2">
                <TeamView ids={[match.a1, match.a2]} names={result.names} />
                <span className="text-sm font-bold text-muted-foreground">VS</span>
                <TeamView ids={[match.b1, match.b2]} names={result.names} />
              </div>
            </Link>
          </div>
        ))}
      </div>

      <div className="rounded-xl bg-muted p-3 text-xs text-muted-foreground">
        <div className="grid grid-cols-2 gap-2">
          <div>出场方差: {result.metrics.alphaVar.toFixed(2)}</div>
          <div>平均接近度: {result.metrics.meanCloseness.toFixed(3)}</div>
          <div>最大接近度: {result.metrics.maxCloseness.toFixed(3)}</div>
          <div>阵容熵: {result.metrics.entropy.toFixed(3)}</div>
        </div>
      </div>
    </div>
  );
}

function TeamView({
  ids,
  names,
}: {
  ids: string[];
  names: Record<string, string>;
}) {
  return (
    <div className="flex flex-1 flex-col items-center gap-1">
      <div className="flex -space-x-2">
        {ids.map((id) => (
          <PlayerAvatar key={id} name={names[id] ?? "?"} size="xs" />
        ))}
      </div>
      <span className="max-w-full truncate text-center text-xs font-medium text-card-foreground">
        {ids.map((id) => names[id] ?? "?").join(" / ")}
      </span>
    </div>
  );
}
