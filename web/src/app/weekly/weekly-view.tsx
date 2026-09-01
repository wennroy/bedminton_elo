"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { PlayerAvatar } from "@/components/player-avatar";
import { Button } from "@/components/ui/button";
import type { WeeklyStats } from "@/lib/weekly";
import { Share2, TrendingDown, TrendingUp } from "lucide-react";

interface WeeklyViewProps {
  stats: WeeklyStats;
  weekStarts: string[];
}

export function WeeklyView({ stats, weekStarts }: WeeklyViewProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  function handleWeekChange(value: string) {
    const params = new URLSearchParams(searchParams.toString());
    params.set("week", value);
    router.push(`/weekly?${params.toString()}`);
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-xl font-bold text-foreground">周报</h1>
        <select
          value={stats.weekStart}
          onChange={(e) => handleWeekChange(e.target.value)}
          className="h-9 rounded-lg border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {weekStarts.map((start) => (
            <option key={start} value={start}>
              {start} 第 {getWeekNumber(start)} 周
            </option>
          ))}
        </select>
      </div>

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-1 text-sm text-muted-foreground">
          {stats.weekStart} ~ {stats.weekEnd}
        </div>
        <div className="text-2xl font-bold text-card-foreground">
          第 {stats.weekNumber} 周战报
        </div>
      </div>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">出勤榜</h2>
        {stats.attendance.length === 0 ? (
          <Empty />
        ) : (
          <div className="space-y-2">
            {stats.attendance.slice(0, 5).map((s, i) => (
              <RankRow
                key={s.playerId}
                rank={i + 1}
                name={s.name}
                value={`${s.matches} 场`}
              />
            ))}
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">战绩王</h2>
        {stats.winKing.length === 0 ? (
          <Empty />
        ) : (
          <div className="space-y-2">
            {stats.winKing.slice(0, 5).map((s, i) => (
              <RankRow
                key={s.playerId}
                rank={i + 1}
                name={s.name}
                value={`${s.wins} 胜 ${s.losses} 负`}
              />
            ))}
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">ELO 涨跌榜</h2>
        {stats.eloChanges.length === 0 ? (
          <Empty />
        ) : (
          <div className="space-y-2">
            {stats.eloChanges.slice(0, 5).map((s, i) => (
              <RankRow
                key={s.playerId}
                rank={i + 1}
                name={s.name}
                value={
                  <span className={s.change >= 0 ? "text-emerald-600" : "text-rose-600"}>
                    {s.change >= 0 ? "+" : ""}
                    {s.change}
                    {s.change >= 0 ? (
                      <TrendingUp className="ml-1 inline size-4" />
                    ) : (
                      <TrendingDown className="ml-1 inline size-4" />
                    )}
                  </span>
                }
              />
            ))}
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">最佳组合</h2>
        {stats.bestPair ? (
          <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex -space-x-2">
                <PlayerAvatar name={stats.bestPair.playerA} size="sm" />
                <PlayerAvatar name={stats.bestPair.playerB} size="sm" />
              </div>
              <div className="text-lg font-bold text-card-foreground">
                {stats.bestPair.playerA} / {stats.bestPair.playerB}
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              {stats.bestPair.wins} 胜 {stats.bestPair.total - stats.bestPair.wins} 负 · {" "}
              {Math.round(stats.bestPair.winRate * 100)}% 胜率
            </div>
          </div>
        ) : (
          <Empty />
        )}
      </section>

      <Button
        size="lg"
        className="h-14 w-full text-lg"
        onClick={() =>
          window.open(`/api/og/weekly?week=${stats.weekStart}`, "_blank")
        }
      >
        <Share2 className="mr-2 size-4" />
        生成分享图
      </Button>
    </div>
  );
}

function RankRow({
  rank,
  name,
  value,
}: {
  rank: number;
  name: string;
  value: React.ReactNode;
}) {
  const medalColors = [
    "bg-amber-100 text-amber-700 ring-amber-200",
    "bg-slate-100 text-slate-700 ring-slate-200",
    "bg-orange-100 text-orange-800 ring-orange-200",
  ];
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-border bg-card p-3 shadow-sm">
      <div
        className={`flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold ring-1 ${
          medalColors[rank - 1] ?? "bg-muted text-muted-foreground"
        }`}
      >
        {rank}
      </div>
      <PlayerAvatar name={name} size="xs" />
      <span className="flex-1 font-medium text-card-foreground">{name}</span>
      <span className="text-sm tabular-nums font-semibold text-card-foreground">
        {value}
      </span>
    </div>
  );
}

function Empty() {
  return (
    <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
      本周无数据
    </div>
  );
}

function getWeekNumber(weekStart: string): number {
  const [y, m, d] = weekStart.split("-").map(Number);
  const startOfYear = new Date(y, 0, 1);
  const monday = new Date(y, m - 1, d);
  const diffMs = monday.getTime() - startOfYear.getTime();
  return Math.floor(diffMs / (7 * 24 * 60 * 60 * 1000)) + 1;
}
