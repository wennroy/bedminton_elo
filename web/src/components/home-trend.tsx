"use client";

import * as React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { EloHistoryPoint } from "@/lib/stats";
import { getMyPlayerId } from "@/lib/identity";

export interface PlayerSummaryLite {
  elo: number;
  rank: number;
  weekDelta: number;
}

interface HomeTrendProps {
  history: EloHistoryPoint[];
  /** key 为球员 id;已选身份时用来显示个人摘要 */
  summaries: Record<number, PlayerSummaryLite>;
}

type RangeKey = "2w" | "4w" | "all";

const RANGES: { key: RangeKey; label: string; days: number | null }[] = [
  { key: "2w", label: "近2周", days: 14 },
  { key: "4w", label: "近4周", days: 28 },
  { key: "all", label: "全部", days: null },
];

const OTHER_LINE = "var(--muted-foreground)";

function cutoffDate(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function HomeTrend({ history, summaries }: HomeTrendProps) {
  const [range, setRange] = React.useState<RangeKey>("4w");
  // localStorage 只能在客户端挂载后读,避免 hydration 不一致
  const [myId, setMyId] = React.useState<number | null>(null);
  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const { data, players, myName } = React.useMemo(() => {
    const cfg = RANGES.find((r) => r.key === range)!;
    const cutoff = cfg.days === null ? null : cutoffDate(cfg.days);
    const filtered = cutoff
      ? history.filter((h) => h.date >= cutoff)
      : history;

    const players = Array.from(
      new Map(filtered.map((h) => [h.playerId, h.playerName])).entries()
    );
    const byDate = new Map<string, Record<string, number | string>>();
    for (const h of filtered) {
      if (!byDate.has(h.date)) byDate.set(h.date, { date: h.date });
      byDate.get(h.date)![h.playerId] = h.elo;
    }
    const data = Array.from(byDate.values()).sort((a, b) =>
      String(a.date).localeCompare(String(b.date))
    );
    const myName =
      myId === null
        ? null
        : players.find(([id]) => Number(id) === myId)?.[1] ?? null;
    return { data, players, myName };
  }, [history, range, myId]);

  if (history.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
        还没有 ELO 数据
      </div>
    );
  }

  const mySummary = myId === null ? undefined : summaries[myId];

  return (
    <section className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-1 flex items-center justify-between">
        <h1 className="text-lg font-bold text-card-foreground">ELO 趋势</h1>
        <div className="inline-flex rounded-lg bg-muted p-0.5">
          {RANGES.map((r) => (
            <button
              key={r.key}
              type="button"
              onClick={() => setRange(r.key)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-all ${
                range === r.key
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground"
              }`}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {myId !== null && mySummary && myName ? (
        <p className="mb-2 text-sm text-muted-foreground">
          <span className="font-medium text-card-foreground">{myName}</span>
          {" · "}
          当前 <span className="font-bold tabular-nums">{mySummary.elo}</span>
          {" · 本周 "}
          <span
            className={`font-bold tabular-nums ${
              mySummary.weekDelta > 0
                ? "text-green-600"
                : mySummary.weekDelta < 0
                ? "text-red-500"
                : ""
            }`}
          >
            {mySummary.weekDelta > 0 ? "+" : ""}
            {mySummary.weekDelta}
          </span>
          {" · 第 "}
          <span className="font-bold tabular-nums">{mySummary.rank}</span> 名
        </p>
      ) : (
        <p className="mb-2 text-xs text-muted-foreground">
          选择「我是谁」后,这里会高亮你的曲线
        </p>
      )}

      <div className="h-56 w-full sm:h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
              tickMargin={6}
              minTickGap={32}
            />
            <YAxis
              domain={["dataMin - 30", "dataMax + 30"]}
              tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
              width={36}
            />
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: "0.75rem",
                fontSize: "0.75rem",
              }}
              labelStyle={{ color: "var(--foreground)", marginBottom: "0.25rem" }}
              itemStyle={{ color: "var(--foreground)" }}
              formatter={(value, name) => [String(value), String(name)]}
            />
            {players.map(([playerId, playerName]) => {
              const isMe = myId !== null && Number(playerId) === myId;
              return (
                <Line
                  key={playerId}
                  type="monotone"
                  dataKey={playerId}
                  name={playerName}
                  stroke={isMe ? "var(--primary)" : OTHER_LINE}
                  strokeOpacity={myId === null ? 0.85 : isMe ? 1 : 0.3}
                  strokeWidth={isMe ? 3 : 1.5}
                  dot={false}
                  activeDot={{ r: isMe ? 5 : 3 }}
                  connectNulls
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
