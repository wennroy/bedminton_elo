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
import { PlayerAvatar } from "@/components/player-avatar";

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
type Mode = "compare" | "rank";

const RANGES: { key: RangeKey; label: string; days: number | null }[] = [
  { key: "2w", label: "近2周", days: 14 },
  { key: "4w", label: "近4周", days: 28 },
  { key: "all", label: "全部", days: null },
];

const MODES: { key: Mode; label: string }[] = [
  { key: "compare", label: "对比" },
  { key: "rank", label: "排名" },
];

const COLORS = [
  "hsl(220 90% 56%)",
  "hsl(340 82% 52%)",
  "hsl(160 84% 39%)",
  "hsl(38 92% 50%)",
  "hsl(270 67% 47%)",
  "hsl(190 95% 39%)",
  "hsl(25 95% 53%)",
  "hsl(142 71% 45%)",
  "hsl(295 72% 50%)",
  "hsl(205 90% 50%)",
  "hsl(0 72% 51%)",
  "hsl(85 75% 40%)",
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
  const [mode, setMode] = React.useState<Mode>("compare");
  // localStorage 只能在客户端挂载后读,避免 hydration 不一致
  const [myId, setMyId] = React.useState<number | null>(null);
  /** 对比 tab 的聚焦人选;null = 尚未初始化(等 effect 读身份) */
  const [selected, setSelected] = React.useState<Set<number> | null>(null);

  // 全部球员(id 升序,保证颜色稳定)
  const allPlayers = React.useMemo(
    () =>
      Array.from(
        new Map(history.map((h) => [h.playerId, h.playerName])).entries()
      ).sort((a, b) => Number(a[0]) - Number(b[0])),
    [history]
  );

  React.useEffect(() => {
    const my = getMyPlayerId();
    setMyId(my);
    // 默认聚焦:选了身份就只看自己,没选就全员
    setSelected(
      my !== null
        ? new Set([my])
        : new Set(allPlayers.map(([id]) => Number(id)))
    );
  }, [allPlayers]);

  const colorOf = React.useCallback(
    (playerId: string) => {
      const idx = allPlayers.findIndex(([id]) => id === playerId);
      return COLORS[idx % COLORS.length];
    },
    [allPlayers]
  );

  const { data, rankData } = React.useMemo(() => {
    const cfg = RANGES.find((r) => r.key === range)!;
    const cutoff = cfg.days === null ? null : cutoffDate(cfg.days);
    const filtered = cutoff
      ? history.filter((h) => h.date >= cutoff)
      : history;

    const byDate = new Map<string, Map<string, number>>();
    for (const h of filtered) {
      if (!byDate.has(h.date)) byDate.set(h.date, new Map());
      byDate.get(h.date)!.set(h.playerId, h.elo);
    }
    const dates = Array.from(byDate.keys()).sort();

    // 对比图:每天一行,每人一个字段(ELO 值)
    const data = dates.map((date) => {
      const row: Record<string, number | string> = { date };
      for (const [pid, elo] of byDate.get(date)!) row[pid] = elo;
      return row;
    });

    // 排名图:同一天按 ELO 降序排出名次
    const rankData = dates.map((date) => {
      const row: Record<string, number | string> = { date };
      const entries = Array.from(byDate.get(date)!.entries()).sort(
        (a, b) => b[1] - a[1]
      );
      entries.forEach(([pid], i) => {
        row[pid] = i + 1;
      });
      return row;
    });

    return { data, rankData };
  }, [history, range]);

  if (history.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
        还没有 ELO 数据
      </div>
    );
  }

  const mySummary = myId === null ? undefined : summaries[myId];
  const myName =
    myId === null
      ? null
      : allPlayers.find(([id]) => Number(id) === myId)?.[1] ?? null;

  function togglePlayer(id: number) {
    setSelected((prev) => {
      const next = new Set(prev ?? []);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const chartData = mode === "compare" ? data : rankData;

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

      <div className="mb-2 inline-flex rounded-lg bg-muted p-0.5">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            onClick={() => setMode(m.key)}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-all ${
              mode === m.key
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {mode === "compare" &&
        (myId !== null && mySummary && myName ? (
          <p className="mb-2 text-sm text-muted-foreground">
            <span className="font-medium text-card-foreground">{myName}</span>
            {" · "}
            当前{" "}
            <span className="font-bold tabular-nums">{mySummary.elo}</span>
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
        ))}

      <div className="h-56 w-full sm:h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 4, right: 4, bottom: 4, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
              tickMargin={6}
              minTickGap={32}
            />
            {mode === "compare" ? (
              <YAxis
                domain={["dataMin - 30", "dataMax + 30"]}
                tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                width={36}
              />
            ) : (
              <YAxis
                reversed
                domain={[1, allPlayers.length]}
                allowDecimals={false}
                tickCount={Math.min(6, allPlayers.length)}
                tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                width={24}
              />
            )}
            <Tooltip
              contentStyle={{
                background: "var(--card)",
                border: "1px solid var(--border)",
                borderRadius: "0.75rem",
                fontSize: "0.75rem",
              }}
              labelStyle={{
                color: "var(--foreground)",
                marginBottom: "0.25rem",
              }}
              itemStyle={{ color: "var(--foreground)" }}
              formatter={(value, name) => [
                mode === "rank" ? `第 ${value} 名` : String(value),
                String(name),
              ]}
            />
            {allPlayers.map(([playerId, playerName]) => {
              const pid = Number(playerId);
              const isMe = myId !== null && pid === myId;
              const isSelected = selected?.has(pid) ?? false;
              const colored = mode === "rank" || isSelected;
              return (
                <Line
                  key={playerId}
                  type="monotone"
                  dataKey={playerId}
                  name={playerName}
                  stroke={
                    isMe
                      ? "var(--primary)"
                      : colored
                      ? colorOf(playerId)
                      : OTHER_LINE
                  }
                  strokeOpacity={colored ? (isMe ? 1 : 0.8) : 0.25}
                  strokeWidth={isMe ? 3 : colored ? 2 : 1.5}
                  dot={false}
                  activeDot={{ r: isMe ? 5 : 3 }}
                  connectNulls
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {mode === "compare" && (
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
          <button
            type="button"
            onClick={() =>
              setSelected(new Set(allPlayers.map(([id]) => Number(id))))
            }
            className="flex shrink-0 items-center rounded-full border border-border px-3 py-1 text-xs text-muted-foreground"
          >
            全选
          </button>
          {allPlayers.map(([playerId, playerName]) => {
            const pid = Number(playerId);
            const isSelected = selected?.has(pid) ?? false;
            return (
              <button
                key={playerId}
                type="button"
                onClick={() => togglePlayer(pid)}
                className={`flex shrink-0 items-center gap-1.5 rounded-full border px-2 py-1 text-xs transition-all ${
                  isSelected
                    ? "border-primary bg-primary/10 text-foreground"
                    : "border-border text-muted-foreground opacity-50"
                }`}
              >
                <PlayerAvatar name={playerName} size="xs" />
                {playerName}
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
