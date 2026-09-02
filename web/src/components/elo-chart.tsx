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
  Legend,
} from "recharts";
import type { EloHistoryPoint } from "@/lib/stats";

interface EloChartProps {
  history: EloHistoryPoint[];
}

const CHART_COLORS = [
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
];

export function EloChart({ history }: EloChartProps) {
  const { data, players } = React.useMemo(() => {
    const players = Array.from(
      new Map(history.map((h) => [h.playerId, h.playerName])).entries()
    );
    const byDate = new Map<string, Record<string, number | string>>();
    for (const h of history) {
      if (!byDate.has(h.date)) {
        byDate.set(h.date, { date: h.date });
      }
      byDate.get(h.date)![h.playerId] = h.elo;
    }
    const data = Array.from(byDate.values()).sort((a, b) =>
      String(a.date).localeCompare(String(b.date))
    );
    return { data, players };
  }, [history]);

  if (history.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground">
        还没有 ELO 数据
      </div>
    );
  }

  return (
    <div className="h-80 w-full rounded-2xl border border-border bg-card p-3 shadow-sm sm:h-96">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickMargin={8}
            minTickGap={24}
          />
          <YAxis
            domain={["dataMin - 50", "dataMax + 50"]}
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            width={40}
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
            itemSorter={(item) => {
              // 按 ELO 值降序;connectNulls 缺口日 value 可能为 undefined,兜底排到末尾
              const v = Number(item.value);
              return Number.isNaN(v) ? Infinity : -v;
            }}
            formatter={(value, name) => [String(value), String(name)]}
          />
          <Legend
            wrapperStyle={{ fontSize: "0.75rem", paddingTop: "0.5rem" }}
            iconType="circle"
          />
          {players.map(([playerId, playerName], index) => (
            <Line
              key={playerId}
              type="monotone"
              dataKey={playerId}
              name={playerName}
              stroke={CHART_COLORS[index % CHART_COLORS.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
