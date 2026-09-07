"use client";

import * as React from "react";
import Link from "next/link";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { ArrowRight, Check } from "lucide-react";
import type { EloHistoryPoint } from "@/lib/stats";
import { INITIAL_RATING } from "@/lib/elo";
import { getMyPlayerId } from "@/lib/identity";
import { cn } from "@/lib/utils";

interface PlayerLite {
  id: number;
  name: string;
}

interface HomeTrendProps {
  history: EloHistoryPoint[];
  /** 全部球员（含尚无比赛者）；chips 与排名按 id 升序固定颜色 */
  players: PlayerLite[];
  /** compact = 首页嵌入（带「展开大图」链接）；full = /trends 大图页 */
  variant?: "compact" | "full";
}

type RangeKey = "4" | "12" | "all";
type Mode = "elo" | "rank";

const RANGES: { key: RangeKey; label: string; weeks: number | null }[] = [
  { key: "4", label: "近 4 周", weeks: 4 },
  { key: "12", label: "近 12 周", weeks: 12 },
  { key: "all", label: "全部", weeks: null },
];

const MODES: { key: Mode; label: string }[] = [
  { key: "elo", label: "ELO 积分" },
  { key: "rank", label: "排名" },
];

function localDateString(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function cutoffDate(weeks: number): string {
  const d = new Date();
  d.setDate(d.getDate() - weeks * 7);
  return localDateString(d);
}

/** 每球员固定颜色：--series-1..8 按 id 升序索引循环 */
function seriesVar(index: number): string {
  return `var(--series-${(index % 8) + 1})`;
}

function shortDate(date: string): string {
  return date.slice(5).replace("-", ".");
}

function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { key: T; label: string }[];
  value: T;
  onChange: (key: T) => void;
}) {
  return (
    <div
      className="inline-flex gap-[2px] rounded-[7px] border border-border p-[3px]"
      role="group"
      aria-label={label}
    >
      {options.map((o) => (
        <button
          key={o.key}
          type="button"
          aria-pressed={value === o.key}
          onClick={() => onChange(o.key)}
          className={cn(
            "min-h-[30px] rounded px-2.5 text-[10px] whitespace-nowrap transition-colors max-[760px]:min-h-9 max-[760px]:px-2",
            value === o.key
              ? "bg-secondary font-bold text-foreground"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function HomeTrend({
  history,
  players,
  variant = "full",
}: HomeTrendProps) {
  const compact = variant === "compact";
  const [range, setRange] = React.useState<RangeKey>("12");
  const [mode, setMode] = React.useState<Mode>("elo");
  const [inspectedDate, setInspectedDate] = React.useState<string | null>(null);
  const [myId, setMyId] = React.useState<number | null>(null);

  const sortedPlayers = React.useMemo(
    () => [...players].sort((a, b) => a.id - b.id),
    [players]
  );
  const colorIndexOf = React.useMemo(
    () => new Map(sortedPlayers.map((p, i) => [p.id, i])),
    [sortedPlayers]
  );

  // 默认全员选中（不再默认只看自己）
  const [selected, setSelected] = React.useState<ReadonlySet<number>>(
    () => new Set(sortedPlayers.map((p) => p.id))
  );

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  // 每球员按日升序的快照序列
  const series = React.useMemo(() => {
    const map = new Map<number, { dates: string[]; elos: number[] }>();
    for (const h of history) {
      const id = Number(h.playerId);
      let s = map.get(id);
      if (!s) {
        s = { dates: [], elos: [] };
        map.set(id, s);
      }
      s.dates.push(h.date);
      s.elos.push(h.elo);
    }
    return map;
  }, [history]);

  /** 某日积分：该日前最后一个快照；无快照沿用初始分（不制造虚假早期历史） */
  const ratingAt = React.useCallback(
    (id: number, date: string): number => {
      const s = series.get(id);
      if (!s) return INITIAL_RATING;
      let lo = 0;
      let hi = s.dates.length - 1;
      let ans = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (s.dates[mid] <= date) {
          ans = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      return ans === -1 ? INITIAL_RATING : s.elos[ans];
    },
    [series]
  );

  const allDates = React.useMemo(
    () => Array.from(new Set(history.map((h) => h.date))).sort(),
    [history]
  );

  // 窗口起点：cutoff 日插入前值（ratingAt 取 cutoff 前最后一个快照）
  const windowDates = React.useMemo(() => {
    const cfg = RANGES.find((r) => r.key === range)!;
    if (cfg.weeks === null) return allDates;
    const cutoff = cutoffDate(cfg.weeks);
    return [cutoff, ...allDates.filter((d) => d > cutoff)];
  }, [allDates, range]);

  /** 某日全员名次：ELO 降序，同分按 id 升序（与排行榜口径一致） */
  const rankingAt = React.useCallback(
    (date: string): PlayerLite[] =>
      [...sortedPlayers].sort(
        (a, b) => ratingAt(b.id, date) - ratingAt(a.id, date)
      ),
    [sortedPlayers, ratingAt]
  );

  const selectedPlayers = React.useMemo(
    () => sortedPlayers.filter((p) => selected.has(p.id)),
    [sortedPlayers, selected]
  );

  const chartData = React.useMemo(
    () =>
      windowDates.map((date) => {
        const row: Record<string, number | string> = { date };
        if (mode === "elo") {
          for (const p of selectedPlayers) row[String(p.id)] = ratingAt(p.id, date);
        } else {
          // 名次 = 全员当日名次，筛掉别人不抬自己排名
          const ordered = rankingAt(date);
          ordered.forEach((p, i) => {
            if (selected.has(p.id)) row[String(p.id)] = i + 1;
          });
        }
        return row;
      }),
    [windowDates, mode, selectedPlayers, rankingAt, ratingAt, selected]
  );

  const inspected =
    inspectedDate && windowDates.includes(inspectedDate)
      ? inspectedDate
      : windowDates[windowDates.length - 1];

  // 读数栏：当日选中成员按当日分数降序
  const readoutRows = React.useMemo(() => {
    if (!inspected) return [];
    return rankingAt(inspected)
      .map((p, i) => ({ player: p, rank: i + 1, elo: ratingAt(p.id, inspected) }))
      .filter((r) => selected.has(r.player.id));
  }, [inspected, rankingAt, ratingAt, selected]);

  function togglePlayer(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const pillClass =
    "inline-flex items-center rounded-[5px] px-[7px] py-1 text-[10px] font-bold bg-secondary text-muted-foreground";

  return (
    <section className="rounded-2xl border border-border bg-card p-[19px] min-[761px]:p-[25px]">
      <div className="mb-5 flex items-center justify-between gap-3.5 max-[760px]:mb-4">
        <div>
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
            {compact ? "全员 ELO 趋势" : "积分与排名"}
          </h2>
          <p className="mt-[5px] text-[10px] text-muted-foreground">
            {sortedPlayers.length} 位成员 · {selected.size} 位已选
          </p>
        </div>
        {compact ? (
          <Link
            href="/trends"
            className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-win"
          >
            展开大图
            <ArrowRight className="size-[15px]" strokeWidth={1.65} />
          </Link>
        ) : (
          <span className={pillClass}>初始 ELO {INITIAL_RATING}</span>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2.5 pb-6 max-[760px]:pb-4">
        <Segmented
          label="全员趋势类型"
          options={MODES}
          value={mode}
          onChange={setMode}
        />
        <Segmented
          label="全员趋势周期"
          options={RANGES}
          value={range}
          onChange={(key) => {
            setRange(key);
            setInspectedDate(null);
          }}
        />
      </div>

      <div className="grid gap-[22px] min-[761px]:grid-cols-[minmax(0,1fr)_165px] max-[760px]:gap-4">
        <div className="min-w-0 self-center">
          {allDates.length === 0 ? (
            <div className="grid min-h-[265px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
              还没有比赛数据，记一场后这里会出现趋势。
            </div>
          ) : selected.size === 0 ? (
            <div className="grid min-h-[265px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
              选择下方成员，查看 ELO 趋势。
            </div>
          ) : (
            <>
              <div className="h-[250px] min-[761px]:h-[290px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={chartData}
                    margin={{ top: 8, right: 8, bottom: 4, left: 0 }}
                    onMouseMove={(state) => {
                      const label = (state as { activeLabel?: string | number })
                        ?.activeLabel;
                      if (typeof label === "string") setInspectedDate(label);
                    }}
                    onMouseLeave={() => setInspectedDate(null)}
                  >
                    <CartesianGrid
                      strokeDasharray="3 5"
                      stroke="var(--border)"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="date"
                      tickFormatter={shortDate}
                      tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                      tickMargin={6}
                      minTickGap={40}
                      tickLine={false}
                      axisLine={false}
                    />
                    {mode === "elo" ? (
                      <YAxis
                        domain={["dataMin - 15", "dataMax + 15"]}
                        tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                        width={36}
                        tickLine={false}
                        axisLine={false}
                      />
                    ) : (
                      <YAxis
                        reversed
                        domain={[1, sortedPlayers.length]}
                        allowDecimals={false}
                        tickCount={Math.min(6, sortedPlayers.length)}
                        tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                        width={24}
                        tickLine={false}
                        axisLine={false}
                      />
                    )}
                    <Tooltip
                      content={() => null}
                      cursor={{
                        stroke: "var(--muted-foreground)",
                        strokeDasharray: "3 4",
                        strokeOpacity: 0.55,
                      }}
                    />
                    {selectedPlayers.map((p) => (
                      <Line
                        key={p.id}
                        type="monotone"
                        dataKey={String(p.id)}
                        name={p.name}
                        stroke={seriesVar(colorIndexOf.get(p.id) ?? 0)}
                        strokeWidth={selectedPlayers.length === 1 ? 3 : 2}
                        dot={false}
                        activeDot={{ r: 3 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {/* 键盘 / 触碰兜底：日期按钮行（hover 图区也可定位） */}
              <div
                className="mt-1 flex gap-0.5 overflow-x-auto pb-1"
                role="group"
                aria-label="选择查看日期"
              >
                {windowDates.map((d) => (
                  <button
                    key={d}
                    type="button"
                    onClick={() => setInspectedDate(d)}
                    onFocus={() => setInspectedDate(d)}
                    aria-label={`查看 ${d} 全员积分`}
                    className={cn(
                      "shrink-0 rounded px-1.5 py-0.5 font-num text-[10px] transition-colors",
                      d === inspected
                        ? "bg-secondary font-bold text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {shortDate(d)}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>

        <aside
          aria-live="polite"
          className="min-[761px]:border-l min-[761px]:border-border min-[761px]:pl-5 max-[760px]:border-t max-[760px]:border-border max-[760px]:pt-3"
        >
          <div className="mb-2 flex items-center justify-between gap-2 text-[9px] text-muted-foreground">
            <span>{inspected ?? "—"}</span>
            <span>{mode === "rank" ? "名次 / ELO" : "ELO"}</span>
          </div>
          {selected.size === 0 ? (
            <p className="py-3 text-xs text-muted-foreground">尚未选择成员</p>
          ) : (
            <div className="max-[760px]:grid max-[760px]:grid-cols-2 max-[760px]:gap-x-5">
              {readoutRows.map(({ player, rank, elo }) => (
                <Link
                  key={player.id}
                  href={`/players/${player.id}`}
                  className="flex items-center gap-[7px] py-[7px] text-[11px] transition-colors hover:text-win"
                >
                  <span className="min-w-[13px] font-num text-[10px] text-muted-foreground">
                    {String(rank).padStart(2, "0")}
                  </span>
                  <span
                    className="size-[7px] shrink-0 rounded-full"
                    style={{
                      background: seriesVar(colorIndexOf.get(player.id) ?? 0),
                    }}
                  />
                  <span className="truncate">
                    {player.name}
                    {player.id === myId && (
                      <small className="ml-1 text-[9px] text-muted-foreground">
                        我
                      </small>
                    )}
                  </span>
                  <strong className="ml-auto font-num text-[17px] font-medium">
                    {elo}
                  </strong>
                </Link>
              ))}
            </div>
          )}
        </aside>
      </div>

      <div className="mt-[17px] flex justify-between gap-3 text-[10px] text-muted-foreground max-[760px]:mt-4 max-[760px]:text-[9px]">
        <span>点击日期查看当日排名与积分</span>
        <span className="max-[760px]:hidden">颜色与球员固定对应</span>
      </div>

      <div className="mt-[19px] border-t border-border pt-[13px] max-[760px]:mt-4 max-[760px]:pt-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-[11px] font-medium text-muted-foreground">
            选择成员
          </h3>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() =>
                setSelected(new Set(sortedPlayers.map((p) => p.id)))
              }
              className="text-[10px] text-muted-foreground transition-colors hover:text-win"
            >
              全选
            </button>
            <button
              type="button"
              disabled={myId === null}
              onClick={() => myId !== null && setSelected(new Set([myId]))}
              className="text-[10px] text-muted-foreground transition-colors hover:text-win disabled:opacity-40"
            >
              只看自己
            </button>
            <button
              type="button"
              onClick={() => setSelected(new Set())}
              className="text-[10px] text-muted-foreground transition-colors hover:text-win"
            >
              清空
            </button>
          </div>
        </div>
        <div className="mt-2.5 flex flex-wrap gap-2 max-[760px]:grid max-[760px]:grid-cols-3 max-[760px]:gap-[7px]">
          {sortedPlayers.map((p) => {
            const isSelected = selected.has(p.id);
            return (
              <button
                key={p.id}
                type="button"
                aria-pressed={isSelected}
                onClick={() => togglePlayer(p.id)}
                className={cn(
                  "inline-flex min-h-9 items-center gap-2 rounded-[7px] border border-border px-3 py-[7px] text-[11px] transition-colors max-[760px]:min-h-[38px] max-[760px]:gap-1.5 max-[760px]:px-2 max-[760px]:text-[10px]",
                  isSelected
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground"
                )}
              >
                <span
                  className="size-[7px] shrink-0 rounded-full max-[760px]:size-1.5"
                  style={{
                    background: isSelected
                      ? seriesVar(colorIndexOf.get(p.id) ?? 0)
                      : "var(--muted-foreground)",
                    opacity: isSelected ? 1 : 0.4,
                  }}
                />
                <span className="truncate">
                  {p.name}
                  {p.id === myId && (
                    <small className="ml-1 text-[9px] max-[760px]:text-[8px]">
                      我
                    </small>
                  )}
                </span>
                <Check
                  className={cn(
                    "size-[13px] shrink-0 max-[760px]:ml-auto max-[760px]:size-[11px]",
                    isSelected ? "opacity-100" : "opacity-0"
                  )}
                  strokeWidth={2}
                />
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}
