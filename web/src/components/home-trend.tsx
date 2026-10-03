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
import type { LocalDate } from "@/lib/ratings/types";
import type { RatingView } from "@/lib/ratings/view-types";
import {
  appendCurrentEstimateRow,
  buildTrendRows,
  currentSegmentOverlay,
  displayRankRows,
  displayRatingRows,
  eventLocalDate,
  formatTrendSeasonLabel,
  listTrendSeasons,
} from "@/lib/ratings/chart-data";
import { cn } from "@/lib/utils";

interface PlayerLite {
  id: number;
  name: string;
}

/**
 * Legacy：history/players 旧口径渲染逐比特不动。
 * glicko2：消费服务端投影 view（points + weekSegments），正式周节点实线、
 * 当前区段预估虚线、季重置/周校准独立 tooltip；同日不同事件不折叠。
 */
export type HomeTrendProps =
  | {
      model: "legacy";
      history: EloHistoryPoint[];
      /** 全部球员（含尚无比赛者）；chips 与排名按 id 升序固定颜色 */
      players: PlayerLite[];
      /** compact = 首页嵌入（带「展开大图」链接）；full = /trends 大图页 */
      variant?: "compact" | "full";
      /** 档案/大图链接保留评分模式，如 "?rating=glicko2"；Legacy 不需要。 */
      ratingQuery?: string;
    }
  | {
      model: "glicko2";
      view: RatingView;
      currentSegmentId: string;
      currentSeason: LocalDate | null;
      /** 服务端注入的当前时点（result.asOf；stale 时如实为旧成功时点）。 */
      now: string;
      variant?: "compact" | "full";
      ratingQuery?: string;
    };

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

/** 新版模式选项：积分口径沿用 elo key，展示文案换「评分」。 */
const Glicko2_MODES: { key: Mode; label: string }[] = [
  { key: "elo", label: "评分" },
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

export function HomeTrend(props: HomeTrendProps) {
  if (props.model === "glicko2") return <Glicko2Trend {...props} />;
  return <LegacyTrend {...props} />;
}

// ---------------------------------------------------------------------------
// Legacy：旧口径渲染路径保持不动。
// ---------------------------------------------------------------------------

function LegacyTrend({
  history,
  players,
  variant = "full",
  ratingQuery = "",
}: {
  history: EloHistoryPoint[];
  players: PlayerLite[];
  variant?: "compact" | "full";
  ratingQuery?: string;
}) {
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
            href={`/trends${ratingQuery}`}
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
                  href={`/players/${player.id}${ratingQuery}`}
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

// ---------------------------------------------------------------------------
// glicko2：正式周节点实线 + 当前区段预估虚线 + 季重置/周校准独立 tooltip。
// ---------------------------------------------------------------------------

type SeasonKey = "current" | "all" | LocalDate;

/** 校准/重置/逐场预估事件的独立 tooltip：只在这些事件上渲染，不随线变色。 */
function TrendEventTooltip({
  active,
  rowKey,
  rowsByKey,
  selectedIds,
  nameOf,
}: {
  active?: boolean;
  rowKey?: string | number;
  rowsByKey: ReadonlyMap<string, { kind: string; correction: Record<number, number>; resetDeltas: Record<number, number>; matchDeltas: Record<number, number> }>;
  selectedIds: readonly number[];
  nameOf: (id: number) => string;
}) {
  const row = rowKey !== undefined ? rowsByKey.get(String(rowKey)) : undefined;
  if (!active || !row) return null;

  const deltaClass = (delta: number) =>
    cn(
      "font-num",
      delta > 0 ? "text-win" : delta < 0 ? "text-loss" : "text-muted-foreground"
    );
  const formatDelta = (delta: number) => (delta > 0 ? `+${delta}` : `${delta}`);

  if (row.kind === "match_estimated") {
    // 逐场预估：只列该场参赛且被勾选的球员；同日多场各自独立点不折叠。
    // 合成「现在」平接行的 matchDeltas 为空，自然不弹。
    const entries = selectedIds
      .map((id) => [id, row.matchDeltas[id]] as const)
      .filter(([, delta]) => delta !== undefined);
    if (entries.length === 0) return null;
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 text-[11px] shadow-md">
        <div className="mb-1 font-bold text-card-foreground">单场预估 · 变化</div>
        {entries.map(([id, delta]) => (
          <div key={id} className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">{nameOf(id)}</span>
            <span className={deltaClass(delta)}>{formatDelta(delta)}</span>
          </div>
        ))}
      </div>
    );
  }
  if (row.kind === "weekly_final") {
    const entries = selectedIds
      .map((id) => [id, row.correction[id]] as const)
      .filter(([, delta]) => delta !== undefined);
    if (entries.length === 0) return null;
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 text-[11px] shadow-md">
        <div className="mb-1 font-bold text-card-foreground">周正式结算 · 校准</div>
        {entries.map(([id, delta]) => (
          <div key={id} className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">{nameOf(id)}</span>
            <span className={deltaClass(delta)}>{formatDelta(delta)}</span>
          </div>
        ))}
      </div>
    );
  }
  if (row.kind === "season_reset") {
    const entries = selectedIds
      .map((id) => [id, row.resetDeltas[id]] as const)
      .filter(([, delta]) => delta !== undefined);
    if (entries.length === 0) return null;
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 text-[11px] shadow-md">
        <div className="mb-1 font-bold text-card-foreground">赛季重置 · 软回中</div>
        {entries.map(([id, delta]) => (
          <div key={id} className="flex items-center justify-between gap-4">
            <span className="text-muted-foreground">{nameOf(id)}</span>
            <span className={deltaClass(delta)}>{formatDelta(delta)}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}

function Glicko2Trend({
  view,
  currentSegmentId,
  currentSeason,
  now,
  variant = "full",
  ratingQuery = "",
}: {
  view: RatingView;
  currentSegmentId: string;
  currentSeason: LocalDate | null;
  now: string;
  variant?: "compact" | "full";
  ratingQuery?: string;
}) {
  const compact = variant === "compact";
  const [mode, setMode] = React.useState<Mode>("elo");
  const [seasonKey, setSeasonKey] = React.useState<SeasonKey>("current");
  const [inspectedKey, setInspectedKey] = React.useState<string | null>(null);
  const [myId, setMyId] = React.useState<number | null>(null);

  const sortedPlayers = React.useMemo(
    () =>
      view.players
        .map((p) => ({ id: p.playerId, name: p.name }))
        .sort((a, b) => a.id - b.id),
    [view.players]
  );
  const colorIndexOf = React.useMemo(
    () => new Map(sortedPlayers.map((p, i) => [p.id, i])),
    [sortedPlayers]
  );
  const nameOf = React.useCallback(
    (id: number) => sortedPlayers.find((p) => p.id === id)?.name ?? String(id),
    [sortedPlayers]
  );

  const [selected, setSelected] = React.useState<ReadonlySet<number>>(
    () => new Set(sortedPlayers.map((p) => p.id))
  );

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  // 季度选择：当前季度 / 各历史季度 / 全部历史（按 points[].season 过滤）
  const seasonOptions = React.useMemo(
    () => [
      { key: "current" as SeasonKey, label: "当前季度" },
      ...listTrendSeasons(view.points).map((season) => ({
        key: season as SeasonKey,
        label: formatTrendSeasonLabel(season),
      })),
      { key: "all" as SeasonKey, label: "全部历史" },
    ],
    [view.points]
  );

  const seasonFilter =
    seasonKey === "current" ? currentSeason : seasonKey === "all" ? undefined : seasonKey;

  const rows = React.useMemo(() => {
    const base = buildTrendRows(view.points, view.weekSegments, {
      season: seasonFilter,
    });
    // 虚线总是平接到今天：当前季度与全部历史视图在末尾追加合成「现在」行
    // （值为与排行榜同口径的当前展示分；历史具体季度不追加）。
    if (seasonKey !== "current" && seasonKey !== "all") return base;
    const values: Record<number, number> = {};
    for (const p of view.players) {
      if (p.displayRating !== null) values[p.playerId] = p.displayRating;
    }
    return appendCurrentEstimateRow(base, { values, currentSegmentId, now });
  }, [view.points, view.weekSegments, view.players, seasonFilter, seasonKey, currentSegmentId, now]);

  const ratingRows = React.useMemo(() => displayRatingRows(rows), [rows]);
  const rankRows = React.useMemo(() => displayRankRows(rows), [rows]);
  const displayRows = mode === "rank" ? rankRows : ratingRows;

  // 当前区段（未结算）：实线停笔在最后一个正式节点，预估虚线从锚点行接续
  const { estimatedKeys, anchorKey } = React.useMemo(
    () => currentSegmentOverlay(rows, currentSegmentId),
    [rows, currentSegmentId]
  );

  // 当前区段内有预估点的球员（锚点行只为这些球员补虚线起点）
  const estimatedPlayerIds = React.useMemo(() => {
    const ids = new Set<number>();
    for (const row of rows) {
      if (estimatedKeys.has(row.key)) {
        for (const playerId of Object.keys(row.r)) ids.add(Number(playerId));
      }
    }
    return ids;
  }, [rows, estimatedKeys]);

  const rowByKey = React.useMemo(
    () => new Map(rows.map((row) => [row.key, row])),
    [rows]
  );

  // Recharts 行：key = 事件 ID（同刻两事件不合并）；预估行只留 `:est`
  // 虚线值（实线 undefined 即停笔）；锚点行带 `:est` 起点让虚线不断线。
  const chartData = React.useMemo(
    () =>
      displayRows.map((row) => {
        const estimatedRow = estimatedKeys.has(row.key);
        const obj: Record<string, number | string> = {
          key: row.key,
          kind: row.kind,
          label: shortDate(eventLocalDate(row.at)),
        };
        if (!estimatedRow) {
          for (const [playerId, value] of Object.entries(row.values)) {
            obj[playerId] = value;
          }
        }
        if (estimatedRow) {
          for (const playerId of Object.keys(row.r)) {
            const value = row.values[Number(playerId)];
            if (value !== undefined) obj[`${playerId}:est`] = value;
          }
        } else if (row.key === anchorKey) {
          for (const playerId of estimatedPlayerIds) {
            const value = row.values[playerId];
            if (value !== undefined) obj[`${playerId}:est`] = value;
          }
        }
        return obj;
      }),
    [displayRows, estimatedKeys, anchorKey, estimatedPlayerIds]
  );

  const maxRank = React.useMemo(
    () =>
      Math.max(
        1,
        ...rankRows.flatMap((row) => Object.values(row.values))
      ),
    [rankRows]
  );

  const inspected =
    inspectedKey && rowByKey.has(inspectedKey)
      ? inspectedKey
      : rows[rows.length - 1]?.key;
  const inspectedRow = inspected ? rowByKey.get(inspected) : undefined;

  const selectedPlayers = React.useMemo(
    () => sortedPlayers.filter((p) => selected.has(p.id)),
    [sortedPlayers, selected]
  );

  // 读数栏：该事件时点的并列名次 + 展示分（缺席者沿用最近分值，与线一致）
  const readoutRows = React.useMemo(() => {
    if (!inspectedRow) return [];
    const ratingRow = ratingRows.find((row) => row.key === inspectedRow.key);
    const rankRow = rankRows.find((row) => row.key === inspectedRow.key);
    return selectedPlayers
      .map((player) => ({
        player,
        rank: rankRow?.values[player.id],
        rating: ratingRow?.values[player.id],
      }))
      .filter((row) => row.rating !== undefined)
      .sort(
        (a, b) =>
          (a.rank ?? Number.POSITIVE_INFINITY) -
          (b.rank ?? Number.POSITIVE_INFINITY)
      );
  }, [inspectedRow, ratingRows, rankRows, selectedPlayers]);

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
  const historical = seasonKey !== "current";

  return (
    <section className="rounded-2xl border border-border bg-card p-[19px] min-[761px]:p-[25px]">
      <div className="mb-5 flex items-center justify-between gap-3.5 max-[760px]:mb-4">
        <div>
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
            {compact ? "全员评分趋势" : "积分与排名"}
          </h2>
          <p className="mt-[5px] text-[10px] text-muted-foreground">
            {sortedPlayers.length} 位成员 · {selected.size} 位已选
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {historical && <span className={pillClass}>历史回放</span>}
          {compact ? (
            <Link
              href={`/trends${ratingQuery}`}
              className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-win"
            >
              展开大图
              <ArrowRight className="size-[15px]" strokeWidth={1.65} />
            </Link>
          ) : null}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2.5 pb-6 max-[760px]:pb-4">
        <Segmented
          label="全员趋势类型"
          options={Glicko2_MODES}
          value={mode}
          onChange={setMode}
        />
        <Segmented
          label="全员趋势季度"
          options={seasonOptions}
          value={seasonKey}
          onChange={(key) => {
            setSeasonKey(key);
            setInspectedKey(null);
          }}
        />
      </div>

      <div className="grid gap-[22px] min-[761px]:grid-cols-[minmax(0,1fr)_165px] max-[760px]:gap-4">
        <div className="min-w-0 self-center">
          {rows.length === 0 ? (
            <div className="grid min-h-[265px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
              该季度还没有比赛数据，记一场后这里会出现趋势。
            </div>
          ) : selected.size === 0 ? (
            <div className="grid min-h-[265px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
              选择下方成员，查看评分趋势。
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
                      if (typeof label === "string") setInspectedKey(label);
                    }}
                    onMouseLeave={() => setInspectedKey(null)}
                  >
                    <CartesianGrid
                      strokeDasharray="3 5"
                      stroke="var(--border)"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="key"
                      tickFormatter={(key) =>
                        rowByKey.get(String(key))
                          ? shortDate(eventLocalDate(rowByKey.get(String(key))!.at))
                          : ""
                      }
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
                        domain={[1, maxRank]}
                        allowDecimals={false}
                        tickCount={Math.min(6, maxRank)}
                        tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                        width={24}
                        tickLine={false}
                        axisLine={false}
                      />
                    )}
                    <Tooltip
                      content={(tooltipProps) => (
                        <TrendEventTooltip
                          active={tooltipProps.active}
                          rowKey={tooltipProps.label}
                          rowsByKey={rowByKey}
                          selectedIds={selectedPlayers.map((p) => p.id)}
                          nameOf={nameOf}
                        />
                      )}
                      cursor={{
                        stroke: "var(--muted-foreground)",
                        strokeDasharray: "3 4",
                        strokeOpacity: 0.55,
                      }}
                    />
                    {selectedPlayers.map((p) => (
                      <React.Fragment key={p.id}>
                        {/* 入场动画缩短至 ~400ms：默认 1500ms 在多人曲线上像「计算中」，
                            且 ResponsiveContainer 重播会造成反复。 */}
                        <Line
                          type="monotone"
                          dataKey={String(p.id)}
                          name={p.name}
                          stroke={seriesVar(colorIndexOf.get(p.id) ?? 0)}
                          strokeWidth={selectedPlayers.length === 1 ? 3 : 2}
                          dot={false}
                          activeDot={{ r: 3 }}
                          animationDuration={400}
                        />
                        <Line
                          type="monotone"
                          dataKey={`${p.id}:est`}
                          name={`${p.name}（本周预估）`}
                          stroke={seriesVar(colorIndexOf.get(p.id) ?? 0)}
                          strokeWidth={selectedPlayers.length === 1 ? 3 : 2}
                          strokeDasharray="5 4"
                          dot={false}
                          activeDot={{ r: 3 }}
                          animationDuration={400}
                        />
                      </React.Fragment>
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {/* 键盘 / 触碰兜底：事件时点按钮行 */}
              <div
                className="mt-1 flex gap-0.5 overflow-x-auto pb-1"
                role="group"
                aria-label="选择查看日期"
              >
                {rows.map((row) => (
                  <button
                    key={row.key}
                    type="button"
                    onClick={() => setInspectedKey(row.key)}
                    onFocus={() => setInspectedKey(row.key)}
                    aria-label={`查看 ${eventLocalDate(row.at)} 全员评分`}
                    className={cn(
                      "shrink-0 rounded px-1.5 py-0.5 font-num text-[10px] transition-colors",
                      row.key === inspected
                        ? "bg-secondary font-bold text-foreground"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {shortDate(eventLocalDate(row.at))}
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
            <span>{inspectedRow ? shortDate(eventLocalDate(inspectedRow.at)) : "—"}</span>
            <span>
              {mode === "rank" ? "名次 / 评分" : "评分"}
              {inspectedRow?.kind === "match_estimated" ? " · 预估" : ""}
              {inspectedRow?.kind === "season_reset" ? " · 重置" : ""}
            </span>
          </div>
          {selected.size === 0 ? (
            <p className="py-3 text-xs text-muted-foreground">尚未选择成员</p>
          ) : (
            <div className="max-[760px]:grid max-[760px]:grid-cols-2 max-[760px]:gap-x-5">
              {readoutRows.map(({ player, rank, rating }) => {
                // 逐场预估行：该场参赛者的分值旁同步该场 +N/-N（手机读数可见）。
                const matchDelta =
                  inspectedRow?.kind === "match_estimated"
                    ? inspectedRow.matchDeltas[player.id]
                    : undefined;
                return (
                  <Link
                    key={player.id}
                    href={`/players/${player.id}${ratingQuery}`}
                    className="flex items-center gap-[7px] py-[7px] text-[11px] transition-colors hover:text-win"
                  >
                    <span className="min-w-[13px] font-num text-[10px] text-muted-foreground">
                      {rank !== undefined ? String(rank).padStart(2, "0") : "—"}
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
                      {rating}
                    </strong>
                    {matchDelta !== undefined && (
                      <small
                        className={cn(
                          "font-num text-[10px]",
                          matchDelta > 0
                            ? "text-win"
                            : matchDelta < 0
                              ? "text-loss"
                              : "text-muted-foreground"
                        )}
                      >
                        {matchDelta > 0 ? `+${matchDelta}` : matchDelta}
                      </small>
                    )}
                  </Link>
                );
              })}
            </div>
          )}
        </aside>
      </div>

      <div className="mt-[17px] flex justify-between gap-3 text-[10px] text-muted-foreground max-[760px]:mt-4 max-[760px]:text-[9px]">
        <span>点击日期查看该时点的排名与评分</span>
        <span className="inline-flex items-center gap-2.5">
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-4 border-t-2 border-current" />
            正式结算
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block w-4 border-t-2 border-dashed border-current" />
            本周预估
          </span>
        </span>
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
