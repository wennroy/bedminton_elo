"use client";

import * as React from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { PlayerMatchRecord } from "@/lib/stats";
import type { RatingView } from "@/lib/ratings/view-types";
import {
  appendCurrentEstimateRow,
  buildTrendRows,
  currentSegmentOverlay,
  displayRatingRows,
  eventLocalDate,
  type DisplayTrendRow,
} from "@/lib/ratings/chart-data";
import { cn } from "@/lib/utils";

export interface TrendPoint {
  date: string;
  elo: number;
}

/**
 * Legacy：单 ELO 面积图逐比特不动。
 * glicko2：消费同一份 RatingView 投影的单人系列——正式周节点实线、
 * 当前区段逐场预估虚线、周校准与季重置独立 tooltip；历史单场始终标 Estimated。
 */
export type PlayerTrendProps =
  | {
      model: "legacy";
      playerName: string;
      /** 该球员全部 ELO 快照，按日升序 */
      points: TrendPoint[];
    }
  | {
      model: "glicko2";
      playerName: string;
      playerId: number;
      view: RatingView;
      currentSegmentId: string;
      /** 服务端注入的当前时点（result.asOf；stale 时如实为旧成功时点）。 */
      now: string;
    };

export function PlayerTrend(props: PlayerTrendProps) {
  if (props.model === "glicko2") return <Glicko2PlayerTrend {...props} />;
  return <LegacyPlayerTrend {...props} />;
}

type RangeKey = "4" | "12" | "all";

const RANGES: { key: RangeKey; label: string; weeks: number | null }[] = [
  { key: "4", label: "近 4 周", weeks: 4 },
  { key: "12", label: "近 12 周", weeks: 12 },
  { key: "all", label: "全部", weeks: null },
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

function shortDate(date: string): string {
  return date.slice(5).replace("-", ".");
}

/** 个人 ELO 趋势（mock profile chart）：周期只过滤图表，不改生涯统计 */
function LegacyPlayerTrend({
  playerName,
  points,
}: {
  playerName: string;
  /** 该球员全部 ELO 快照，按日升序 */
  points: TrendPoint[];
}) {
  const [range, setRange] = React.useState<RangeKey>("4");
  const [readout, setReadout] = React.useState<TrendPoint | null>(null);

  // cutoff 前最后一个 ELO 作区间起点
  const windowPoints = React.useMemo(() => {
    const cfg = RANGES.find((r) => r.key === range)!;
    if (cfg.weeks === null) return points;
    const cutoff = cutoffDate(cfg.weeks);
    const rest = points.filter((p) => p.date >= cutoff);
    const prev = points.filter((p) => p.date < cutoff).at(-1);
    return prev ? [{ date: cutoff, elo: prev.elo }, ...rest] : rest;
  }, [points, range]);

  const latest = windowPoints[windowPoints.length - 1];
  const diff = windowPoints.length > 0 ? latest.elo - windowPoints[0].elo : 0;
  const shown = readout ?? latest;

  return (
    <section className="min-w-0 rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
      <div className="mb-[22px] flex flex-wrap items-center justify-between gap-3 max-[760px]:mb-[18px]">
        <div>
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
            ELO 趋势
          </h2>
          <div className="mt-[3px] text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
            ELO PROGRESSION
          </div>
        </div>
        <div
          className="inline-flex gap-[2px] rounded-[7px] border border-border p-[3px]"
          role="group"
          aria-label="积分趋势周期"
        >
          {RANGES.map((r) => (
            <button
              key={r.key}
              type="button"
              aria-pressed={range === r.key}
              onClick={() => {
                setRange(r.key);
                setReadout(null);
              }}
              className={cn(
                "min-h-[30px] rounded px-2.5 text-[10px] whitespace-nowrap transition-colors max-[760px]:min-h-9 max-[760px]:px-2",
                range === r.key
                  ? "bg-secondary font-bold text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {points.length === 0 ? (
        <div className="grid min-h-[215px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
          还没有比赛数据，记一场后这里会出现趋势。
        </div>
      ) : (
        <>
          <div className="mb-3 flex items-center justify-between gap-2.5 text-[10px] text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="h-[3px] w-4 rounded-[3px] bg-chart" />
              {playerName}
            </span>
            <span aria-live="polite">
              {shown ? `${shortDate(shown.date)} · ELO ${shown.elo}` : "—"}
            </span>
          </div>
          <div className="h-[215px] min-[1600px]:h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={windowPoints}
                margin={{ top: 8, right: 8, bottom: 4, left: 0 }}
                onMouseMove={(state) => {
                  const p = (
                    state as {
                      activePayload?: { payload?: TrendPoint }[];
                    }
                  )?.activePayload?.[0]?.payload;
                  if (p) setReadout(p);
                }}
                onMouseLeave={() => setReadout(null)}
              >
                <defs>
                  <linearGradient
                    id="player-trend-fill"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="var(--chart-fill)"
                      stopOpacity={0.24}
                    />
                    <stop
                      offset="100%"
                      stopColor="var(--chart-fill)"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
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
                <YAxis
                  domain={["dataMin - 10", "dataMax + 10"]}
                  tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                  width={36}
                  tickLine={false}
                  axisLine={false}
                />
                <Area
                  type="monotone"
                  dataKey="elo"
                  stroke="var(--chart)"
                  strokeWidth={2.5}
                  fill="url(#player-trend-fill)"
                  dot={{
                    r: 2.5,
                    fill: "var(--card)",
                    stroke: "var(--chart)",
                    strokeWidth: 2,
                  }}
                  activeDot={{ r: 4 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-[9px] flex justify-between gap-2.5 border-t border-border pt-4 text-[10px] text-muted-foreground max-[760px]:text-[9px]">
            <span>
              区间变化{" "}
              <strong
                className={cn(
                  "font-semibold",
                  diff > 0
                    ? "text-win"
                    : diff < 0
                      ? "text-loss"
                      : "text-foreground"
                )}
              >
                {diff > 0 ? "+" : ""}
                {diff} ELO
              </strong>
            </span>
            <span>按比赛日汇总 · 触碰数据点查看</span>
          </div>
        </>
      )}
    </section>
  );
}

/** 近期手感（mock form-panel）：近 8 场胜负点 + 近 8 场胜率 / 生涯场均净胜分 */
export function RecentForm({
  matches,
  avgPointDiff,
  peakElo,
  currentElo,
  peakLabel = "生涯最高积分",
}: {
  /** 全部比赛记录，最新在前 */
  matches: PlayerMatchRecord[];
  avgPointDiff: number;
  peakElo: number;
  currentElo: number;
  /** 峰值口径名称：legacy 为「生涯最高积分」，glicko2 为「生涯最高评分」 */
  peakLabel?: string;
}) {
  // 左早右晚
  const recent = matches.slice(0, 8).reverse();
  const recentWins = recent.filter((m) => m.won).length;
  const gap = Math.max(0, peakElo - currentElo);

  return (
    <section className="rounded-2xl border border-border bg-card p-5 min-[761px]:p-[23px]">
      <div className="flex items-center justify-between gap-3.5 max-[760px]:mb-0">
        <div>
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
            近期手感
          </h2>
          <div className="mt-[3px] text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
            RECENT FORM
          </div>
        </div>
        <span className="inline-flex items-center rounded-[5px] bg-secondary px-[7px] py-1 text-[10px] font-bold text-muted-foreground">
          近 8 场
        </span>
      </div>

      {recent.length === 0 ? (
        <div className="mt-4 grid min-h-[120px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
          还没有比赛记录。
        </div>
      ) : (
        <div
          className="mt-[18px] mb-5 flex gap-[5px] max-[760px]:my-[15px] max-[760px]:gap-1.5"
          aria-label="最近八场，从左到右由早到晚"
        >
          {recent.map((m) => (
            <span
              key={m.id}
              title={`${m.date} · ${m.scoreFor}:${m.scoreAgainst}`}
              className={cn(
                "grid h-[31px] w-[29px] shrink-0 place-items-center rounded-[5px] text-[11px] font-bold max-[760px]:h-7 max-[760px]:w-auto max-[760px]:max-w-[38px] max-[760px]:flex-1",
                m.won ? "bg-win-bg text-win" : "bg-loss-bg text-loss"
              )}
            >
              {m.won ? "胜" : "负"}
            </span>
          ))}
        </div>
      )}

      <div className="max-[760px]:grid max-[760px]:grid-cols-2 max-[760px]:gap-4">
        <div className="mt-3.5 flex items-center justify-between text-[11px] text-muted-foreground max-[760px]:mt-0">
          <span>近 8 场胜率</span>
          <span className="font-num text-xl text-foreground">
            {recent.length > 0 ? (
              <>
                {Math.round((recentWins / recent.length) * 100)}
                <small className="text-sm">%</small>
              </>
            ) : (
              "—"
            )}
          </span>
        </div>
        <div className="mt-3.5 flex items-center justify-between text-[11px] text-muted-foreground max-[760px]:mt-0">
          <span>生涯场均净胜分</span>
          <span
            className={cn(
              "font-num text-xl",
              avgPointDiff > 0
                ? "text-win"
                : avgPointDiff < 0
                  ? "text-loss"
                  : "text-foreground"
            )}
          >
            {matches.length > 0
              ? `${avgPointDiff > 0 ? "+" : ""}${avgPointDiff.toFixed(1)}`
              : "—"}
          </span>
        </div>
      </div>

      <div className="mt-[18px] border-t border-border pt-4 text-[11px] leading-[1.85] text-muted-foreground">
        {matches.length === 0 ? (
          "完成首场比赛后，这里会显示与生涯峰值的距离。"
        ) : gap > 0 ? (
          <>
            目前距离{peakLabel}{" "}
            <strong className="font-semibold text-win">{gap} 分</strong>。
          </>
        ) : (
          `目前正处于${peakLabel}。`
        )}
      </div>
    </section>
  );
}

/** 个人曲线 tooltip：周正式结算给校准、季重置给重置增量、逐场预估给该场变化。 */
function PlayerEventTooltip({
  active,
  row,
  playerId,
}: {
  active?: boolean;
  row?: DisplayTrendRow;
  playerId: number;
}) {
  if (!active || !row) return null;
  const deltaClass = (delta: number) =>
    cn(
      "font-num",
      delta > 0 ? "text-win" : delta < 0 ? "text-loss" : "text-muted-foreground"
    );
  const formatDelta = (delta: number) =>
    delta > 0 ? `+${delta}` : `${delta}`;

  if (row.kind === "match_estimated") {
    // 逐场预估：该场本人的单场变化；合成「现在」平接行的 matchDeltas
    // 为空，自然不弹。
    const delta = row.matchDeltas[playerId];
    if (delta === undefined) return null;
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 text-[11px] shadow-md">
        <div className="mb-1 font-bold text-card-foreground">
          单场预估 · 变化
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">该场变化</span>
          <span className={deltaClass(delta)}>{formatDelta(delta)}</span>
        </div>
      </div>
    );
  }
  if (row.kind === "weekly_final") {
    const correction = row.correction[playerId];
    if (correction === undefined) return null;
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 text-[11px] shadow-md">
        <div className="mb-1 font-bold text-card-foreground">
          周正式结算 · 校准
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">评分校准</span>
          <span className={deltaClass(correction)}>{formatDelta(correction)}</span>
        </div>
      </div>
    );
  }
  if (row.kind === "season_reset") {
    const delta = row.resetDeltas[playerId];
    if (delta === undefined) return null;
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 text-[11px] shadow-md">
        <div className="mb-1 font-bold text-card-foreground">
          赛季重置 · 软回中
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-muted-foreground">重置变化</span>
          <span className={deltaClass(delta)}>{formatDelta(delta)}</span>
        </div>
      </div>
    );
  }
  return null;
}

/**
 * 个人评分曲线（glicko2）：与全员大图同一视觉语义——正式实线、
 * 当前区段逐场预估虚线、校准/重置独立 tooltip；缺席周沿用最近值不断线。
 */
function Glicko2PlayerTrend({
  playerName,
  playerId,
  view,
  currentSegmentId,
  now,
}: {
  playerName: string;
  playerId: number;
  view: RatingView;
  currentSegmentId: string;
  now: string;
}) {
  const [range, setRange] = React.useState<RangeKey>("4");
  const [readoutKey, setReadoutKey] = React.useState<string | null>(null);

  const rows = React.useMemo(() => {
    // 虚线总是平接到今天：追加合成「现在」行（值仅含本球员的当前展示分，
    // 自己本周没打也从最后事件延伸到今天；未评级者不追加）。
    const base = buildTrendRows(view.points, view.weekSegments);
    const displayRating = view.players.find(
      (p) => p.playerId === playerId
    )?.displayRating;
    const withNow =
      displayRating === null || displayRating === undefined
        ? base
        : appendCurrentEstimateRow(base, {
            values: { [playerId]: displayRating },
            currentSegmentId,
            now,
          });
    return displayRatingRows(withNow).filter(
      // 只保留该球员自身有事件的行（逐场参赛/周 Final/季重置/合成现在行）；
      // 缺席间隔由连线沿用，不为别人的比赛造平点。
      (row) => row.r[playerId] !== undefined
    );
  }, [view.points, view.weekSegments, view.players, playerId, currentSegmentId, now]);
  const rowByKey = React.useMemo(
    () => new Map(rows.map((row) => [row.key, row])),
    [rows]
  );

  // 周期只过滤图表，不改生涯统计；cutoff 前最后一个事件值已由沿用规则带入窗口。
  const windowRows = React.useMemo(() => {
    const cfg = RANGES.find((r) => r.key === range)!;
    if (cfg.weeks === null) return rows;
    const cutoff = cutoffDate(cfg.weeks);
    return rows.filter((row) => eventLocalDate(row.at) >= cutoff);
  }, [rows, range]);

  // 当前区段（未结算）：实线停笔在最后一个正式节点，虚线从锚点行接续。
  // 在窗口行上计算：锚点被周期过滤裁掉时虚线直接从首个预估点开始。
  const { estimatedKeys, anchorKey } = React.useMemo(
    () => currentSegmentOverlay(windowRows, currentSegmentId),
    [windowRows, currentSegmentId]
  );

  const chartData = React.useMemo(
    () =>
      windowRows.map((row) => {
        const value = row.values[playerId];
        const estimated = estimatedKeys.has(row.key);
        return {
          key: row.key,
          kind: row.kind,
          label: shortDate(eventLocalDate(row.at)),
          final: estimated ? undefined : value,
          ...(estimated || row.key === anchorKey ? { est: value } : {}),
        };
      }),
    [windowRows, playerId, estimatedKeys, anchorKey]
  );

  const latest = windowRows[windowRows.length - 1];
  const first = windowRows[0];
  const diff =
    latest !== undefined && first !== undefined
      ? (latest.values[playerId] ?? 0) - (first.values[playerId] ?? 0)
      : 0;
  const shown =
    (readoutKey !== null && rowByKey.get(readoutKey)) || latest || null;
  const shownValue = shown ? shown.values[playerId] : undefined;
  // 逐场预估行：读数行在分值后同步该场 +N/-N（手机读数可见）。
  const shownMatchDelta =
    shown?.kind === "match_estimated"
      ? shown.matchDeltas[playerId]
      : undefined;

  return (
    <section className="min-w-0 rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
      <div className="mb-[22px] flex flex-wrap items-center justify-between gap-3 max-[760px]:mb-[18px]">
        <div>
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
            评分趋势
          </h2>
          <div className="mt-[3px] text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
            RATING PROGRESSION
          </div>
        </div>
        <div
          className="inline-flex gap-[2px] rounded-[7px] border border-border p-[3px]"
          role="group"
          aria-label="评分趋势周期"
        >
          {RANGES.map((r) => (
            <button
              key={r.key}
              type="button"
              aria-pressed={range === r.key}
              onClick={() => {
                setRange(r.key);
                setReadoutKey(null);
              }}
              className={cn(
                "min-h-[30px] rounded px-2.5 text-[10px] whitespace-nowrap transition-colors max-[760px]:min-h-9 max-[760px]:px-2",
                range === r.key
                  ? "bg-secondary font-bold text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="grid min-h-[215px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
          还没有比赛数据，记一场后这里会出现趋势。
        </div>
      ) : (
        <>
          <div className="mb-3 flex items-center justify-between gap-2.5 text-[10px] text-muted-foreground">
            <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <span className="flex items-center gap-1.5">
                <span className="h-[3px] w-4 rounded-[3px] bg-chart" />
                {playerName}（正式）
              </span>
              <span className="flex items-center gap-1.5">
                <span className="h-0 w-4 border-t-2 border-dashed border-chart" />
                本周预估
              </span>
            </span>
            <span aria-live="polite">
              {shown && shownValue !== undefined ? (
                <>
                  {`${shortDate(eventLocalDate(shown.at))} · 评分 ${shownValue}`}
                  {shownMatchDelta !== undefined && (
                    <small
                      className={cn(
                        "font-num",
                        shownMatchDelta > 0
                          ? "text-win"
                          : shownMatchDelta < 0
                            ? "text-loss"
                            : "text-muted-foreground"
                      )}
                    >
                      {` ${shownMatchDelta > 0 ? "+" : ""}${shownMatchDelta}`}
                    </small>
                  )}
                  {shown.kind === "match_estimated"
                    ? shown.segment === currentSegmentId
                      ? " · 预估"
                      : " · 历史预估"
                    : shown.kind === "season_reset"
                      ? " · 重置"
                      : " · 正式"}
                </>
              ) : (
                "—"
              )}
            </span>
          </div>
          <div className="h-[215px] min-[1600px]:h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartData}
                margin={{ top: 8, right: 8, bottom: 4, left: 0 }}
                onMouseMove={(state) => {
                  const key = (state as { activeLabel?: string | number })
                    ?.activeLabel;
                  if (typeof key === "string") setReadoutKey(key);
                }}
                onMouseLeave={() => setReadoutKey(null)}
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
                <YAxis
                  domain={["dataMin - 10", "dataMax + 10"]}
                  tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                  width={36}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  content={(tooltipProps) => (
                    <PlayerEventTooltip
                      active={tooltipProps.active}
                      row={
                        tooltipProps.label !== undefined
                          ? rowByKey.get(String(tooltipProps.label))
                          : undefined
                      }
                      playerId={playerId}
                    />
                  )}
                  cursor={{
                    stroke: "var(--muted-foreground)",
                    strokeDasharray: "3 4",
                    strokeOpacity: 0.55,
                  }}
                />
                {/* 入场动画缩短至 ~400ms，与首页大图一致；默认 1500ms 在手机上像「计算中」。 */}
                <Line
                  type="monotone"
                  dataKey="final"
                  name={playerName}
                  stroke="var(--chart)"
                  strokeWidth={2.5}
                  dot={{
                    r: 2.5,
                    fill: "var(--card)",
                    stroke: "var(--chart)",
                    strokeWidth: 2,
                  }}
                  activeDot={{ r: 4 }}
                  animationDuration={400}
                />
                <Line
                  type="monotone"
                  dataKey="est"
                  name={`${playerName}（本周预估）`}
                  stroke="var(--chart)"
                  strokeWidth={2.5}
                  strokeDasharray="5 4"
                  dot={false}
                  activeDot={{ r: 4 }}
                  animationDuration={400}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-[9px] flex justify-between gap-2.5 border-t border-border pt-4 text-[10px] text-muted-foreground max-[760px]:text-[9px]">
            <span>
              区间变化{" "}
              <strong
                className={cn(
                  "font-semibold",
                  diff > 0
                    ? "text-win"
                    : diff < 0
                      ? "text-loss"
                      : "text-foreground"
                )}
              >
                {diff > 0 ? "+" : ""}
                {diff} 分
              </strong>
            </span>
            <span>正式为周结算 · 单场为预估 · 触碰数据点查看</span>
          </div>
        </>
      )}
    </section>
  );
}
