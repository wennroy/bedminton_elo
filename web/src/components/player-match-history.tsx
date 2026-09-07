"use client";

import * as React from "react";
import type { PlayerMatchRecord } from "@/lib/stats";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 10;

type Filter = "all" | "wins" | "losses";

const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "全部" },
  { key: "wins", label: "获胜" },
  { key: "losses", label: "失利" },
];

interface PlayerMatchHistoryProps {
  /** 全部比赛记录，最新在前；delta 为服务端重放的单场 ELO 变化 */
  matches: PlayerMatchRecord[];
  playerName: string;
}

export function PlayerMatchHistory({
  matches,
  playerName,
}: PlayerMatchHistoryProps) {
  const [filter, setFilter] = React.useState<Filter>("all");
  const [visibleCount, setVisibleCount] = React.useState(PAGE_SIZE);
  const sentinelRef = React.useRef<HTMLDivElement>(null);

  const filtered = React.useMemo(
    () =>
      matches.filter((m) =>
        filter === "all" ? true : filter === "wins" ? m.won : !m.won
      ),
    [matches, filter]
  );
  const hasMore = visibleCount < filtered.length;

  // 滚动到底部哨兵时自动追加一页;IntersectionObserver 不可用时靠按钮兜底
  React.useEffect(() => {
    if (!hasMore) return;
    const sentinel = sentinelRef.current;
    if (!sentinel || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting) {
        setVisibleCount((c) => Math.min(c + PAGE_SIZE, filtered.length));
      }
    });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [hasMore, filtered.length]);

  return (
    <section className="overflow-hidden rounded-2xl border border-border bg-card">
      <div className="flex flex-wrap items-center justify-between gap-3 px-[18px] pt-[19px] min-[761px]:px-[25px] min-[761px]:pt-[22px]">
        <div className="pb-1">
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
            比赛记录
          </h2>
          <div className="mt-[3px] text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
            MATCH HISTORY
          </div>
        </div>
        <div
          className="inline-flex gap-[2px] rounded-[7px] border border-border p-[3px]"
          role="group"
          aria-label="比赛胜负筛选"
        >
          {FILTERS.map((f) => (
            <button
              key={f.key}
              type="button"
              aria-pressed={filter === f.key}
              onClick={() => {
                setFilter(f.key);
                setVisibleCount(PAGE_SIZE);
              }}
              className={cn(
                "min-h-[30px] rounded px-2.5 text-[10px] whitespace-nowrap transition-colors max-[760px]:min-h-9 max-[760px]:px-2",
                filter === f.key
                  ? "bg-secondary font-bold text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-[22px] max-[760px]:mt-[18px]">
        {matches.length === 0 ? (
          <div className="border-t border-border py-[45px] text-center text-[13px] text-muted-foreground">
            还没有比赛记录，记一场后这里会列出战绩。
          </div>
        ) : filtered.length === 0 ? (
          <div className="border-t border-border py-[45px] text-center text-[13px] text-muted-foreground">
            这个筛选下还没有比赛记录。
          </div>
        ) : (
          <>
            {filtered.slice(0, visibleCount).map((m) => (
              <MatchRow key={m.id} match={m} playerName={playerName} />
            ))}
            <div className="border-t border-border p-2 text-center">
              {hasMore ? (
                <div ref={sentinelRef}>
                  <button
                    type="button"
                    onClick={() =>
                      setVisibleCount((c) =>
                        Math.min(c + PAGE_SIZE, filtered.length)
                      )
                    }
                    className="inline-flex min-h-9 items-center text-xs text-muted-foreground transition-colors hover:text-win"
                  >
                    加载更多 · 还剩 {filtered.length - visibleCount} 场
                  </button>
                </div>
              ) : (
                <span className="inline-flex min-h-9 items-center text-xs text-muted-foreground">
                  已展示全部 {filtered.length} 场比赛
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function MatchRow({
  match,
  playerName,
}: {
  match: PlayerMatchRecord;
  playerName: string;
}) {
  const delta = match.delta;
  return (
    <div className="relative grid grid-cols-[29px_1fr_63px_1fr] items-center gap-[9px] border-t border-border px-[18px] py-[14px] text-xs min-[761px]:grid-cols-[90px_42px_1fr_80px_1fr_60px] min-[761px]:gap-[15px] min-[761px]:px-[25px] min-[761px]:py-[15px]">
      {/* 手机：日期提到行顶，delta 放右上角 */}
      <span className="col-span-full text-[10px] text-muted-foreground min-[761px]:col-span-1">
        {match.date.replaceAll("-", ".")}
      </span>
      <span
        className={cn(
          "grid h-7 w-[26px] place-items-center rounded-[5px] text-[11px] font-bold min-[761px]:h-[31px] min-[761px]:w-[29px]",
          match.won ? "bg-win-bg text-win" : "bg-loss-bg text-loss"
        )}
      >
        {match.won ? "胜" : "负"}
      </span>
      <div className="min-w-0 text-[11px] text-card-foreground min-[761px]:text-xs">
        <span className="block truncate">
          {playerName}
          {match.teammates.length > 0 ? ` / ${match.teammates.join(" / ")}` : ""}
        </span>
        <small className="mt-0.5 block text-[8px] text-muted-foreground min-[761px]:text-[9px]">
          我方阵容
        </small>
      </div>
      <span className="text-center font-num text-[21px] text-card-foreground min-[761px]:text-[22px]">
        {match.scoreFor}
        <span className="px-[5px] text-xs text-muted-foreground">:</span>
        {match.scoreAgainst}
      </span>
      <div className="min-w-0 text-[11px] text-card-foreground min-[761px]:text-xs">
        <span className="block truncate">{match.opponents.join(" / ")}</span>
        <small className="mt-0.5 block text-[8px] text-muted-foreground min-[761px]:text-[9px]">
          对方阵容
        </small>
      </div>
      <span
        className={cn(
          "absolute top-3 right-[18px] font-num text-xs min-[761px]:static min-[761px]:text-right min-[761px]:text-[17px]",
          delta >= 0 ? "text-win" : "text-loss"
        )}
      >
        {delta > 0 ? `+${delta}` : delta}
      </span>
    </div>
  );
}
