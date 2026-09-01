"use client";

import * as React from "react";
import type { PlayerMatchRecord } from "@/lib/stats";

const PAGE_SIZE = 10;

interface PlayerMatchHistoryProps {
  matches: PlayerMatchRecord[];
  playerName: string;
}

export function PlayerMatchHistory({
  matches,
  playerName,
}: PlayerMatchHistoryProps) {
  const [visibleCount, setVisibleCount] = React.useState(PAGE_SIZE);
  const sentinelRef = React.useRef<HTMLDivElement>(null);
  const hasMore = visibleCount < matches.length;

  // 滚动到底部哨兵时自动追加一页;IntersectionObserver 不可用时靠按钮兜底
  React.useEffect(() => {
    if (!hasMore) return;
    const sentinel = sentinelRef.current;
    if (!sentinel || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting) {
        setVisibleCount((c) => Math.min(c + PAGE_SIZE, matches.length));
      }
    });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [hasMore, matches.length]);

  if (matches.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
        还没有参赛记录
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {matches.slice(0, visibleCount).map((m) => (
        <MatchCard key={m.id} match={m} playerName={playerName} />
      ))}
      {hasMore ? (
        <div ref={sentinelRef} className="flex justify-center pt-1">
          <button
            onClick={() =>
              setVisibleCount((c) => Math.min(c + PAGE_SIZE, matches.length))
            }
            className="rounded-full bg-muted px-4 py-2 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
          >
            加载更多（还剩 {matches.length - visibleCount} 场）
          </button>
        </div>
      ) : (
        <p className="pt-1 text-center text-xs text-muted-foreground">
          已显示全部 {matches.length} 场
        </p>
      )}
    </div>
  );
}

function MatchCard({
  match,
  playerName,
}: {
  match: PlayerMatchRecord;
  playerName: string;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-3 shadow-sm">
      <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
        <span>{match.date}</span>
        <span className={match.won ? "text-emerald-600" : "text-rose-600"}>
          {match.won ? "胜" : "负"}
        </span>
      </div>
      <div className="flex items-center justify-between gap-2">
        <div className="flex flex-1 flex-col gap-1 text-sm">
          <div className="flex items-center gap-1 text-card-foreground">
            <span className="font-medium">
              {playerName} / {match.teammates.join(" / ")}
            </span>
          </div>
          <div className="text-muted-foreground">
            VS {match.opponents.join(" / ")}
          </div>
        </div>
        <div className="text-lg font-bold tabular-nums text-card-foreground">
          {match.scoreFor} : {match.scoreAgainst}
        </div>
      </div>
    </div>
  );
}
