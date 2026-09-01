"use client";

import * as React from "react";
import { PlayerAvatar } from "@/components/player-avatar";
import type { HeadToHeadRecord } from "@/lib/stats";

interface H2hTableProps {
  records: HeadToHeadRecord[];
}

export function H2hTable({ records }: H2hTableProps) {
  if (records.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
        还没有对战记录
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-border bg-card shadow-sm">
      <div className="grid grid-cols-12 gap-2 border-b border-border px-4 py-3 text-xs font-medium text-muted-foreground">
        <div className="col-span-5">对手</div>
        <div className="col-span-3 text-center">战绩</div>
        <div className="col-span-2 text-center">胜率</div>
        <div className="col-span-2 text-right">场次</div>
      </div>
      <div className="divide-y divide-border">
        {records.map((r) => {
          const winRate = r.total > 0 ? Math.round((r.wins / r.total) * 100) : 0;
          return (
            <div
              key={r.opponentId}
              className="grid grid-cols-12 items-center gap-2 px-4 py-3"
            >
              <div className="col-span-5 flex items-center gap-2">
                <PlayerAvatar name={r.opponentName} size="xs" />
                <span className="truncate text-sm font-medium text-card-foreground">
                  {r.opponentName}
                </span>
              </div>
              <div className="col-span-3 text-center text-sm text-card-foreground">
                <span className="font-semibold text-emerald-600">{r.wins}</span>
                <span className="mx-1 text-muted-foreground">胜</span>
                <span className="font-semibold text-rose-600">{r.losses}</span>
                <span className="text-muted-foreground">负</span>
              </div>
              <div className="col-span-2 text-center text-sm tabular-nums text-card-foreground">
                {winRate}%
              </div>
              <div className="col-span-2 text-right text-sm tabular-nums text-muted-foreground">
                {r.total}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
