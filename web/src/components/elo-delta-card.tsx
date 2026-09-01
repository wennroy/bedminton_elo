"use client";

import * as React from "react";
import { PlayerAvatar } from "@/components/player-avatar";
import { TrendingUp, TrendingDown } from "lucide-react";

export interface EloDeltaPlayer {
  id: number;
  name: string;
  before: number;
  after: number;
}

interface EloDeltaCardProps {
  players: EloDeltaPlayer[];
}

export function EloDeltaCard({ players }: EloDeltaCardProps) {
  return (
    <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <h3 className="mb-3 text-center text-base font-semibold text-card-foreground">
        ELO 变化
      </h3>
      <div className="space-y-3">
        {players.map((player) => {
          const delta = player.after - player.before;
          const positive = delta >= 0;
          return (
            <div
              key={player.id}
              className="flex items-center justify-between rounded-xl bg-muted/50 p-3"
            >
              <div className="flex items-center gap-3">
                <PlayerAvatar name={player.name} size="sm" />
                <span className="font-medium text-card-foreground">
                  {player.name}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <div className="flex flex-col items-end">
                  <span className="text-xs text-muted-foreground">赛前</span>
                  <span className="font-semibold tabular-nums text-card-foreground">
                    {Math.round(player.before)}
                  </span>
                </div>
                <div className="text-muted-foreground">→</div>
                <div className="flex flex-col items-end">
                  <span className="text-xs text-muted-foreground">赛后</span>
                  <span className="font-semibold tabular-nums text-card-foreground">
                    {Math.round(player.after)}
                  </span>
                </div>
                <div
                  className={`flex items-center gap-1 rounded-full px-2 py-1 text-xs font-semibold ${
                    positive
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-rose-100 text-rose-700"
                  }`}
                >
                  {positive ? (
                    <TrendingUp className="size-3" />
                  ) : (
                    <TrendingDown className="size-3" />
                  )}
                  {positive ? "+" : ""}
                  {Math.round(delta)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
