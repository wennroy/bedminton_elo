"use client";

import * as React from "react";
import Link from "next/link";
import { PlayerAvatar } from "@/components/player-avatar";
import { cn } from "@/lib/utils";

export interface EloDeltaPlayer {
  id: number;
  name: string;
  before: number;
  after: number;
}

interface EloDeltaCardProps {
  players: EloDeltaPlayer[];
}

/** 成功弹层里的四人前后积分与变化列表（mock delta-list） */
export function EloDeltaCard({ players }: EloDeltaCardProps) {
  return (
    <div>
      {players.map((player) => {
        const before = Math.round(player.before);
        const after = Math.round(player.after);
        const delta = after - before;
        return (
          <div
            key={player.id}
            className="flex items-center justify-between gap-3 border-t border-border py-3 first:border-t-0"
          >
            <Link
              href={`/players/${player.id}`}
              className="flex min-w-0 items-center gap-[11px]"
            >
              <PlayerAvatar name={player.name} size="xs" />
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold text-foreground">
                  {player.name}
                </span>
                <span className="block font-num text-[10px] text-muted-foreground">
                  {before} → {after} ELO
                </span>
              </span>
            </Link>
            <span
              className={cn(
                "font-num text-xl",
                delta >= 0 ? "text-win" : "text-loss"
              )}
            >
              {delta > 0 ? `+${delta}` : delta}
            </span>
          </div>
        );
      })}
    </div>
  );
}
