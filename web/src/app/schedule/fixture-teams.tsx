"use client";

import * as React from "react";
import { PlayerAvatar } from "@/components/player-avatar";
import type { Fixture } from "./types";

interface FixtureTeamsProps {
  fixture: Fixture;
  names: Record<string, string>;
}

/** 场次的固定双方阵容展示（详情卡片与记分页共用，不含任何换人/交换操作）。 */
export function FixtureTeams({ fixture, names }: FixtureTeamsProps) {
  const teamA = [fixture.pa1, fixture.pa2];
  const teamB = [fixture.pb1, fixture.pb2];
  const renderTeam = (ids: string[]) => (
    <div className="flex min-w-0 flex-1 flex-col items-center gap-1">
      <div className="flex -space-x-2">
        {ids.map((id) => (
          <PlayerAvatar key={id} name={names[id] ?? "?"} size="xs" />
        ))}
      </div>
      <span className="max-w-full truncate text-center text-xs font-medium text-card-foreground">
        {ids.map((id) => names[id] ?? "?").join(" / ")}
      </span>
    </div>
  );
  return (
    <div className="flex items-center justify-between gap-2">
      {renderTeam(teamA)}
      <span className="shrink-0 text-sm font-bold text-muted-foreground">VS</span>
      {renderTeam(teamB)}
    </div>
  );
}
