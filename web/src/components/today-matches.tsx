"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { PlayerAvatar } from "@/components/player-avatar";
import { Button } from "@/components/ui/button";
import { Undo2 } from "lucide-react";

interface MatchWithNames {
  id: number;
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
  enteredBy: number | null;
  createdAt: string;
  pa1Name: string;
  pa2Name: string;
  pb1Name: string;
  pb2Name: string;
}

interface TodayMatchesProps {
  matches: MatchWithNames[];
}

const TEN_MINUTES = 10 * 60 * 1000;

export function TodayMatches({ matches }: TodayMatchesProps) {
  const router = useRouter();
  const [deleting, setDeleting] = React.useState<number | null>(null);

  async function handleDelete(id: number) {
    if (!window.confirm("确定撤回这场比赛？")) return;
    setDeleting(id);
    try {
      const response = await fetch(`/api/matches/${id}`, { method: "DELETE" });
      if (!response.ok) {
        const data = await response.json();
        alert(data.error || "撤回失败");
        return;
      }
      router.refresh();
    } finally {
      setDeleting(null);
    }
  }

  if (matches.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
        今天还没有比赛
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {matches.map((match) => {
        const createdAt = new Date(match.createdAt).getTime();
        const canDelete = Date.now() - createdAt < TEN_MINUTES;
        return (
          <div
            key={match.id}
            className="rounded-2xl border border-border bg-card p-3 shadow-sm"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex flex-1 flex-col gap-2">
                <div className="flex items-center gap-2">
                  <div className="flex -space-x-2">
                    <PlayerAvatar name={match.pa1Name} size="xs" />
                    <PlayerAvatar name={match.pa2Name} size="xs" />
                  </div>
                  <span className="text-sm font-medium text-card-foreground">
                    {match.pa1Name} / {match.pa2Name}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex -space-x-2">
                    <PlayerAvatar name={match.pb1Name} size="xs" />
                    <PlayerAvatar name={match.pb2Name} size="xs" />
                  </div>
                  <span className="text-sm font-medium text-card-foreground">
                    {match.pb1Name} / {match.pb2Name}
                  </span>
                </div>
              </div>
              <div className="flex flex-col items-end gap-1">
                <span className="text-xl font-bold tabular-nums text-card-foreground">
                  {match.scoreA} : {match.scoreB}
                </span>
                {canDelete && (
                  <Button
                    variant="destructive"
                    size="xs"
                    disabled={deleting === match.id}
                    onClick={() => handleDelete(match.id)}
                  >
                    <Undo2 className="size-3" />
                    撤回
                  </Button>
                )}
              </div>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">
              录入：
              {match.enteredBy === null
                ? "匿名"
                : [match.pa1, match.pa2, match.pb1, match.pb2].includes(
                    match.enteredBy
                  )
                ? [
                    match.pa1Name,
                    match.pa2Name,
                    match.pb1Name,
                    match.pb2Name,
                  ][
                    [match.pa1, match.pa2, match.pb1, match.pb2].indexOf(
                      match.enteredBy
                    )
                  ]
                : "未知"}
            </div>
          </div>
        );
      })}
    </div>
  );
}
