"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { PlayerAvatar } from "@/components/player-avatar";
import { formatMatchMeta } from "@/lib/format";
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

interface WeekMatchesProps {
  matches: MatchWithNames[];
}

const TEN_MINUTES = 10 * 60 * 1000;
const WEEKDAYS = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];

function localDateString(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function dayLabel(playedAt: string): string {
  const today = localDateString(new Date());
  if (playedAt === today) return "今天";
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  if (playedAt === localDateString(yesterday)) return "昨天";
  const [y, m, d] = playedAt.split("-").map(Number);
  const date = new Date(y, m - 1, d);
  return `${m}月${d}日 ${WEEKDAYS[date.getDay()]}`;
}

function enteredByLabel(match: MatchWithNames): string {
  if (match.enteredBy === null) return "匿名";
  const ids = [match.pa1, match.pa2, match.pb1, match.pb2];
  const names = [match.pa1Name, match.pa2Name, match.pb1Name, match.pb2Name];
  const idx = ids.indexOf(match.enteredBy);
  return idx >= 0 ? names[idx] : "未知";
}

export function WeekMatches({ matches }: WeekMatchesProps) {
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

  const groups = React.useMemo(() => {
    const sorted = [...matches].sort((a, b) => {
      if (a.playedAt !== b.playedAt) return b.playedAt.localeCompare(a.playedAt);
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });
    const map = new Map<string, MatchWithNames[]>();
    for (const m of sorted) {
      const list = map.get(m.playedAt) ?? [];
      list.push(m);
      map.set(m.playedAt, list);
    }
    return Array.from(map.entries());
  }, [matches]);

  if (matches.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
        本周还没有比赛
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {groups.map(([date, dayMatches]) => (
        <div key={date}>
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            {dayLabel(date)} · {dayMatches.length} 场
          </div>
          <div className="space-y-2">
            {dayMatches.map((match) => {
              const createdAt = new Date(match.createdAt).getTime();
              const canDelete = Date.now() - createdAt < TEN_MINUTES;
              return (
                <div
                  key={match.id}
                  className="rounded-xl border border-border bg-background p-3"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex flex-1 flex-col gap-1.5">
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
                  <div className="mt-1.5 text-xs text-muted-foreground">
                    {formatMatchMeta(
                      match.playedAt,
                      match.createdAt,
                      enteredByLabel(match)
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
