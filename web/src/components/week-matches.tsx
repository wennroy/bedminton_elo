"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight, Undo2 } from "lucide-react";
import { PlayerAvatar } from "@/components/player-avatar";
import { formatMatchMeta } from "@/lib/format";
import { Button } from "@/components/ui/button";
import { getMyPlayerId } from "@/lib/identity";
import { cn } from "@/lib/utils";

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
  /** 全部比赛（升序）；已选身份时过滤本人最近 4 场 */
  matches: MatchWithNames[];
  /** 本周一 YYYY-MM-DD；未选身份时回退显示本周全部比赛 */
  weekStart: string;
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

const pillClass =
  "inline-flex items-center rounded-[5px] px-[7px] py-1 text-[10px] font-bold";

export function WeekMatches({ matches, weekStart }: WeekMatchesProps) {
  const router = useRouter();
  const [deleting, setDeleting] = React.useState<number | null>(null);
  const [myId, setMyId] = React.useState<number | null>(null);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

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

  // 已选身份：本人最近 4 场（matchMini 视角，比分从我方出发）
  const myMatches = React.useMemo(() => {
    if (myId === null) return [];
    return matches
      .filter((m) => [m.pa1, m.pa2, m.pb1, m.pb2].includes(myId))
      .map((m) => {
        const inA = m.pa1 === myId || m.pa2 === myId;
        return {
          id: m.id,
          date: m.playedAt,
          team: inA ? [m.pa1Name, m.pa2Name] : [m.pb1Name, m.pb2Name],
          opponents: inA ? [m.pb1Name, m.pb2Name] : [m.pa1Name, m.pa2Name],
          scoreFor: inA ? m.scoreA : m.scoreB,
          scoreAgainst: inA ? m.scoreB : m.scoreA,
          won: inA ? m.scoreA > m.scoreB : m.scoreB > m.scoreA,
        };
      })
      .sort((a, b) => b.date.localeCompare(a.date) || b.id - a.id)
      .slice(0, 4);
  }, [matches, myId]);

  // 未选身份：本周全部比赛，按日分组
  const groups = React.useMemo(() => {
    const sorted = matches
      .filter((m) => m.playedAt >= weekStart)
      .sort((a, b) => {
        if (a.playedAt !== b.playedAt)
          return b.playedAt.localeCompare(a.playedAt);
        return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      });
    const map = new Map<string, MatchWithNames[]>();
    for (const m of sorted) {
      const list = map.get(m.playedAt) ?? [];
      list.push(m);
      map.set(m.playedAt, list);
    }
    return Array.from(map.entries());
  }, [matches, weekStart]);

  const hasIdentity = myId !== null;

  return (
    <div>
      <div className="mb-5 flex items-center justify-between gap-3.5">
        <h2 className="text-lg font-bold text-card-foreground">
          {hasIdentity ? "我的最近比赛" : "本周比赛"}
        </h2>
        <Link
          href={hasIdentity ? `/players/${myId}` : "/players"}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-win"
        >
          {hasIdentity ? "全部战绩" : "全部球员"}
          <ArrowRight className="size-[15px]" strokeWidth={1.65} />
        </Link>
      </div>

      {hasIdentity ? (
        myMatches.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
            最近还没有你的比赛，记一场吧。
          </div>
        ) : (
          <div>
            {myMatches.map((m) => (
              <div
                key={m.id}
                className="border-t border-border py-4 first:border-t-0 first:pt-0 last:pb-0"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-muted-foreground">
                    {m.date.slice(5).replace("-", ".")} · 双打
                  </span>
                  <span
                    className={cn(
                      pillClass,
                      m.won
                        ? "bg-win-bg text-win"
                        : "bg-loss-bg text-loss"
                    )}
                  >
                    {m.won ? "胜" : "负"}
                  </span>
                </div>
                <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
                  <span className="text-xs text-card-foreground">
                    {m.team.join(" / ")}
                  </span>
                  <div className="font-num text-[23px] text-card-foreground">
                    {m.scoreFor}
                    <span className="px-1.5 text-[13px] text-muted-foreground">
                      :
                    </span>
                    {m.scoreAgainst}
                  </div>
                  <span className="text-right text-xs text-muted-foreground">
                    {m.opponents.join(" / ")}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )
      ) : groups.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          本周还没有比赛
        </div>
      ) : (
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
                          <span className="font-num text-xl font-bold text-card-foreground">
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
      )}
    </div>
  );
}
