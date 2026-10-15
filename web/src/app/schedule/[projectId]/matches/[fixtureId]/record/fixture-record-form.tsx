"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Check, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PlayerAvatar } from "@/components/player-avatar";
import { ScoreInput } from "@/components/score-input";
import {
  MatchRatingResult,
  type MatchRatingFieldClient,
} from "@/components/match-rating-result";
import type { EloDeltaPlayer } from "@/components/elo-delta-card";
import { useMyPlayerId } from "../../../../use-my-player-id";
import { FixtureTeams } from "../../../../fixture-teams";
import type { Fixture } from "../../../../types";

interface FixtureRecordFormProps {
  projectId: number;
  projectName: string;
  fixture: Fixture;
  names: Record<string, string>;
}

interface RecordSuccess {
  scoreA: number;
  scoreB: number;
  rating: MatchRatingFieldClient | null;
  legacyPlayers: EloDeltaPlayer[];
  legacyAvailable: boolean;
  slotIds: [number, number, number, number];
  playerMap: Map<number, { id: number; name: string }>;
}

/**
 * 项目场次记分：从 DB 载入的固定阵容（不允许换人/交换双方）。
 * 不占用比赛、无编辑锁；提交遇 409（已登记/被替换/状态变化）先刷新为只读再提示。
 */
export function FixtureRecordForm({
  projectId,
  projectName,
  fixture,
  names,
}: FixtureRecordFormProps) {
  const router = useRouter();
  const myId = useMyPlayerId();
  const [current, setCurrent] = React.useState(fixture);
  const [scoreA, setScoreA] = React.useState(21);
  const [scoreB, setScoreB] = React.useState(0);
  const [submitting, setSubmitting] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<RecordSuccess | null>(null);

  const readOnly = current.status === "recorded" || current.status === "superseded";
  const playerMap = React.useMemo(() => {
    const map = new Map<number, { id: number; name: string }>();
    for (const id of [current.pa1, current.pa2, current.pb1, current.pb2]) {
      map.set(Number(id), { id: Number(id), name: names[id] ?? "?" });
    }
    return map;
  }, [current, names]);
  const slotIds: [number, number, number, number] = [
    Number(current.pa1),
    Number(current.pa2),
    Number(current.pb1),
    Number(current.pb2),
  ];

  const scoresValid =
    Number.isInteger(scoreA) &&
    Number.isInteger(scoreB) &&
    scoreA >= 0 &&
    scoreA <= 99 &&
    scoreB >= 0 &&
    scoreB <= 99;
  const canSubmit =
    !readOnly && scoresValid && scoreA !== scoreB && myId !== null && !submitting;

  async function refreshLatest() {
    try {
      const res = await fetch(`/api/schedule/projects/${projectId}`, {
        cache: "no-store",
      });
      if (!res.ok) return;
      const data = await res.json();
      const latest = [...data.project.fixtures, ...data.project.superseded].find(
        (f: Fixture) => f.id === current.id
      );
      if (latest) setCurrent(latest);
    } catch {
      // 刷新失败保持现状，错误提示仍展示。
    }
  }

  async function handleSubmit() {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/schedule/projects/${projectId}/matches/${current.id}/score`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scoreA,
            scoreB,
            expectedFixtureRevision: current.revision,
            actorId: String(myId),
          }),
        }
      );
      const data = await res.json();
      if (res.status === 201) {
        const legacyAvailable: boolean =
          data.legacyRatingsAvailable !== false &&
          Array.isArray(data.before) &&
          Array.isArray(data.after);
        const legacyPlayers: EloDeltaPlayer[] = legacyAvailable
          ? data.after.map(
              (item: { id: number; name: string; elo: number }, index: number) => ({
                id: item.id,
                name: item.name,
                before: data.before[index].elo,
                after: item.elo,
              })
            )
          : [];
        setResult({
          scoreA,
          scoreB,
          rating: (data.rating ?? null) as MatchRatingFieldClient | null,
          legacyPlayers,
          legacyAvailable,
          slotIds,
          playerMap,
        });
        if (data.fixture) setCurrent(data.fixture);
        router.refresh();
        return;
      }
      if (res.status === 409) {
        // 先刷新为只读现状，再按 409 语义提示，不盲目重试写入。
        await refreshLatest();
        if (data?.code === "already_recorded") {
          setError("这场已由其他人登记比分，已刷新为最新结果");
        } else if (data?.code === "fixture_superseded") {
          setError("这场已被新一批对阵替换，已刷新为最新状态");
        } else if (data?.code === "revision_conflict") {
          setError("这场状态刚有变化，已刷新，请确认后再提交");
        } else {
          setError(data?.error || "提交冲突，已刷新最新状态");
        }
        return;
      }
      setError(data?.error || "提交失败，请重试");
    } catch {
      setError("网络错误，可安全重试");
    } finally {
      setSubmitting(false);
    }
  }

  if (result) {
    return (
      <div className="mx-auto flex w-full max-w-xl flex-col gap-4">
        <div className="rounded-2xl border border-border bg-card p-6 text-center shadow-sm">
          <div className="mx-auto grid size-[54px] place-items-center rounded-full bg-primary text-primary-foreground">
            <Check className="size-6" strokeWidth={2} />
          </div>
          <h2 className="mt-3 text-center text-xl font-bold text-card-foreground">
            比赛已登记
          </h2>
          <p className="mt-1 text-center font-num text-2xl font-bold text-foreground">
            {result.scoreA} : {result.scoreB}
          </p>
          <div className="mt-4 text-left">
            <MatchRatingResult
              rating={result.rating}
              legacyPlayers={result.legacyPlayers}
              legacyAvailable={result.legacyAvailable}
              slotIds={result.slotIds}
              playerMap={result.playerMap}
            />
          </div>
          <Button asChild className="mt-5 w-full" size="lg">
            <Link href={`/schedule/${projectId}`}>
              返回项目
              <ChevronRight />
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  if (readOnly) {
    return (
      <div className="mx-auto flex w-full max-w-xl flex-col gap-4">
        <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
          <div className="mb-3 flex items-center justify-between gap-2">
            <span className="text-xs font-bold text-muted-foreground">
              第 {current.displayNo} 场
            </span>
            <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-bold text-muted-foreground">
              {current.status === "recorded" ? "已登记" : "已被替换"}
            </span>
          </div>
          <FixtureTeams fixture={current} names={names} />
          {current.status === "recorded" ? (
            <>
              <p className="mt-3 text-center font-num text-2xl font-bold text-foreground">
                {current.scoreA} : {current.scoreB}
              </p>
              <p className="mt-1 text-center text-xs text-muted-foreground">
                {current.markedBy ? `${names[current.markedBy] ?? "?"} 录入` : "已录入"}{" "}
                · {current.playedAt ?? "未知日期"}
              </p>
            </>
          ) : (
            <p className="mt-3 text-center text-xs text-muted-foreground">
              这场对阵已被新一批替换，仅保留历史记录。
            </p>
          )}
          {error && (
            <p role="alert" className="mt-3 text-center text-xs text-destructive">
              {error}
            </p>
          )}
          <Button asChild variant="outline" className="mt-4 w-full">
            <Link href={`/schedule/${projectId}`}>返回项目</Link>
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-xl flex-col gap-4">
      <section
        aria-label="双打比赛记分板"
        className="relative overflow-hidden rounded-[18px] bg-court p-[18px] text-[#f1f5e9] min-[761px]:p-[25px]"
      >
        <div className="mb-4 flex items-center justify-between min-[761px]:mb-[25px]">
          <span className="text-[10px] font-bold tracking-[2px] text-[#bcc8b7] max-[760px]:text-[9px]">
            第 {current.displayNo} 场 · 阵容已固定
          </span>
          <small className="text-[10px] text-[#bac6b4]">FINAL SCORE</small>
        </div>
        <div className="relative grid grid-cols-2 gap-[22px] min-[761px]:gap-6">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0 left-1/2 border-l border-dashed border-[#75816866]"
          />
          {(
            [
              [0, scoreA, setScoreA],
              [1, scoreB, setScoreB],
            ] as const
          ).map(([team, score, setScore]) => {
            const label = team === 0 ? "A" : "B";
            const teamIds = team === 0 ? [current.pa1, current.pa2] : [current.pb1, current.pb2];
            return (
              <div key={team} className="min-w-0 text-center">
                <span
                  className="inline-flex items-center gap-[7px] text-[10px] font-bold tracking-[1.4px]"
                  style={{ color: team === 0 ? "var(--team-a)" : "var(--team-b)" }}
                >
                  <span className="size-[5px] rounded-full bg-current" />
                  TEAM {label} / {label} 队
                </span>
                <div className="mt-3 flex flex-col items-center gap-1">
                  <div className="flex -space-x-2">
                    {teamIds.map((id) => (
                      <PlayerAvatar
                        key={id}
                        name={names[id] ?? "?"}
                        size="xs"
                        className="size-[31px] text-xs"
                      />
                    ))}
                  </div>
                  <span className="max-w-full truncate text-xs font-medium text-[#e2e9d9]">
                    {teamIds.map((id) => names[id] ?? "?").join(" / ")}
                  </span>
                </div>
                <ScoreInput team={team} score={score} onScoreChange={setScore} />
              </div>
            );
          })}
        </div>
      </section>

      {current.status === "played_unscored" && (
        <p className="rounded-xl border border-border bg-muted/40 p-3 text-center text-xs leading-relaxed text-muted-foreground">
          这场已于 {current.playedAt ?? "未知日期"}
          标记为已打，补录比分将保留原比赛日期。
        </p>
      )}

      {error && (
        <p
          role="alert"
          className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-center text-sm text-destructive"
        >
          {error}
        </p>
      )}
      {myId === null && (
        <p className="text-center text-xs text-muted-foreground">
          请先在右上角选择球员身份，再登记比分。
        </p>
      )}

      <Button
        size="lg"
        disabled={!canSubmit}
        onClick={handleSubmit}
        className="h-[52px] w-full rounded-[9px] text-sm font-bold"
      >
        <Check strokeWidth={2} />
        {submitting ? "正在登记…" : "确认登记比分"}
        <ChevronRight strokeWidth={1.65} />
      </Button>
      <p className="text-center text-[10px] text-muted-foreground">
        {scoresValid && scoreA !== scoreB
          ? `提交后为 ${projectName} 的这场比赛写入最终比分`
          : "比赛不能以平局结束，请输入最终比分"}
      </p>
    </div>
  );
}
