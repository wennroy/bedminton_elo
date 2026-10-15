"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ChevronRight,
  Pencil,
  Plus,
  RotateCcw,
  Target,
} from "lucide-react";
import { PlayerAvatar } from "@/components/player-avatar";
import { Button } from "@/components/ui/button";
import { useMyPlayerId } from "../use-my-player-id";
import { FixtureTeams } from "../fixture-teams";
import {
  readGenerationOptions,
  type Fixture,
  type ProjectDetail,
} from "../types";

interface DirectoryPlayer {
  id: number;
  name: string;
}

interface ProjectDetailClientProps {
  initialProject: ProjectDetail;
  initialNames: Record<string, string>;
  directory: DirectoryPlayer[];
}

const POLL_MS = 5000;

/** 生成时胜率展示（原快照口径，不得冒充最新预测）。 */
function GeneratedWinRate({ fixture }: { fixture: Fixture }) {
  if (fixture.generatedWinRate === null) return null;
  return (
    <span className="text-xs text-muted-foreground">
      生成时预测 A 队胜率 {Math.round(fixture.generatedWinRate * 100)}%
    </span>
  );
}

/** 跳现有 /predict 页的「当前预测」入口（跟随站点当前评分模型）。 */
function CurrentPredictLink({ fixture }: { fixture: Fixture }) {
  return (
    <Link
      href={`/predict?pa1=${fixture.pa1}&pa2=${fixture.pa2}&pb1=${fixture.pb1}&pb2=${fixture.pb2}`}
      className="inline-flex items-center gap-0.5 text-xs text-muted-foreground transition-colors hover:text-primary"
    >
      <Target className="size-3" />
      当前预测
      <ChevronRight className="size-3" />
    </Link>
  );
}

export function ProjectDetailClient({
  initialProject,
  initialNames,
  directory,
}: ProjectDetailClientProps) {
  const router = useRouter();
  const myId = useMyPlayerId();
  const [project, setProject] = React.useState(initialProject);
  const [names, setNames] = React.useState(initialNames);
  const [missing, setMissing] = React.useState(false);
  const [notice, setNotice] = React.useState<string | null>(null);

  // 改名草稿：独立 state，轮询刷新不会覆盖输入中的内容。
  const [renaming, setRenaming] = React.useState(false);
  const [nameDraft, setNameDraft] = React.useState(initialProject.name);
  const [renameBusy, setRenameBusy] = React.useState(false);

  // 续排草稿（不立即生效，确认后才提交 generate）。
  const [regenOpen, setRegenOpen] = React.useState(false);
  const [roster, setRoster] = React.useState<string[]>(() =>
    initialProject.members.map((m) => m.playerId)
  );
  const [futureMatches, setFutureMatches] = React.useState<number | null>(null);
  const [avoidRematch, setAvoidRematch] = React.useState<boolean | null>(null);
  const [pickerOpen, setPickerOpen] = React.useState(false);
  const [regenBusy, setRegenBusy] = React.useState(false);

  const [busyFixtureId, setBusyFixtureId] = React.useState<number | null>(null);

  const pending = project.fixtures.filter((f) => f.status === "pending");
  const playedUnscored = project.fixtures.filter(
    (f) => f.status === "played_unscored"
  );
  const recorded = project.fixtures.filter((f) => f.status === "recorded");
  const superseded = project.superseded;
  // 「保留 X 场已打比赛，替换 Y 场待打比赛」由详情数据现算。
  const keptCount = playedUnscored.length + recorded.length;
  const replacedCount = pending.length;

  const genOptions = readGenerationOptions(project.latestGeneration?.input);
  const lastMatches = genOptions.matches;
  const matchesDraft =
    futureMatches ?? (pending.length > 0 ? pending.length : (lastMatches ?? 1));
  const avoidDraft = avoidRematch ?? genOptions.avoidPlayedRematches;

  /**
   * 轮询/写后刷新：改名与续排草稿是独立 state，应用新数据不会覆盖正在输入的内容；
   * 若用户正在改名输入框中打字，连 project 对象也保持其可见状态稳定（nameDraft 优先展示）。
   */
  const applyDetail = React.useCallback(
    (next: ProjectDetail, nextNames: Record<string, string>) => {
      setProject(next);
      setNames(nextNames);
    },
    []
  );

  const refresh = React.useCallback(async () => {
    try {
      const res = await fetch(`/api/schedule/projects/${initialProject.id}`, {
        cache: "no-store",
      });
      if (res.status === 404) {
        setMissing(true);
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      applyDetail(data.project, data.names);
    } catch {
      // 网络抖动：保持现状，下个周期再试。
    }
  }, [initialProject.id, applyDetail]);

  // 前台同步：每 5 秒轮询（页面隐藏时暂停）、窗口聚焦刷新。
  React.useEffect(() => {
    const timer = setInterval(() => {
      if (document.hidden) return;
      void refresh();
    }, POLL_MS);
    const onVisibility = () => {
      if (!document.hidden) void refresh();
    };
    const onFocus = () => void refresh();
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearInterval(timer);
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [refresh]);

  async function saveRename() {
    const trimmed = nameDraft.trim();
    if (!trimmed || renameBusy) return;
    setRenameBusy(true);
    setNotice(null);
    try {
      const res = await fetch(`/api/schedule/projects/${project.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: trimmed,
          expectedRevision: project.revision,
          actorId: String(myId),
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setProject((p) => ({ ...p, name: data.name, revision: data.revision }));
        setRenaming(false);
        return;
      }
      const data = await res.json();
      // 409 一律先刷新再提示，不盲目重试写入；退出编辑态展示最新名称。
      await refresh();
      setRenaming(false);
      setNameDraft("");
      if (res.status === 409 && data?.code === "revision_conflict") {
        setNotice("名称刚被其他人改过，已刷新为最新，请确认后再修改");
      } else {
        setNotice(data?.error || "改名失败，请重试");
      }
    } catch {
      setNotice("网络错误，请重试");
    } finally {
      setRenameBusy(false);
    }
  }

  async function markPlayed(fixture: Fixture) {
    if (busyFixtureId !== null || myId === null) return;
    setBusyFixtureId(fixture.id);
    setNotice(null);
    try {
      const res = await fetch(
        `/api/schedule/projects/${project.id}/matches/${fixture.id}/played`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            expectedFixtureRevision: fixture.revision,
            actorId: String(myId),
          }),
        }
      );
      if (res.status === 200) {
        await refresh();
        return;
      }
      const data = await res.json();
      await refresh();
      setNotice(data?.error || "操作未完成，已刷新为最新状态");
    } catch {
      setNotice("网络错误，请重试");
    } finally {
      setBusyFixtureId(null);
    }
  }

  async function submitRegen() {
    if (regenBusy || myId === null) return;
    setRegenBusy(true);
    setNotice(null);
    // 名单不足 4 人：只保存名单、清空未打（matches=0）。
    const matches = roster.length >= 4 ? matchesDraft : 0;
    try {
      const res = await fetch(`/api/schedule/projects/${project.id}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          playerIds: roster,
          matches,
          avoidPlayedRematches: avoidDraft,
          expectedRevision: project.revision,
          actorId: String(myId),
          requestId: crypto.randomUUID(),
        }),
      });
      const data = await res.json();
      if (res.status === 201) {
        applyDetail(data.project, data.names);
        setRegenOpen(false);
        setPickerOpen(false);
        setNotice(
          matches > 0
            ? `已重新生成 ${data.fixtures.length} 场待打比赛`
            : "名单已保存，暂未安排比赛"
        );
        router.refresh();
        return;
      }
      // 409 一律先刷新再提示，不盲目重试写入。
      await refresh();
      if (res.status === 409 && data?.code === "revision_conflict") {
        setNotice("项目刚被其他人更新，已刷新为最新，请确认后再提交");
      } else if (res.status === 409 && data?.code === "rating_unavailable") {
        setNotice("评分服务暂不可用，无法生成新对阵；名单未改动");
      } else {
        setNotice(data?.error || "提交失败，已刷新为最新状态");
      }
    } catch {
      setNotice("网络错误，可安全重试");
    } finally {
      setRegenBusy(false);
    }
  }

  if (missing) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card p-10 text-center text-sm text-muted-foreground">
        项目不存在或已删除。
      </div>
    );
  }

  const nameOf = (id: string) => names[id] ?? "?";
  const identityHint = "请先在右上角选择球员身份";

  return (
    <div data-revision={project.revision} className="flex flex-col gap-5">
      <div>
        <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
          SCHEDULE PROJECT
        </div>
        {renaming ? (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <input
              value={nameDraft}
              onChange={(e) => setNameDraft(e.target.value)}
              maxLength={60}
              aria-label="项目名称"
              className="h-11 min-w-0 flex-1 rounded-xl border border-border bg-background px-3 text-lg font-bold outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            <Button
              onClick={saveRename}
              disabled={renameBusy || !nameDraft.trim()}
            >
              {renameBusy ? "保存中…" : "保存"}
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setRenaming(false);
                setNameDraft(project.name);
              }}
            >
              取消
            </Button>
          </div>
        ) : (
          <div className="mt-2 flex items-center gap-2">
            <h1 className="min-w-0 truncate text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
              {project.name}
            </h1>
            <button
              type="button"
              onClick={() => {
                setNameDraft(project.name);
                setRenaming(true);
              }}
              disabled={myId === null}
              aria-label="改名"
              title={myId === null ? identityHint : "修改项目名称"}
              className="grid size-9 shrink-0 place-items-center rounded-full border border-border bg-card text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
            >
              <Pencil className="size-4" />
            </button>
          </div>
        )}
        <p className="mt-2 text-xs text-muted-foreground">
          场次编号按生成顺序排列，可自由选择开打顺序。
        </p>
        {myId === null && (
          <p className="mt-2 text-xs text-muted-foreground">
            请先在右上角选择球员身份，即可改名、续排、标记已打与记分。
          </p>
        )}
        {notice && (
          <div
            role="status"
            className="mt-3 rounded-xl border border-border bg-muted/40 p-3 text-xs leading-relaxed text-foreground"
          >
            {notice}
          </div>
        )}
      </div>

      <section className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="font-semibold text-card-foreground">
            当前名单 · {project.members.length} 人
          </h2>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setRegenOpen((v) => !v)}
            aria-expanded={regenOpen}
          >
            <RotateCcw />
            调整人员并续排
          </Button>
        </div>
        <ul className="flex flex-wrap gap-2">
          {project.members.map((m) => (
            <li
              key={m.playerId}
              className="flex items-center gap-2 rounded-full border border-border bg-background py-1 pr-3 pl-1"
            >
              <PlayerAvatar name={nameOf(m.playerId)} size="xs" className="size-6 text-[10px]" />
              <span className="text-xs font-medium text-foreground">
                {nameOf(m.playerId)}
              </span>
              <span className="text-[10px] text-muted-foreground">
                已打 {m.actualPlayed} · 待打 {m.pendingCount}
              </span>
            </li>
          ))}
        </ul>
      </section>

      {regenOpen && (
        <section className="flex flex-col gap-4 rounded-2xl border border-primary/30 bg-card p-4 shadow-sm">
          <p className="text-xs leading-relaxed text-muted-foreground">
            修改名单只影响本项目未来的对阵：不会删除球员，也不会删除任何已登记的历史比赛。
          </p>

          <div>
            <div className="mb-2 text-xs font-medium text-muted-foreground">
              目标名单（{roster.length} 人）
            </div>
            <ul className="flex flex-wrap gap-2">
              {roster.map((pid) => (
                <li
                  key={pid}
                  className="flex items-center gap-1.5 rounded-full border border-border bg-background py-1 pr-1.5 pl-1"
                >
                  <PlayerAvatar name={nameOf(pid)} size="xs" className="size-6 text-[10px]" />
                  <span className="text-xs font-medium text-foreground">
                    {nameOf(pid)}
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      setRoster((prev) => prev.filter((id) => id !== pid))
                    }
                    aria-label={`移除 ${nameOf(pid)}`}
                    className="grid size-5 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
            <button
              type="button"
              onClick={() => setPickerOpen((v) => !v)}
              className="mt-2 inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
              aria-expanded={pickerOpen}
            >
              <Plus className="size-3.5" />
              添加球员（含之前离开的球员，可重新加入）
            </button>
            {pickerOpen && (
              <div className="mt-2 grid max-h-56 grid-cols-3 gap-2 overflow-y-auto rounded-xl border border-border bg-background p-2 sm:grid-cols-4">
                {directory
                  .filter((p) => !roster.includes(String(p.id)))
                  .map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => {
                        setRoster((prev) =>
                          [...prev, String(p.id)].sort(
                            (a, b) => Number(a) - Number(b)
                          )
                        );
                      }}
                      className="flex items-center gap-2 rounded-lg border border-border p-2 text-left transition-colors hover:border-primary/50"
                    >
                      <PlayerAvatar name={p.name} size="xs" className="size-6 text-[10px]" />
                      <span className="truncate text-xs font-medium text-foreground">
                        {p.name}
                      </span>
                    </button>
                  ))}
              </div>
            )}
          </div>

          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-card-foreground">
              接下来安排几场
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setFutureMatches((m) => Math.max(0, (m ?? matchesDraft) - 1))}
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
                aria-label="减少续排场次数"
              >
                −
              </button>
              <input
                type="number"
                min={0}
                max={99}
                value={matchesDraft}
                onChange={(e) => {
                  const n = Number(e.target.value);
                  if (Number.isFinite(n)) {
                    setFutureMatches(Math.max(0, Math.min(99, n)));
                  }
                }}
                aria-label="续排场次数"
                className="h-10 w-16 rounded-xl border border-border bg-background text-center text-base font-bold outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              <button
                type="button"
                onClick={() => setFutureMatches((m) => Math.min(99, (m ?? matchesDraft) + 1))}
                className="flex h-9 w-9 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
                aria-label="增加续排场次数"
              >
                +
              </button>
            </div>
          </div>

          <label className="flex items-start gap-3">
            <input
              type="checkbox"
              checked={avoidDraft}
              onChange={(e) => setAvoidRematch(e.target.checked)}
              className="mt-0.5 size-4 accent-primary"
            />
            <span>
              <span className="block text-sm font-medium text-card-foreground">
                减少已打对阵重复
              </span>
              <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">
                优先安排本项目里没打过的对阵，必要时仍可重复
              </span>
            </span>
          </label>

          <div className="flex flex-col gap-2">
            <p className="text-xs text-muted-foreground">
              保留 {keptCount} 场已打比赛，替换 {replacedCount} 场待打比赛
            </p>
            {myId === null && (
              <p className="text-xs text-muted-foreground">{identityHint}</p>
            )}
            <Button
              onClick={submitRegen}
              disabled={regenBusy || myId === null}
              className="h-12 w-full text-base"
            >
              {regenBusy
                ? "提交中…"
                : roster.length >= 4
                  ? "重新生成待打比赛"
                  : "保存名单，暂不安排比赛"}
            </Button>
          </div>
        </section>
      )}

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">
          待打 · {pending.length}
        </h2>
        {pending.length === 0 ? (
          <p className="rounded-2xl border border-dashed border-border bg-card p-6 text-center text-xs text-muted-foreground">
            暂无待打场次，可在上方「调整人员并续排」生成新一批。
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {pending.map((fixture) => (
              <li
                key={fixture.id}
                className="rounded-2xl border border-border bg-card p-3 shadow-sm"
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-muted-foreground">
                    第 {fixture.displayNo} 场
                  </span>
                  <GeneratedWinRate fixture={fixture} />
                </div>
                <FixtureTeams fixture={fixture} names={names} />
                <div className="mt-3 flex items-center justify-between gap-2">
                  <CurrentPredictLink fixture={fixture} />
                  <div className="flex items-center gap-2">
                    <Button asChild size="sm">
                      <Link
                        href={`/schedule/${project.id}/matches/${fixture.id}/record`}
                      >
                        记分
                      </Link>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => markPlayed(fixture)}
                      disabled={myId === null || busyFixtureId !== null}
                      title={myId === null ? identityHint : undefined}
                    >
                      {busyFixtureId === fixture.id ? "标记中…" : "标记已打"}
                    </Button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">
          已打待录分 · {playedUnscored.length}
        </h2>
        {playedUnscored.length === 0 ? (
          <p className="rounded-2xl border border-dashed border-border bg-card p-6 text-center text-xs text-muted-foreground">
            暂无已打待录分的场次。
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {playedUnscored.map((fixture) => (
              <li
                key={fixture.id}
                className="rounded-2xl border border-border bg-card p-3 shadow-sm"
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-muted-foreground">
                    第 {fixture.displayNo} 场
                  </span>
                  <span className="text-xs text-muted-foreground">
                    已于 {fixture.playedAt ?? "未知日期"} 标记已打
                  </span>
                </div>
                <FixtureTeams fixture={fixture} names={names} />
                <div className="mt-3 flex justify-end">
                  <Button asChild variant="outline" size="sm">
                    <Link
                      href={`/schedule/${project.id}/matches/${fixture.id}/record`}
                    >
                      补录比分
                    </Link>
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">
          已登记 · {recorded.length}
        </h2>
        {recorded.length === 0 ? (
          <p className="rounded-2xl border border-dashed border-border bg-card p-6 text-center text-xs text-muted-foreground">
            暂无已登记的比赛。
          </p>
        ) : (
          <ul className="flex flex-col gap-3">
            {recorded.map((fixture) => (
              <li
                key={fixture.id}
                className="rounded-2xl border border-border bg-card p-3 shadow-sm"
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-muted-foreground">
                    第 {fixture.displayNo} 场
                  </span>
                  <GeneratedWinRate fixture={fixture} />
                </div>
                <FixtureTeams fixture={fixture} names={names} />
                <p className="mt-3 text-center font-num text-xl font-bold text-foreground">
                  {fixture.scoreA} : {fixture.scoreB}
                </p>
                <p className="mt-1 text-center text-xs text-muted-foreground">
                  {fixture.markedBy ? `${nameOf(fixture.markedBy)} 录入` : "已录入"} ·{" "}
                  {fixture.playedAt ?? "未知日期"}
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>

      {superseded.length > 0 && (
        <section className="flex flex-col gap-3">
          <h2 className="text-lg font-bold text-foreground">
            已被替换 · {superseded.length}
          </h2>
          <ul className="flex flex-col gap-3">
            {superseded.map((fixture) => (
              <li
                key={fixture.id}
                className="rounded-2xl border border-dashed border-border bg-muted/30 p-3"
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-muted-foreground">
                    第 {fixture.displayNo} 场
                  </span>
                  <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-bold text-muted-foreground">
                    已被替换
                  </span>
                </div>
                <div className="opacity-60">
                  <FixtureTeams fixture={fixture} names={names} />
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
