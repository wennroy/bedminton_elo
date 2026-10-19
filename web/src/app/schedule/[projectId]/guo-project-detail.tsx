"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowRight, BookOpen, Check, Pencil, Shuffle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { GuoRound, GuoStanding } from "@/lib/scheduling/guo";
import { useMyPlayerId } from "../use-my-player-id";
import { FixtureTeams } from "../fixture-teams";
import type { Fixture, ProjectDetail } from "../types";

interface Props {
  initialProject: ProjectDetail;
  initialNames: Record<string, string>;
}

const kindLabels = { initial: "实力分组", promotion: "升降换组", cross: "交叉交流" };

function signed(value: number, digits = 0): string {
  const rounded = Number(value.toFixed(digits));
  return `${rounded > 0 ? "+" : ""}${rounded.toFixed(digits)}`;
}

function GroupBoard({ label, standings, fixtures, names, projectId, canRecord }: {
  label: string;
  standings: GuoStanding[];
  fixtures: Fixture[];
  names: Record<string, string>;
  projectId: number;
  canRecord: boolean;
}) {
  const headingId = React.useId();
  return (
    <section aria-labelledby={headingId} className="min-w-0 overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
      <div className="flex items-center justify-between border-b border-border bg-muted/40 px-4 py-3">
        <h3 id={headingId} className="text-sm font-bold text-card-foreground">{label}</h3>
        <span className="text-xs tabular-nums text-muted-foreground">{fixtures.filter((f) => f.status === "recorded").length} / 3 已记分</span>
      </div>
      <div className="p-3 sm:p-4">
        <table className="w-full table-fixed text-right text-[11px] tabular-nums sm:text-xs">
          <caption className="sr-only">{label}个人本轮战绩，按胜场、净胜分、超预期分排序</caption>
          <thead className="text-muted-foreground">
            <tr className="border-b border-border">
              <th scope="col" className="w-[30%] pb-2 text-left font-medium">球员</th>
              <th scope="col" className="w-[23%] pb-2 font-medium">胜场 / 胜率</th>
              <th scope="col" className="pb-2 font-medium">得分</th>
              <th scope="col" className="pb-2 font-medium">净胜分</th>
              <th scope="col" className="pb-2 font-medium">超预期</th>
            </tr>
          </thead>
          <tbody>
            {standings.map((s, i) => (
              <tr key={s.playerId} className={`${i === 1 ? "border-b border-dashed border-border" : ""} ${s.played > 0 && i < 2 ? "text-card-foreground" : "text-muted-foreground"}`}>
                <th scope="row" className="truncate py-2.5 text-left font-medium" title={names[s.playerId]}><span className="mr-1.5 font-mono text-muted-foreground">{i + 1}</span>{names[s.playerId] ?? "?"}</th>
                <td>{s.wins}<span className="ml-1 text-muted-foreground">/ {s.played ? `${Math.round(s.wins / s.played * 100)}%` : "—"}</span></td>
                <td>{s.pointsFor}</td>
                <td className="font-semibold">{signed(s.netPoints)}</td>
                <td>{signed(s.overperformance, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <ol className="mt-4 space-y-3">
          {fixtures.map((f, i) => (
            <li key={f.id} className="rounded-xl border border-border bg-background p-3">
              <div className="mb-3 flex items-center justify-between text-[11px] text-muted-foreground">
                <span>组内第 {i + 1} 场 <span className="ml-1 opacity-60">#{f.displayNo}</span></span>
                <span className={f.status === "recorded" ? "flex items-center gap-1 text-win" : ""}>{f.status === "recorded" ? <><Check className="size-3" />已登记</> : f.status === "played_unscored" ? "已打 · 待补比分" : "待打"}</span>
              </div>
              <FixtureTeams fixture={f} names={names} />
              <div className="mt-3 flex items-center justify-between gap-2 border-t border-border pt-3">
                {f.status === "recorded" ? <span className="font-mono text-lg font-bold tabular-nums text-foreground">{f.scoreA}<span className="mx-2 text-muted-foreground">:</span>{f.scoreB}</span> : canRecord ? (
                  <Button size="sm" asChild><Link href={`/schedule/${projectId}/matches/${f.id}/record`}>{f.status === "played_unscored" ? "补记分" : "记分"}<ArrowRight className="size-3" /></Link></Button>
                ) : <span className="text-[11px] text-muted-foreground">选择球员身份后记分</span>}
                {f.generatedWinRate !== null && <span className="text-[10px] text-muted-foreground">生成时左队胜率 {Math.round(f.generatedWinRate * 100)}%</span>}
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

export function GuoProjectDetail({ initialProject, initialNames }: Props) {
  const myId = useMyPlayerId();
  const [project, setProject] = React.useState(initialProject);
  const [names, setNames] = React.useState(initialNames);
  const initialLatest = initialProject.guo!.rounds.at(-1)!;
  const [selectedRoundNo, setSelectedRoundNo] = React.useState(initialLatest.roundNo);
  const [notice, setNotice] = React.useState<string | null>(null);
  const [missing, setMissing] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [renaming, setRenaming] = React.useState(false);
  const [nameDraft, setNameDraft] = React.useState(initialProject.name);
  const [renameBusy, setRenameBusy] = React.useState(false);
  const revisionRef = React.useRef(project.revision);
  const latestRef = React.useRef(initialLatest.roundNo);
  const requestRef = React.useRef<{ requestId: string; revision: number } | null>(null);

  const applyDetail = React.useCallback((next: ProjectDetail, nextNames: Record<string, string>) => {
    if (next.revision < revisionRef.current || !next.guo) return;
    revisionRef.current = next.revision;
    const previousLatest = latestRef.current;
    const nextLatest = next.guo.rounds.at(-1)!.roundNo;
    latestRef.current = nextLatest;
    setProject(next);
    setNames(nextNames);
    setSelectedRoundNo((current) => current === previousLatest ? nextLatest : current);
  }, []);

  const refresh = React.useCallback(async () => {
    const response = await fetch(`/api/schedule/projects/${initialProject.id}`, { cache: "no-store" });
    if (response.status === 404) { setMissing(true); return; }
    if (!response.ok) return;
    const data = await response.json();
    applyDetail(data.project, data.names);
  }, [initialProject.id, applyDetail]);

  React.useEffect(() => {
    const safelyRefresh = () => { if (!document.hidden) void refresh().catch(() => {}); };
    const timer = setInterval(safelyRefresh, 5000);
    window.addEventListener("focus", safelyRefresh);
    document.addEventListener("visibilitychange", safelyRefresh);
    return () => { clearInterval(timer); window.removeEventListener("focus", safelyRefresh); document.removeEventListener("visibilitychange", safelyRefresh); };
  }, [refresh]);

  async function advance() {
    if (busy || myId === null || !project.guo?.canAdvance) return;
    setBusy(true);
    setNotice(null);
    const pendingRequest = requestRef.current?.revision === project.revision ? requestRef.current : { requestId: crypto.randomUUID(), revision: project.revision };
    requestRef.current = pendingRequest;
    try {
      const response = await fetch(`/api/schedule/projects/${project.id}/generate`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actorId: String(myId), expectedRevision: pendingRequest.revision, requestId: pendingRequest.requestId }),
      });
      const data = await response.json();
      if (response.ok) {
        applyDetail(data.project, data.names);
        setSelectedRoundNo(data.project.guo.rounds.at(-1).roundNo);
        requestRef.current = null;
      } else {
        await refresh();
        setNotice(response.status === 409 ? `${data.error || "项目刚被更新"}，已刷新，请确认最新轮次后操作` : data.error || "开启下一轮失败，请重试");
      }
    } catch {
      setNotice("网络错误，未确认下一轮是否已开启；可直接重试，不会重复开轮");
    } finally { setBusy(false); }
  }

  async function saveName() {
    if (renameBusy || myId === null || !nameDraft.trim()) return;
    setRenameBusy(true);
    setNotice(null);
    try {
      const response = await fetch(`/api/schedule/projects/${project.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: nameDraft.trim(), expectedRevision: project.revision, actorId: String(myId) }) });
      const data = await response.json();
      if (response.ok) {
        if (data.revision >= revisionRef.current) {
          revisionRef.current = data.revision;
          setProject((p) => ({ ...p, name: data.name, revision: data.revision }));
        }
        setRenaming(false);
      } else { await refresh(); setNotice(`${data.error || "改名失败"}，已刷新，请重试`); }
    } catch { setNotice("网络错误，请重试"); }
    finally { setRenameBusy(false); }
  }

  if (missing) return <p className="py-12 text-center text-muted-foreground">项目不存在或已被删除。<Link className="ml-2 underline" href="/schedule">返回配对项目</Link></p>;

  const guo = project.guo!;
  const latest = guo.rounds.at(-1)!;
  const selected = guo.rounds.find((r) => r.roundNo === selectedRoundNo) ?? latest;
  const roundFixtures = (round: GuoRound) => project.fixtures.filter((f) => f.generationId === round.generationId);
  const fixtures = roundFixtures(selected);
  const latestFixtures = roundFixtures(latest);
  const remaining = latestFixtures.filter((f) => f.status !== "recorded").length;
  const recordedCount = fixtures.filter((f) => f.status === "recorded").length;
  const nextRoundNo = latest.roundNo + 1;
  const cross = guo.nextRoundKind === "cross";
  const nextA = [...latest.standingsA.slice(0, 2), ...(cross ? latest.standingsB.slice(2) : latest.standingsB.slice(0, 2))];
  const nextB = [...latest.standingsA.slice(2), ...(cross ? latest.standingsB.slice(0, 2) : latest.standingsB.slice(2))];

  return (
    <div className="flex flex-col gap-5" data-revision={project.revision}>
      <header>
        <div className="flex items-center justify-between gap-3">
          <span className="inline-flex items-center gap-1.5 rounded-md bg-win-bg px-2 py-1 text-[11px] font-bold text-win"><Shuffle className="size-3" />郭老师分配方法</span>
          <Link href="/methodology/pairing#guo" className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"><BookOpen className="size-3.5" />玩法说明</Link>
        </div>
        {renaming ? <div className="mt-3 flex gap-2"><input aria-label="项目名称" value={nameDraft} onChange={(e) => setNameDraft(e.target.value)} maxLength={60} className="h-10 min-w-0 flex-1 rounded-xl border border-border bg-background px-3 outline-none focus-visible:ring-2 focus-visible:ring-ring" /><Button onClick={saveName} disabled={renameBusy || !nameDraft.trim()}>保存</Button><Button variant="outline" onClick={() => setRenaming(false)}>取消</Button></div> : <div className="mt-3 flex items-center gap-2"><h1 className="min-w-0 break-words text-2xl font-bold tracking-tight text-foreground sm:text-3xl">{project.name}</h1><Button variant="ghost" size="icon-sm" aria-label="改名" disabled={myId === null} onClick={() => { setNameDraft(project.name); setRenaming(true); }}><Pencil className="size-4" /></Button></div>}
        <p className="mt-2 text-xs text-muted-foreground">8 人 · 每组 4 人 · 每轮各打 3 场 · 最多 4 轮</p>
      </header>

      {notice && <p role="alert" className="rounded-xl border border-destructive/20 bg-destructive/5 p-3 text-xs leading-relaxed text-destructive">{notice}</p>}

      <nav aria-label="查看轮次" className="flex gap-2">
        {guo.rounds.map((r) => <button key={r.roundNo} type="button" aria-pressed={selected.roundNo === r.roundNo} onClick={() => setSelectedRoundNo(r.roundNo)} className={`flex-1 rounded-xl border px-2 py-3 text-xs font-bold transition-colors ${selected.roundNo === r.roundNo ? "border-primary bg-primary text-primary-foreground" : "border-border bg-card text-muted-foreground hover:text-foreground"}`}>第 {r.roundNo} 轮{r.complete && <Check className="ml-1 inline size-3" />}</button>)}
      </nav>

      <div className="flex items-end justify-between gap-3">
        <div><h2 className="text-xl font-bold text-foreground">第 {selected.roundNo} 轮<span className="ml-2 text-xs font-medium text-muted-foreground">{kindLabels[selected.kind]}</span></h2><p className="mt-1 text-[11px] text-muted-foreground">{selected.kind === "cross" ? "上轮 A 组前二 + B 组后二同场，其余四人另一场。" : selected.kind === "initial" ? "按创建时实力分分组，每人与三位组员各搭档一次。" : "上轮 A 组后二与 B 组前二交换。"}</p></div>
        <span className="shrink-0 font-mono text-sm tabular-nums text-muted-foreground">{recordedCount}<span className="mx-1 opacity-50">/</span>6</span>
      </div>

      <p className="text-[11px] leading-relaxed text-muted-foreground">排名：胜场 → 净胜分 → 超预期分；仍相同按固定抽签顺序。超预期分 = 已赢场数 − 已记分场次的生成时预测胜率之和。未录齐时为暂列排名，每轮重新计算。</p>
      <div className="grid min-w-0 gap-4 xl:grid-cols-2">
        <GroupBoard label="A 组 · 场地 1" standings={selected.standingsA} fixtures={fixtures.filter((f) => selected.groupA.includes(f.pa1))} names={names} projectId={project.id} canRecord={myId !== null} />
        <GroupBoard label="B 组 · 场地 2" standings={selected.standingsB} fixtures={fixtures.filter((f) => selected.groupB.includes(f.pa1))} names={names} projectId={project.id} canRecord={myId !== null} />
      </div>

      <section className="rounded-2xl border border-primary/25 bg-primary/5 p-4 sm:p-5">
        {latest.roundNo >= guo.maxRounds ? <p className="text-sm font-bold text-foreground">{latest.complete ? "四轮已完成，今天辛苦了！" : "最后一轮，记齐六场比分即可收官。"}</p> : <>
          <h2 className="text-sm font-bold text-foreground">{latest.roundNo === 3 && latest.complete ? "三轮已打，可以收拍，也可以再来一轮" : `下一轮 · ${cross ? "交叉交流" : "升降换组"}`}</h2>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{cross ? "按第 2 轮排名，A 组前二与 B 组后二进入场地 1，其余四人进入场地 2。" : `按第 ${latest.roundNo} 轮排名，A 组后二与 B 组前二交换。`}</p>
          {guo.canAdvance ? <div className="my-3 grid gap-2 text-xs sm:grid-cols-2"><p><span className="mr-2 font-semibold text-win">下轮 A 组</span>{nextA.map((s) => names[s.playerId] ?? "?").join("、")}</p><p><span className="mr-2 font-semibold text-win">下轮 B 组</span>{nextB.map((s) => names[s.playerId] ?? "?").join("、")}</p></div> : <p className="mt-3 text-xs font-medium text-foreground">第 {latest.roundNo} 轮还差 {remaining} 场比分，录齐后才能开启下一轮。</p>}
          <Button size="lg" className="mt-3 h-11 w-full sm:w-auto" onClick={advance} disabled={!guo.canAdvance || busy || myId === null}>{busy ? "正在开轮…" : latest.roundNo === 3 ? "再打第 4 轮" : `开启第 ${nextRoundNo} 轮`}<ArrowRight className="size-4" /></Button>
        </>}
        {myId === null && <p className="mt-2 text-xs text-muted-foreground">请先在右上角选择球员身份，再记分或开启下一轮。</p>}
      </section>
      <p className="text-[11px] leading-relaxed text-muted-foreground">每轮分组保存的是生成时的排名结果；之后修改或撤回旧比分，会更新该轮战绩，已生成的后续轮保持原分组。第三轮起 A、B 只区分场地。名单固定为这 8 人，如需换人请新建项目。</p>
    </div>
  );
}
