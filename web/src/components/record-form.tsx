"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowLeftRight,
  ArrowRight,
  Calendar,
  Check,
  ChevronRight,
  Plus,
  RotateCcw,
  Search,
  UserPlus,
} from "lucide-react";
import { getMyPlayerId, setMyPlayerId } from "@/lib/identity";
import { PlayerAvatar } from "@/components/player-avatar";
import { IdentityPicker } from "@/components/identity-picker";
import { EloDeltaCard, type EloDeltaPlayer } from "@/components/elo-delta-card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface Player {
  id: number;
  name: string;
}

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

interface RecordFormProps {
  players: Player[];
  /** 配对页跳转带来的预填阵容(A1,A2,B1,B2),缺省全空 */
  initialSlots?: [Slot, Slot, Slot, Slot];
  /** 全部比赛（升序），用于侧栏「我的最近一场」 */
  matches: MatchWithNames[];
}

type Slot = number | null;

function todayString(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export function RecordForm({ players, initialSlots, matches }: RecordFormProps) {
  const router = useRouter();
  const [myId, setMyId] = React.useState<number | null>(null);
  const [slots, setSlots] = React.useState<[Slot, Slot, Slot, Slot]>(
    initialSlots ?? [null, null, null, null]
  );
  const [scoreA, setScoreA] = React.useState(21);
  const [scoreB, setScoreB] = React.useState(0);
  const [submitting, setSubmitting] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = React.useState(false);
  const [result, setResult] = React.useState<{
    players: EloDeltaPlayer[];
    enteredBy: number;
  } | null>(null);
  const [toast, setToast] = React.useState<string | null>(null);
  const toastTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // 阵容槽位 picker
  const [pickerSlot, setPickerSlot] = React.useState<number | null>(null);
  const [pickerSearch, setPickerSearch] = React.useState("");
  const [adding, setAdding] = React.useState(false);
  const [addName, setAddName] = React.useState("");
  const [addError, setAddError] = React.useState<string | null>(null);
  const [addBusy, setAddBusy] = React.useState(false);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  function showToast(message: string) {
    setToast(message);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 3500);
  }

  const playerMap = React.useMemo(
    () => new Map(players.map((p) => [p.id, p])),
    [players]
  );

  const filled = slots.filter((id): id is number => id !== null);
  const isDistinct = new Set(filled).size === filled.length;
  const scoresValid =
    Number.isInteger(scoreA) &&
    Number.isInteger(scoreB) &&
    scoreA >= 0 &&
    scoreA <= 99 &&
    scoreB >= 0 &&
    scoreB <= 99;
  const canSubmit =
    filled.length === 4 &&
    isDistinct &&
    scoresValid &&
    scoreA !== scoreB &&
    myId !== null;

  function recordHint(): string {
    if (filled.length < 4) {
      return `已选 ${filled.length} / 4 人 · 选好双方球员后确认比分`;
    }
    if (!isDistinct) return "同一位球员不能出现在两个位置。";
    if (!scoresValid) return "请输入 0–99 的整数比分。";
    if (scoreA === scoreB) return "比赛不能以平局结束，请填写最终比分。";
    if (myId === null) return "请先选择录入人身份，再确认比分。";
    return "确认后展示四位球员的 ELO 变化";
  }

  function openSlotPicker(index: number) {
    setPickerSlot(index);
    setPickerSearch("");
    setAdding(false);
    setAddName("");
    setAddError(null);
  }

  function pickPlayer(id: number) {
    if (pickerSlot === null) return;
    setSlots((current) => {
      const next: [Slot, Slot, Slot, Slot] = [...current];
      next[pickerSlot] = id;
      return next;
    });
    setPickerSlot(null);
  }

  function removeSlot() {
    if (pickerSlot === null) return;
    setSlots((current) => {
      const next: [Slot, Slot, Slot, Slot] = [...current];
      next[pickerSlot] = null;
      return next;
    });
    setPickerSlot(null);
  }

  function adjustScore(team: 0 | 1, delta: number) {
    const setter = team === 0 ? setScoreA : setScoreB;
    setter((value) =>
      Math.max(0, Math.min(99, (Number.isFinite(value) ? value : 0) + delta))
    );
  }

  function swapSides() {
    setSlots((current) => [current[2], current[3], current[0], current[1]]);
    const a = scoreA;
    setScoreA(scoreB);
    setScoreB(a);
    showToast("双方阵容和比分已一起交换。");
  }

  function clearSlots() {
    setSlots([null, null, null, null]);
    showToast("已清空阵容，比分已保留。");
  }

  async function handleSubmit() {
    if (!canSubmit || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch("/api/matches", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pa1: slots[0],
          pa2: slots[1],
          pb1: slots[2],
          pb2: slots[3],
          scoreA,
          scoreB,
          playedAt: todayString(),
          enteredBy: myId,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.error || "提交失败，请重试");
        return;
      }
      const deltas: EloDeltaPlayer[] = data.after.map(
        (item: { id: number; name: string; elo: number }, index: number) => ({
          id: item.id,
          name: item.name,
          before: data.before[index].elo,
          after: item.elo,
        })
      );
      setConfirmOpen(false);
      setResult({ players: deltas, enteredBy: myId! });
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "提交失败，请重试");
    } finally {
      setSubmitting(false);
    }
  }

  function nextRecord() {
    setScoreA(21);
    setScoreB(0);
    setResult(null);
    showToast("已保留双方阵容，可以继续记分或更换球员。");
  }

  // Courtside case: recorder adds someone ELSE, so identity stays untouched.
  async function handleAddPlayer(event: React.FormEvent) {
    event.preventDefault();
    const name = addName.trim();
    if (!name || addBusy) return;
    setAddBusy(true);
    setAddError(null);
    try {
      const res = await fetch("/api/players", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) {
        setAddError(data.error || "添加失败");
        return;
      }
      // 新球员直接落入当前槽位；router.refresh 让名单同步
      if (pickerSlot !== null && Number.isInteger(data.id)) {
        setSlots((current) => {
          const next: [Slot, Slot, Slot, Slot] = [...current];
          next[pickerSlot] = data.id;
          return next;
        });
      }
      setPickerSlot(null);
      router.refresh();
      showToast(`已添加球员：${name}`);
    } catch {
      setAddError("网络错误，请重试");
    } finally {
      setAddBusy(false);
    }
  }

  const me = myId === null ? null : playerMap.get(myId) ?? null;

  // 侧栏「我的最近一场」（客户端按身份过滤，matchMini 视角）
  const myLatest = React.useMemo(() => {
    if (myId === null) return null;
    const m = [...matches]
      .reverse()
      .find((m) => [m.pa1, m.pa2, m.pb1, m.pb2].includes(myId));
    if (!m) return null;
    const inA = m.pa1 === myId || m.pa2 === myId;
    return {
      date: m.playedAt,
      team: inA ? [m.pa1Name, m.pa2Name] : [m.pb1Name, m.pb2Name],
      opponents: inA ? [m.pb1Name, m.pb2Name] : [m.pa1Name, m.pa2Name],
      scoreFor: inA ? m.scoreA : m.scoreB,
      scoreAgainst: inA ? m.scoreB : m.scoreA,
      won: inA ? m.scoreA > m.scoreB : m.scoreB > m.scoreA,
    };
  }, [matches, myId]);

  const pickerTerm = pickerSearch.trim();
  const pickerVisible = pickerTerm
    ? players.filter((p) => p.name.includes(pickerTerm))
    : players;
  const currentSlotValue = pickerSlot === null ? null : slots[pickerSlot];

  const teamColumn = (team: 0 | 1) => {
    const label = team === 0 ? "A" : "B";
    const score = team === 0 ? scoreA : scoreB;
    const setScore = team === 0 ? setScoreA : setScoreB;
    return (
      <div className="min-w-0 text-center">
        <span
          className="inline-flex items-center gap-[7px] text-[10px] font-bold tracking-[1.4px]"
          style={{ color: team === 0 ? "var(--team-a)" : "var(--team-b)" }}
        >
          <span className="size-[5px] rounded-full bg-current" />
          TEAM {label} / {label} 队
        </span>
        <div className="mt-[17px] grid grid-cols-2 gap-2 max-[760px]:gap-1.5">
          {[team * 2, team * 2 + 1].map((index) => {
            const slotValue = slots[index];
            const player = slotValue === null ? null : playerMap.get(slotValue);
            return (
              <button
                key={index}
                type="button"
                onClick={() => openSlotPicker(index)}
                aria-label={`选择${label}队${(index % 2) + 1}号球员${
                  player ? `，当前${player.name}` : ""
                }`}
                className={cn(
                  "flex min-h-[80px] flex-col items-center justify-center gap-[7px] rounded-[10px] border px-[3px] py-[9px] text-[11px] transition-colors min-[761px]:min-h-[91px] min-[761px]:py-3 min-[761px]:text-xs",
                  player
                    ? "border-[#66735b] text-[#e2e9d9] hover:border-[#d3f36b] hover:bg-[#ffffff0d]"
                    : "border-dashed border-[#66735b] text-[#c0ccb8] hover:border-[#d3f36b] hover:bg-[#ffffff0d]"
                )}
              >
                {player ? (
                  <>
                    <PlayerAvatar
                      name={player.name}
                      size="xs"
                      className="size-[31px] text-xs"
                    />
                    <span className="max-w-full truncate">{player.name}</span>
                  </>
                ) : (
                  <>
                    <Plus className="size-[26px]" strokeWidth={1.65} />
                    <span>选择球员</span>
                  </>
                )}
              </button>
            );
          })}
        </div>
        <input
          type="number"
          inputMode="numeric"
          min={0}
          max={99}
          step={1}
          value={Number.isFinite(score) ? score : ""}
          onChange={(e) => {
            const raw = e.target.value;
            setScore(raw === "" ? NaN : Number(raw));
          }}
          aria-label={`${label}队比分`}
          className="w-full min-w-0 appearance-none border-0 bg-transparent py-3 text-center font-num text-[84px] leading-none font-bold tracking-[-4px] outline-none min-[761px]:py-[17px] min-[761px]:pb-3 min-[761px]:text-[103px] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
          style={{
            color: team === 0 ? "var(--team-a)" : "var(--team-b)",
            MozAppearance: "textfield",
          }}
        />
        <div className="flex justify-center gap-[9px] min-[761px]:gap-2.5">
          <button
            type="button"
            onClick={() => adjustScore(team, -1)}
            disabled={!(score > 0)}
            aria-label={`${label}队减一分`}
            className="h-[46px] w-[54px] rounded-lg border border-[#617057] text-[22px] text-[#f0f4e9] transition-colors hover:bg-[#ffffff0d] disabled:opacity-40 min-[761px]:h-11 min-[761px]:w-[58px]"
          >
            −
          </button>
          <button
            type="button"
            onClick={() => adjustScore(team, 1)}
            disabled={!(score < 99)}
            aria-label={`${label}队加一分`}
            className="h-[46px] w-[54px] rounded-lg border border-[#617057] bg-[#ffffff12] text-[22px] text-[#f0f4e9] transition-colors hover:bg-[#ffffff1f] disabled:opacity-40 min-[761px]:h-11 min-[761px]:w-[58px]"
          >
            +
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="mx-auto grid w-full max-w-[1060px] items-start gap-6 min-[761px]:grid-cols-[minmax(0,1.65fr)_minmax(260px,1fr)] min-[761px]:gap-6">
      <div className="min-w-0">
        <div className="mb-4 flex items-center justify-between gap-3 text-[11px] text-muted-foreground max-[760px]:mb-3 max-[760px]:text-[10px]">
          <span className="flex items-center gap-[7px]">
            <Calendar className="size-[15px]" strokeWidth={1.65} />
            {todayString().replaceAll("-", ".")} · 今天
          </span>
          {/* 未选身份时 IdentityPicker 自动弹出；已选则可点「录入人」切换 */}
          <IdentityPicker
            players={players}
            onSelect={(id) => {
              setMyPlayerId(id);
              setMyId(id);
            }}
            trigger={
              <button
                type="button"
                className="inline-flex min-h-9 items-center gap-[7px] text-[11px] text-muted-foreground transition-colors hover:text-win max-[760px]:text-[10px]"
              >
                录入人：{me?.name ?? "选择身份"}
                <ChevronRight className="size-[15px]" strokeWidth={1.65} />
              </button>
            }
          />
        </div>

        <section
          aria-label="双打比赛记分板"
          className="relative overflow-hidden rounded-[18px] bg-court p-[18px] text-[#f1f5e9] min-[761px]:p-[25px]"
        >
          <div className="mb-4 flex items-center justify-between min-[761px]:mb-[25px]">
            <span className="text-[10px] font-bold tracking-[2px] text-[#bcc8b7] max-[760px]:text-[9px]">
              DOUBLES / 双打
            </span>
            <small className="text-[10px] text-[#bac6b4]">FINAL SCORE</small>
          </div>
          <div className="relative grid grid-cols-2 gap-[22px] min-[761px]:gap-6">
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-y-0 left-1/2 border-l border-dashed border-[#75816866]"
            />
            {teamColumn(0)}
            {teamColumn(1)}
          </div>
          <div className="mt-[17px] flex items-center justify-between border-t border-[#5c685344] pt-3 min-[761px]:mt-[23px] min-[761px]:pt-[18px]">
            <button
              type="button"
              onClick={swapSides}
              className="inline-flex min-h-[35px] items-center gap-[7px] text-[11px] text-[#c3ccbb] transition-colors hover:text-[#f1f5e9]"
            >
              <ArrowLeftRight className="size-[15px]" strokeWidth={1.65} />
              交换两边
            </button>
            <small className="text-[10px] text-[#a3b297] max-[760px]:text-[9px]">
              点击数字可直接输入
            </small>
            <button
              type="button"
              onClick={clearSlots}
              className="inline-flex min-h-[35px] items-center gap-[7px] text-[11px] text-[#c3ccbb] transition-colors hover:text-[#f1f5e9]"
            >
              <RotateCcw className="size-[15px]" strokeWidth={1.65} />
              清空阵容
            </button>
          </div>
        </section>

        <Button
          size="lg"
          disabled={!canSubmit || submitting}
          onClick={() => {
            setError(null);
            setConfirmOpen(true);
          }}
          className="mt-[17px] h-[52px] w-full rounded-[9px] text-sm font-bold"
        >
          <Check strokeWidth={2} />
          {submitting ? "正在记录…" : "确认比分"}
          <ArrowRight strokeWidth={1.65} />
        </Button>
        <p className="mt-2.5 text-center text-[10px] text-muted-foreground">
          {recordHint()}
        </p>
      </div>

      {/* 桌面说明侧栏；手机隐藏 */}
      <aside className="hidden min-[761px]:flex min-[761px]:flex-col min-[761px]:gap-5">
        <section className="rounded-2xl border border-border bg-card p-[25px]">
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground">
            MATCH ENTRY
          </div>
          <h2 className="mt-[9px] text-lg font-bold text-card-foreground">
            录入说明
          </h2>
          {[
            ["01", "按实际阵容选人", "左右各一队，每队两位球员。"],
            ["02", "输入最终比分", "支持直接输入，也可用 + / − 调整。"],
            ["03", "确认后查看积分变化", "查看四位球员的积分变化。"],
          ].map(([step, title, desc]) => (
            <div key={step} className="mt-[23px] flex gap-3">
              <span className="grid size-6 shrink-0 place-items-center rounded-full border border-border font-num text-[10px] text-muted-foreground">
                {step}
              </span>
              <div>
                <h3 className="text-xs font-semibold text-card-foreground">
                  {title}
                </h3>
                <p className="mt-[3px] text-[11px] text-muted-foreground">
                  {desc}
                </p>
              </div>
            </div>
          ))}
        </section>
        <section className="rounded-2xl border border-border bg-card p-[25px]">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-sm font-semibold text-card-foreground">
              你的最近一场
            </h3>
            {me && (
              <Link
                href={`/players/${me.id}`}
                className="inline-flex items-center text-xs text-muted-foreground transition-colors hover:text-win"
                aria-label="查看我的全部战绩"
              >
                <ArrowRight className="size-[15px]" strokeWidth={1.65} />
              </Link>
            )}
          </div>
          {!myLatest ? (
            <p className="mt-4 text-xs text-muted-foreground">
              {me ? "最近还没有你的比赛。" : "选择录入人身份后显示最近一场。"}
            </p>
          ) : (
            <div className="mt-3">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs text-muted-foreground">
                  {myLatest.date.slice(5).replace("-", ".")} · 双打
                </span>
                <span
                  className={cn(
                    "inline-flex items-center rounded-[5px] px-[7px] py-1 text-[10px] font-bold",
                    myLatest.won
                      ? "bg-win-bg text-win"
                      : "bg-loss-bg text-loss"
                  )}
                >
                  {myLatest.won ? "胜" : "负"}
                </span>
              </div>
              <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-center gap-3">
                <span className="text-xs text-card-foreground">
                  {myLatest.team.join(" / ")}
                </span>
                <span className="font-num text-[23px] text-card-foreground">
                  {myLatest.scoreFor}
                  <span className="px-1.5 text-[13px] text-muted-foreground">
                    :
                  </span>
                  {myLatest.scoreAgainst}
                </span>
                <span className="text-right text-xs text-muted-foreground">
                  {myLatest.opponents.join(" / ")}
                </span>
              </div>
            </div>
          )}
        </section>
      </aside>

      {/* 阵容槽位 picker（含新增球员流程） */}
      <Dialog
        open={pickerSlot !== null}
        onOpenChange={(open) => {
          if (!open) setPickerSlot(null);
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              {adding
                ? "添加新球员"
                : `选择 ${pickerSlot !== null && pickerSlot < 2 ? "A" : "B"} 队 · ${
                    pickerSlot !== null ? (pickerSlot % 2) + 1 : ""
                  } 号球员`}
            </DialogTitle>
            <DialogDescription>
              {adding
                ? "添加后会落入当前阵容位置，不会改变你自己的身份。"
                : "已在其他位置的球员不能重复选择。"}
            </DialogDescription>
          </DialogHeader>

          {adding ? (
            <form onSubmit={handleAddPlayer} className="flex flex-col gap-3">
              <input
                autoFocus
                value={addName}
                onChange={(e) => {
                  setAddName(e.target.value);
                  setAddError(null);
                }}
                maxLength={20}
                placeholder="输入名字"
                className="h-12 w-full rounded-xl border border-border bg-background px-4 text-base outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              {addError && <p className="text-sm text-destructive">{addError}</p>}
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className="flex-1"
                  onClick={() => {
                    setAdding(false);
                    setAddName("");
                    setAddError(null);
                  }}
                >
                  返回
                </Button>
                <Button
                  type="submit"
                  className="flex-1"
                  disabled={addBusy || !addName.trim()}
                >
                  {addBusy ? "添加中…" : "确认添加"}
                </Button>
              </div>
            </form>
          ) : (
            <>
              <label className="flex h-11 items-center gap-[9px] rounded-[9px] border border-border bg-card px-[13px] focus-within:ring-2 focus-within:ring-ring">
                <Search
                  className="size-[17px] shrink-0 text-muted-foreground"
                  strokeWidth={1.65}
                />
                <input
                  autoFocus
                  value={pickerSearch}
                  onChange={(e) => setPickerSearch(e.target.value)}
                  placeholder="搜索球员姓名"
                  aria-label="搜索可选球员"
                  className="h-full w-full min-w-0 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
                />
              </label>
              <div className="grid max-h-[45dvh] grid-cols-2 gap-[9px] overflow-y-auto">
                {pickerVisible.length === 0 ? (
                  <div className="col-span-full py-[30px] text-center text-[13px] text-muted-foreground">
                    没有匹配的球员
                  </div>
                ) : (
                  pickerVisible.map((p) => {
                    const occupied =
                      slots.includes(p.id) && currentSlotValue !== p.id;
                    const selected = currentSlotValue === p.id;
                    return (
                      <button
                        key={p.id}
                        type="button"
                        disabled={occupied}
                        onClick={() => pickPlayer(p.id)}
                        aria-label={`${p.name}${occupied ? "，已在阵容中" : ""}`}
                        className={cn(
                          "flex min-w-0 items-center gap-2.5 rounded-[10px] border border-border p-3 text-left transition-colors disabled:opacity-40",
                          selected
                            ? "border-win bg-win-bg"
                            : "hover:border-win hover:bg-win-bg"
                        )}
                      >
                        <PlayerAvatar name={p.name} size="xs" />
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-semibold text-foreground">
                            {p.name}
                          </span>
                          <span className="block text-[10px] text-muted-foreground">
                            {occupied
                              ? "已在阵容中"
                              : selected
                                ? "当前选择"
                                : "可选择"}
                          </span>
                        </span>
                      </button>
                    );
                  })
                )}
              </div>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className="flex-1"
                  onClick={() => setAdding(true)}
                >
                  <UserPlus strokeWidth={1.65} />
                  新球员
                </Button>
                {currentSlotValue !== null && (
                  <Button
                    type="button"
                    variant="outline"
                    className="flex-1"
                    onClick={removeSlot}
                  >
                    移除此位置球员
                  </Button>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* 确认弹层：返回修改不写入 */}
      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>确认这场比赛</DialogTitle>
            <DialogDescription>
              检查双方阵容和最终比分。
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-4 rounded-[13px] bg-secondary p-[22px_16px] text-center">
            <div>
              {[slots[0], slots[1]].map((id) => (
                <div key={id} className="truncate text-sm font-semibold">
                  {id === null ? "—" : playerMap.get(id)?.name}
                </div>
              ))}
              <div className="mt-2 text-xs text-muted-foreground">A 队</div>
            </div>
            <span className="font-num text-[38px] text-foreground">
              {scoreA} : {scoreB}
            </span>
            <div>
              {[slots[2], slots[3]].map((id) => (
                <div key={id} className="truncate text-sm font-semibold">
                  {id === null ? "—" : playerMap.get(id)?.name}
                </div>
              ))}
              <div className="mt-2 text-xs text-muted-foreground">B 队</div>
            </div>
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <div className="flex gap-2.5">
            <Button
              variant="outline"
              className="flex-1"
              disabled={submitting}
              onClick={() => setConfirmOpen(false)}
            >
              返回修改
            </Button>
            <Button
              className="flex-1"
              disabled={submitting}
              onClick={handleSubmit}
            >
              {submitting ? "正在记录…" : "确认记分"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* 成功弹层：四人前后积分与变化 */}
      <Dialog
        open={result !== null}
        onOpenChange={(open) => {
          if (!open) setResult(null);
        }}
      >
        <DialogContent className="sm:max-w-md">
          <div className="mx-auto grid size-[54px] place-items-center rounded-full bg-primary text-primary-foreground">
            <Check className="size-6" strokeWidth={2} />
          </div>
          <DialogHeader>
            <DialogTitle className="text-center text-xl">
              比赛已记录
            </DialogTitle>
            <DialogDescription className="text-center">
              四位球员的 ELO 已更新
            </DialogDescription>
          </DialogHeader>
          {result && <EloDeltaCard players={result.players} />}
          <div className="flex gap-2.5">
            <Button
              variant="outline"
              className="flex-1"
              onClick={() => {
                if (result) router.push(`/players/${result.enteredBy}`);
              }}
            >
              查看我的数据
            </Button>
            <Button className="flex-1" onClick={nextRecord}>
              再记一场
              <ArrowRight strokeWidth={1.65} />
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* 轻量 toast（无新依赖） */}
      <div
        role="status"
        aria-live="polite"
        className={cn(
          "fixed bottom-[91px] left-1/2 z-[80] max-w-[calc(100%-32px)] -translate-x-1/2 rounded-[10px] bg-foreground px-[19px] py-[11px] text-xs text-background shadow-lg transition-all duration-200 min-[761px]:bottom-[30px]",
          toast
            ? "translate-y-0 opacity-100"
            : "pointer-events-none translate-y-[15px] opacity-0"
        )}
      >
        {toast}
      </div>
    </div>
  );
}
