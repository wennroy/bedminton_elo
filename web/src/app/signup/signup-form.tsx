"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Calendar,
  Check,
  ChevronsUpDown,
} from "lucide-react";
import { getMyPlayerId } from "@/lib/identity";
import { getAdminKey } from "@/lib/admin";
import { Button } from "@/components/ui/button";
import { IdentityPicker } from "@/components/identity-picker";
import { PlayerAvatar } from "@/components/player-avatar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

interface SignupRow {
  playerId: number;
  name: string;
  partySize: number;
}

interface SignupFormProps {
  sessionDate: string;
  signups: SignupRow[];
  summary: { count: number; totalPeople: number };
  /** 全部球员：身份选择弹层与当前身份名称 */
  players: { id: number; name: string }[];
}

export function SignupForm({ sessionDate, signups, summary, players }: SignupFormProps) {
  const router = useRouter();
  const [myId, setMyId] = React.useState<number | null>(null);
  const [isAdmin, setIsAdmin] = React.useState(false);
  const [partySize, setPartySize] = React.useState<1 | 2>(1);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [cancelOpen, setCancelOpen] = React.useState(false);

  // IdentityPicker 按需挂载：未选身份时点「选择身份」会自动弹开
  const [pickerMounted, setPickerMounted] = React.useState(false);
  const pickerTriggerRef = React.useRef<HTMLButtonElement>(null);
  const clickTriggerAfterMount = React.useRef(false);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
    setIsAdmin(!!getAdminKey());
  }, []);

  const mySignup = signups.find((s) => s.playerId === myId);
  const myPartySize = mySignup?.partySize;

  // 同步已报名的人数到本地切换状态；切换身份后按该成员已有报名恢复，未报名者默认 1 人
  React.useEffect(() => {
    setPartySize(myPartySize === 2 ? 2 : 1);
  }, [myPartySize]);

  const openIdentityPicker = () => {
    if (pickerMounted) {
      pickerTriggerRef.current?.click();
      return;
    }
    clickTriggerAfterMount.current = getMyPlayerId() !== null;
    setPickerMounted(true);
  };

  React.useEffect(() => {
    if (pickerMounted && clickTriggerAfterMount.current) {
      clickTriggerAfterMount.current = false;
      pickerTriggerRef.current?.click();
    }
  }, [pickerMounted]);

  async function callApi(
    method: "POST" | "DELETE",
    playerId: number,
    size?: 1 | 2
  ) {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/signups", {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          method === "POST"
            ? { playerId, partySize: size ?? partySize }
            : { playerId }
        ),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || "操作失败");
      }
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "操作失败");
    } finally {
      setBusy(false);
    }
  }

  function changePartySize(size: 1 | 2) {
    setPartySize(size);
    // 已报名时切换人数 = 直接 upsert 更新
    if (mySignup && myId !== null && size !== mySignup.partySize) {
      callApi("POST", myId, size);
    }
  }

  const [year, month, day] = sessionDate.split("-").map(Number);
  const guests = summary.totalPeople - summary.count;
  const meName =
    myId === null
      ? null
      : (players.find((p) => p.id === myId)?.name ?? null);

  const pillClass =
    "inline-flex items-center rounded-[5px] px-[7px] py-1 text-[10px] font-bold";

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-[18px]">
      {/* 场次横幅 */}
      <section className="flex items-center justify-between gap-[30px] rounded-2xl border border-border bg-card p-7 max-[760px]:flex-col max-[760px]:items-stretch max-[760px]:gap-[21px] max-[760px]:p-5">
        <div className="flex items-center gap-[22px] max-[760px]:gap-[17px]">
          <div className="flex shrink-0 flex-col items-center rounded-[10px] bg-court px-[22px] py-3 max-[760px]:px-[18px] max-[760px]:py-2.5">
            <span className="text-[9px] tracking-[1px] text-[#c2ccb9]">
              {year} / {String(month).padStart(2, "0")}
            </span>
            <strong className="font-num text-[43px] leading-[1.2] font-semibold text-primary min-[761px]:text-5xl">
              {String(day).padStart(2, "0")}
            </strong>
            <span className="text-[9px] tracking-[1px] text-[#c2ccb9]">
              WED · 周三
            </span>
          </div>
          <div>
            <h2 className="text-xl font-bold text-card-foreground min-[761px]:text-[23px]">
              周三羽毛球局
            </h2>
            <div className="mt-2.5 flex items-center gap-[7px] text-xs text-muted-foreground max-[760px]:text-[10px]">
              <Calendar className="size-[15px]" strokeWidth={1.65} />
              {month} 月 {day} 日 · 18:00–20:00
            </div>
            <p className="mt-[5px] text-xs text-muted-foreground max-[760px]:text-[10px]">
              每周三固定场次
            </p>
          </div>
        </div>
        <div className="flex gap-6 max-[760px]:justify-between max-[760px]:gap-0 max-[760px]:border-t max-[760px]:border-border max-[760px]:pt-[15px]">
          {[
            [summary.totalPeople, "参加人数"],
            [summary.count, "报名成员"],
            [guests, "随行小伙伴"],
          ].map(([value, label], i) => (
            <div
              key={label}
              className={cn(
                "flex flex-col items-center border-l border-border pl-6 max-[760px]:w-1/3 max-[760px]:pl-0",
                i === 0 && "max-[760px]:border-l-0"
              )}
            >
              <span className="font-num text-[30px] leading-[1.4] font-medium text-card-foreground min-[761px]:text-[37px]">
                {value}
              </span>
              <span className="text-[10px] text-muted-foreground max-[760px]:text-[9px]">
                {label}
              </span>
            </div>
          ))}
        </div>
      </section>

      <div className="grid items-start gap-6 min-[761px]:grid-cols-[minmax(0,1.6fr)_minmax(290px,1fr)] max-[760px]:flex max-[760px]:flex-col max-[760px]:gap-[18px]">
        {/* 报名名单（按报名先后） */}
        <section className="rounded-2xl border border-border bg-card px-[26px] pb-[9px] pt-6 max-[760px]:px-5 max-[760px]:pt-5">
          <div className="mb-[22px] flex items-center justify-between gap-3.5 max-[760px]:mb-[18px]">
            <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
              报名名单
            </h2>
            <span className="text-xs text-muted-foreground">
              {summary.count} 位成员 · 共 {summary.totalPeople} 人
            </span>
          </div>
          <div className="flex justify-between pb-3 text-[10px] text-muted-foreground">
            <span>成员</span>
            <span>参加人数</span>
          </div>
          {signups.length === 0 ? (
            <div className="border-t border-border py-[45px] text-center text-[13px] text-muted-foreground">
              本期还没有人报名。
            </div>
          ) : (
            <div>
              {signups.map((s, i) => (
                <div
                  key={s.playerId}
                  className="flex items-center gap-4 border-t border-border py-[18px] max-[760px]:py-[15px]"
                >
                  <span className="w-[19px] shrink-0 font-num text-xs text-muted-foreground">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <Link
                    href={`/players/${s.playerId}`}
                    className="flex min-w-0 items-center gap-[11px] transition-colors hover:text-win"
                  >
                    <PlayerAvatar name={s.name} size="xs" />
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-semibold text-card-foreground">
                        {s.name}
                        {s.playerId === myId && (
                          <small className="ml-1 text-[9px] font-normal text-muted-foreground">
                            我
                          </small>
                        )}
                      </span>
                      <span className="block text-[10px] text-muted-foreground">
                        {s.partySize === 2 ? "携带 1 位小伙伴" : "本人参加"}
                      </span>
                    </span>
                  </Link>
                  <span className="ml-auto font-num text-[22px] text-card-foreground">
                    {s.partySize}
                    <small className="text-[11px] text-muted-foreground">
                      {" "}
                      人
                    </small>
                  </span>
                  {isAdmin && (
                    <Button
                      variant="destructive"
                      size="xs"
                      disabled={busy}
                      onClick={() => callApi("DELETE", s.playerId)}
                    >
                      移除
                    </Button>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>

        {/* 操作栏：手机置于名单前 */}
        <aside className="rounded-2xl border border-border bg-card p-[25px] max-[760px]:order-[-1] max-[760px]:p-5">
          <div className="mb-[22px] flex items-center justify-between gap-3.5">
            <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
              我的报名
            </h2>
            <span
              className={cn(
                pillClass,
                mySignup
                  ? "bg-win-bg text-win"
                  : "bg-secondary text-muted-foreground"
              )}
            >
              {mySignup ? "已报名" : "未报名"}
            </span>
          </div>

          <div className="flex items-center justify-between gap-2 border-t border-border pt-[19px]">
            <span className="flex min-w-0 items-center gap-[11px]">
              {meName || myId !== null ? (
                <PlayerAvatar name={meName ?? "?"} size="xs" />
              ) : (
                <span className="grid size-8 place-items-center rounded-full bg-muted text-xs text-muted-foreground">
                  ?
                </span>
              )}
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold text-card-foreground">
                  {meName ?? (myId !== null ? `#${myId}` : "未选择身份")}
                </span>
                <span className="block text-[10px] text-muted-foreground">
                  当前报名身份
                </span>
              </span>
            </span>
            <button
              type="button"
              onClick={openIdentityPicker}
              className="inline-flex min-h-9 shrink-0 items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-win"
            >
              {myId === null ? "选择身份" : "切换"}
              <ChevronsUpDown className="size-[15px]" strokeWidth={1.65} />
            </button>
          </div>

          <div className="mt-[23px] text-[11px] text-muted-foreground">
            参加人数
          </div>
          <div className="mt-2.5 grid grid-cols-2 gap-[9px] min-[761px]:grid-cols-1">
            {([1, 2] as const).map((n) => {
              const selected = partySize === n;
              return (
                <button
                  key={n}
                  type="button"
                  aria-pressed={selected}
                  disabled={busy || myId === null}
                  onClick={() => changePartySize(n)}
                  className={cn(
                    "flex min-h-[70px] items-center gap-[13px] rounded-[9px] border border-border p-3.5 text-left transition-colors disabled:opacity-50 max-[760px]:gap-[9px] max-[760px]:p-[12px_10px]",
                    selected && "border-win bg-win-bg"
                  )}
                >
                  <span
                    className={cn(
                      "grid size-[18px] shrink-0 place-items-center rounded-full border",
                      selected
                        ? "border-win bg-win text-card"
                        : "border-muted-foreground"
                    )}
                  >
                    {selected && (
                      <Check className="size-[13px]" strokeWidth={2.5} />
                    )}
                  </span>
                  <span>
                    <strong className="block text-xs font-semibold text-card-foreground max-[760px]:text-[11px]">
                      {n === 1 ? "自己来" : "带一位小伙伴"}
                    </strong>
                    <small className="mt-0.5 block text-[10px] text-muted-foreground max-[760px]:text-[9px]">
                      共 {n} 人
                    </small>
                  </span>
                </button>
              );
            })}
          </div>

          {!mySignup ? (
            <Button
              className="mt-[18px] h-[47px] w-full rounded-[9px] text-xs font-bold"
              disabled={busy || myId === null}
              onClick={() => myId !== null && callApi("POST", myId)}
            >
              {busy ? "正在提交…" : "确认报名"}
              {!busy && <ArrowRight strokeWidth={1.65} />}
            </Button>
          ) : (
            <>
              <div className="mt-5 flex items-center gap-2 text-xs font-medium text-win">
                <Check className="size-4" strokeWidth={2} />
                已报名 · {mySignup.partySize} 人参加
              </div>
              <Button
                variant="outline"
                className="mt-3 h-[47px] w-full rounded-[9px] text-xs font-bold"
                disabled={busy}
                onClick={() => setCancelOpen(true)}
              >
                取消报名
              </Button>
              <p className="mt-2 text-[10px] text-muted-foreground">
                切换人数会自动更新报名
              </p>
            </>
          )}

          {myId === null && !mySignup && (
            <p className="mt-2 text-[10px] text-muted-foreground">
              选择身份后即可报名；名单对所有人可见。
            </p>
          )}
          {error && (
            <p role="alert" className="mt-2 text-sm text-destructive">
              {error}
            </p>
          )}

          <div className="mt-6 border-t border-border pt-[18px] max-[760px]:mt-5 max-[760px]:pt-[15px]">
            <h3 className="text-[11px] font-medium text-card-foreground">
              报名说明
            </h3>
            <p className="mt-[7px] text-[10px] leading-[1.9] text-muted-foreground">
              每位成员最多带 1 位小伙伴。
              <br />
              本期报名于周三 20:00 切换至下周。
            </p>
          </div>
        </aside>
      </div>

      {/* 取消确认：保留报名 / 确认取消 */}
      <Dialog open={cancelOpen} onOpenChange={setCancelOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>取消本周报名？</DialogTitle>
            <DialogDescription>
              将移除{meName ?? "当前成员"}及随行小伙伴的报名。之后可以重新报名。
            </DialogDescription>
          </DialogHeader>
          <div className="flex gap-2.5">
            <Button
              variant="outline"
              className="flex-1"
              disabled={busy}
              onClick={() => setCancelOpen(false)}
            >
              保留报名
            </Button>
            <Button
              className="flex-1"
              disabled={busy}
              onClick={async () => {
                if (myId === null) return;
                await callApi("DELETE", myId);
                setCancelOpen(false);
              }}
            >
              确认取消
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {pickerMounted && (
        <IdentityPicker
          players={players}
          onSelect={() => {
            setMyId(getMyPlayerId());
            router.refresh();
          }}
          trigger={
            <button
              ref={pickerTriggerRef}
              type="button"
              className="hidden"
              tabIndex={-1}
              aria-hidden="true"
            />
          }
        />
      )}
    </div>
  );
}
