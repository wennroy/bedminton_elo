"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { getMyPlayerId } from "@/lib/identity";
import { getAdminKey } from "@/lib/admin";
import { Button } from "@/components/ui/button";
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
}

export function SignupForm({ signups, summary }: SignupFormProps) {
  const router = useRouter();
  const [myId, setMyId] = React.useState<number | null>(null);
  const [isAdmin, setIsAdmin] = React.useState(false);
  const [partySize, setPartySize] = React.useState<1 | 2>(1);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
    setIsAdmin(!!getAdminKey());
    setMounted(true);
  }, []);

  const mySignup = signups.find((s) => s.playerId === myId);
  const myPartySize = mySignup?.partySize;

  // 同步已报名的人数到本地切换状态
  React.useEffect(() => {
    if (myPartySize) setPartySize(myPartySize === 2 ? 2 : 1);
  }, [myPartySize]);

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
          method === "POST" ? { playerId, partySize: size ?? partySize } : { playerId }
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

  return (
    <div className="flex flex-col gap-4">
      <section className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-sm font-medium text-muted-foreground">
            报名名单
          </h2>
          <span className="text-xs text-muted-foreground">
            已报 {summary.count} 人 · 含小伙伴共 {summary.totalPeople} 人
          </span>
        </div>
        {signups.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            还没有人报名，来抢沙发～
          </p>
        ) : (
          <ul className="flex flex-col divide-y divide-border">
            {signups.map((s, i) => (
              <li
                key={s.playerId}
                className="flex items-center gap-2 py-2 text-sm"
              >
                <span className="w-6 text-xs text-muted-foreground">
                  {i + 1}
                </span>
                <span className="flex-1 font-medium text-card-foreground">
                  {s.name}
                  {s.partySize > 1 && (
                    <span className="ml-1 text-muted-foreground">
                      ×{s.partySize}
                    </span>
                  )}
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
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        {!mounted ? (
          <p className="text-sm text-muted-foreground">加载中…</p>
        ) : myId === null ? (
          <p className="text-sm text-muted-foreground">
            请先到
            <Link href="/me" className="mx-1 text-primary underline">
              「我的」
            </Link>
            页选择身份后再报名
          </p>
        ) : (
          <div className="flex flex-col gap-3">
            {mySignup && (
              <p className="text-sm font-medium text-card-foreground">
                已报名
                {mySignup.partySize > 1 && ` ×${mySignup.partySize}`}
              </p>
            )}
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">人数</span>
              {([1, 2] as const).map((size) => (
                <button
                  key={size}
                  disabled={busy}
                  onClick={() => changePartySize(size)}
                  className={cn(
                    "h-8 rounded-lg border px-3 text-sm font-medium transition-colors",
                    partySize === size
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border bg-background text-foreground hover:bg-muted"
                  )}
                >
                  {size === 1 ? "1（自己来）" : "2（带小伙伴）"}
                </button>
              ))}
            </div>
            {!mySignup ? (
              <Button
                className="w-full"
                disabled={busy}
                onClick={() => myId !== null && callApi("POST", myId)}
              >
                报名
              </Button>
            ) : (
              <>
                <Button
                  variant="outline"
                  className="w-full"
                  disabled={busy}
                  onClick={() => myId !== null && callApi("DELETE", myId)}
                >
                  取消报名
                </Button>
                <p className="text-xs text-muted-foreground">
                  切换人数会自动更新报名
                </p>
              </>
            )}
          </div>
        )}
        {error && <p className="mt-2 text-sm text-destructive">{error}</p>}
      </section>
    </div>
  );
}
