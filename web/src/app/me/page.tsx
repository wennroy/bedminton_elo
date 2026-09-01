"use client";

import * as React from "react";
import Link from "next/link";
import { getMyPlayerId, setMyPlayerId } from "@/lib/identity";
import { IdentityPicker } from "@/components/identity-picker";
import { PlayerAvatar } from "@/components/player-avatar";
import { Shield, ChevronRight, Sparkles } from "lucide-react";

interface Player {
  id: number;
  name: string;
}

export default function MePage() {
  const [players, setPlayers] = React.useState<Player[]>([]);
  const [myId, setMyId] = React.useState<number | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
    fetch("/api/players")
      .then((res) => {
        if (!res.ok) throw new Error("加载球员失败");
        return res.json();
      })
      .then((data: Player[]) => {
        setPlayers(data);
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : "加载失败");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  function handleSelect(id: number) {
    setMyPlayerId(id);
    setMyId(id);
    // The selection may be a just-added player not yet in our local list.
    fetch("/api/players")
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data: Player[]) => setPlayers(data))
      .catch(() => {});
  }

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <h1 className="mb-6 text-xl font-bold text-foreground">我的</h1>

      <section className="mb-6 flex flex-col gap-3 rounded-2xl border border-border bg-card p-4 shadow-sm">
        <h2 className="text-sm font-medium text-muted-foreground">当前身份</h2>
        {loading ? (
          <div className="text-sm text-muted-foreground">加载中…</div>
        ) : error ? (
          <div className="text-sm text-destructive">{error}</div>
        ) : (
          <>
            <IdentityPicker
              players={players}
              onSelect={handleSelect}
              trigger={
                <button className="flex items-center gap-3 rounded-xl border border-border bg-background p-3 text-left transition-colors hover:bg-muted">
                  <PlayerAvatar
                    name={players.find((p) => p.id === myId)?.name || "?"}
                    size="sm"
                  />
                  <div className="flex flex-1 flex-col">
                    <span className="font-medium text-card-foreground">
                      {players.find((p) => p.id === myId)?.name || "未选择"}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      点这里更换身份或添加新球员
                    </span>
                  </div>
                  <ChevronRight className="size-4 text-muted-foreground" />
                </button>
              }
            />
          </>
        )}
      </section>

      <section className="mb-6 flex flex-col gap-3">
        <h2 className="text-sm font-medium text-muted-foreground">关于</h2>
        <Link
          href="/changelog"
          className="flex items-center justify-between rounded-2xl border border-border bg-card p-4 shadow-sm transition-colors hover:bg-muted/30"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Sparkles className="size-5" />
            </div>
            <div className="flex flex-col">
              <span className="font-medium text-card-foreground">更新日志</span>
              <span className="text-xs text-muted-foreground">
                看看有什么新功能
              </span>
            </div>
          </div>
          <ChevronRight className="size-5 text-muted-foreground" />
        </Link>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-medium text-muted-foreground">管理</h2>
        <Link
          href="/admin"
          className="flex items-center justify-between rounded-2xl border border-border bg-card p-4 shadow-sm transition-colors hover:bg-muted/30"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Shield className="size-5" />
            </div>
            <div className="flex flex-col">
              <span className="font-medium text-card-foreground">管理员登录</span>
              <span className="text-xs text-muted-foreground">比赛/球员管理</span>
            </div>
          </div>
          <ChevronRight className="size-5 text-muted-foreground" />
        </Link>
      </section>
    </main>
  );
}
