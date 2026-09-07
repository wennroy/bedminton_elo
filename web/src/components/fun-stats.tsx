"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Search, Users } from "lucide-react";
import { PlayerAvatar } from "@/components/player-avatar";
import { CollapsibleSection } from "@/components/collapsible-section";
import { getMyPlayerId } from "@/lib/identity";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

/**
 * 球员分析页辅助组件：
 * ProfileHeader —— 头部（大头像 / 排名 / 切换球员弹层）；
 * MoreMetrics —— 「更多指标」展开区（TrueSkill、最长连胜、峰值，默认折叠）。
 */

interface SwitcherPlayer {
  id: number;
  name: string;
  elo: number;
}

export function ProfileHeader({
  id,
  name,
  rank,
  players,
}: {
  id: number;
  name: string;
  rank: number;
  players: SwitcherPlayer[];
}) {
  const router = useRouter();
  const [myId, setMyId] = React.useState<number | null>(null);
  const [open, setOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const term = search.trim();
  const visible = term
    ? players.filter((p) => p.name.includes(term))
    : players;

  return (
    <div className="flex items-center justify-between gap-4 max-[760px]:items-start">
      <div className="flex min-w-0 items-center gap-5 max-[760px]:gap-3.5">
        <PlayerAvatar
          name={name}
          size="lg"
          className="size-[62px] rounded-[19px] text-2xl min-[761px]:size-[76px] min-[761px]:rounded-[23px] min-[761px]:text-[27px]"
        />
        <div className="min-w-0">
          <div className="text-[9px] font-bold tracking-[1px] text-muted-foreground min-[761px]:tracking-[2px]">
            PLAYER {String(id).padStart(2, "0")}
          </div>
          <h1 className="my-1 text-[27px] font-bold tracking-[-0.8px] text-foreground min-[761px]:text-[31px]">
            {name}
            {id === myId && (
              <span className="ml-2 inline-flex items-center rounded-[5px] bg-secondary px-[7px] py-0.5 align-middle text-[10px] font-bold text-muted-foreground">
                我
              </span>
            )}
          </h1>
          <div className="text-[11px] text-muted-foreground max-[760px]:text-[10px]">
            俱乐部排名{" "}
            <strong className="font-semibold text-foreground">#{rank}</strong>
            <span className="px-2">·</span>双打球员
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={() => setOpen(true)}
        className="inline-flex min-h-[37px] shrink-0 items-center gap-2 rounded-[9px] border border-border bg-card px-3 text-[11px] font-bold text-card-foreground transition-colors hover:bg-secondary min-[761px]:min-h-11 min-[761px]:px-[17px] min-[761px]:text-xs"
      >
        <Users className="size-[15px]" strokeWidth={1.65} />
        切换球员
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>选择球员</DialogTitle>
            <DialogDescription>
              选择球员以查看详细数据。
            </DialogDescription>
          </DialogHeader>
          <label className="flex h-11 items-center gap-[9px] rounded-[9px] border border-border bg-card px-[13px] focus-within:ring-2 focus-within:ring-ring">
            <Search
              className="size-[17px] shrink-0 text-muted-foreground"
              strokeWidth={1.65}
            />
            <input
              autoFocus
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索球员姓名"
              aria-label="搜索可选球员"
              className="h-full w-full min-w-0 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
            />
          </label>
          <div className="grid max-h-[50dvh] grid-cols-2 gap-[9px] overflow-y-auto">
            {visible.length === 0 ? (
              <div className="col-span-full py-[30px] text-center text-[13px] text-muted-foreground">
                没有匹配的球员
              </div>
            ) : (
              visible.map((p) => {
                const selected = p.id === id;
                return (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => {
                      if (!selected) {
                        setOpen(false);
                        router.push(`/players/${p.id}`);
                      }
                    }}
                    aria-label={`${p.name}${selected ? "，当前选择" : ""}`}
                    className={cn(
                      "flex min-w-0 items-center gap-2.5 rounded-[10px] border border-border p-3 text-left transition-colors",
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
                        {selected ? "当前选择" : `ELO ${p.elo}`}
                      </span>
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/** 「更多指标」展开区：TrueSkill μ/σ/区间、最长连胜、峰值 ELO 及日期，默认折叠 */
export function MoreMetrics({
  mu,
  sigma,
  longestWinStreak,
  peakElo,
  peakEloDate,
}: {
  mu: number;
  sigma: number;
  longestWinStreak: number;
  peakElo: number;
  peakEloDate: string | null;
}) {
  const items: { label: string; value: string; sub?: string }[] = [
    { label: "TrueSkill μ", value: String(Math.round(mu)) },
    { label: "TrueSkill σ", value: sigma.toFixed(1) },
    {
      label: "TrueSkill 区间",
      value: `${Math.round(mu - 3 * sigma)} – ${Math.round(mu + 3 * sigma)}`,
    },
    { label: "最长连胜", value: `${longestWinStreak} 连胜` },
    {
      label: "峰值 ELO",
      value: String(peakElo),
      sub: peakEloDate ? `${peakEloDate} 达成` : undefined,
    },
  ];

  return (
    <CollapsibleSection title="更多指标" defaultOpen={false}>
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 pt-1 min-[761px]:grid-cols-3">
        {items.map((item) => (
          <div key={item.label}>
            <div className="text-[11px] text-muted-foreground">{item.label}</div>
            <div className="mt-0.5 font-num text-xl text-card-foreground">
              {item.value}
            </div>
            {item.sub && (
              <div className="text-[10px] text-muted-foreground">{item.sub}</div>
            )}
          </div>
        ))}
      </div>
    </CollapsibleSection>
  );
}
