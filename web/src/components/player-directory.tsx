"use client";

import * as React from "react";
import Link from "next/link";
import { Search } from "lucide-react";
import { PlayerAvatar } from "@/components/player-avatar";
import { getMyPlayerId } from "@/lib/identity";
import { cn } from "@/lib/utils";

export interface PlayerDirectoryEntry {
  id: number;
  name: string;
  elo: number;
  /** 恒按 ELO 排定的名次（同分按 id 升序），切换排序不改变 */
  rank: number;
  total: number;
  winRate: number;
  /** 最近 9 个 ELO 快照（升序） */
  sparkline: number[];
}

type SortKey = "elo" | "winRate" | "total";

const SORTS: { key: SortKey; label: string }[] = [
  { key: "elo", label: "按 ELO" },
  { key: "winRate", label: "按胜率" },
  { key: "total", label: "按场次" },
];

function Sparkline({ points, name }: { points: number[]; name: string }) {
  const min = Math.min(...points) - 5;
  const max = Math.max(...points) + 5;
  const span = max - min || 1;
  const coords = points
    .map(
      (n, i) =>
        `${((i / (points.length - 1)) * 95).toFixed(1)},${(
          32 -
          ((n - min) / span) * 29
        ).toFixed(1)}`
    )
    .join(" ");
  return (
    <svg
      viewBox="0 0 95 35"
      role="img"
      aria-label={`${name}最近积分走势`}
      className="h-[28px] w-[56px] min-[761px]:h-[35px] min-[761px]:w-[95px]"
    >
      <polyline
        points={coords}
        fill="none"
        stroke="var(--chart)"
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function PlayerDirectory({ entries }: { entries: PlayerDirectoryEntry[] }) {
  const [search, setSearch] = React.useState("");
  const [sort, setSort] = React.useState<SortKey>("elo");
  const [myId, setMyId] = React.useState<number | null>(null);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const visible = React.useMemo(() => {
    const term = search.trim();
    const filtered = term
      ? entries.filter((p) => p.name.includes(term))
      : entries;
    // entries 已按 ELO 降序（同分 id 升序）；稳定排序保持平局口径
    return [...filtered].sort((a, b) => b[sort] - a[sort]);
  }, [entries, search, sort]);

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-[18px]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <label className="flex h-11 w-full items-center gap-[9px] rounded-[9px] border border-border bg-card px-[13px] focus-within:ring-2 focus-within:ring-ring min-[761px]:w-[270px]">
          <Search
            className="size-[17px] shrink-0 text-muted-foreground"
            strokeWidth={1.65}
          />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索球员姓名"
            aria-label="搜索球员姓名"
            className="h-full w-full min-w-0 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground"
          />
        </label>
        <div
          className="inline-flex gap-[2px] rounded-[7px] border border-border p-[3px]"
          role="group"
          aria-label="球员排序"
        >
          {SORTS.map((s) => (
            <button
              key={s.key}
              type="button"
              aria-pressed={sort === s.key}
              onClick={() => setSort(s.key)}
              className={cn(
                "min-h-[30px] rounded px-2.5 text-[10px] whitespace-nowrap transition-colors max-[760px]:min-h-9 max-[760px]:px-2",
                sort === s.key
                  ? "bg-secondary font-bold text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 min-[761px]:gap-5 min-[1191px]:grid-cols-3">
        {visible.length === 0 ? (
          <div className="col-span-full py-[45px] text-center text-[13px] text-muted-foreground">
            没有找到这位球员，试试其他名字。
          </div>
        ) : (
          visible.map((p) => (
            <Link
              key={p.id}
              href={`/players/${p.id}`}
              aria-label={`查看${p.name}的详细分析`}
              className="relative block rounded-2xl border border-border bg-card p-[18px] transition-[transform,border-color] duration-150 hover:-translate-y-[3px] hover:border-win min-[761px]:p-6"
            >
              <PlayerAvatar
                name={p.name}
                size="sm"
                className="mb-4 size-12 text-lg"
              />
              <span className="absolute top-[18px] right-4 font-num text-xs text-muted-foreground min-[761px]:top-[22px] min-[761px]:right-[22px]">
                #{String(p.rank).padStart(2, "0")}
              </span>
              <h2 className="text-[19px] font-semibold tracking-[-0.3px] text-card-foreground">
                {p.name}
                {p.id === myId && (
                  <span className="ml-1.5 inline-flex items-center rounded-[5px] bg-secondary px-[7px] py-0.5 align-middle text-[10px] font-bold text-muted-foreground">
                    我
                  </span>
                )}
              </h2>
              <div className="mt-0.5 text-xs text-muted-foreground">
                {p.total} 场比赛 · {p.total > 0 ? `${p.winRate}%` : "—"} 胜率
              </div>
              <div className="mt-[18px] flex items-end justify-between gap-2 border-t border-border pt-[15px]">
                <div>
                  <div className="text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
                    ELO RATING
                  </div>
                  <div className="mt-0.5 font-num text-[25px] leading-[1.1] text-card-foreground min-[761px]:text-[30px]">
                    {p.elo}
                  </div>
                </div>
                {p.sparkline.length >= 2 && (
                  <Sparkline points={p.sparkline} name={p.name} />
                )}
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}
