"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowRight, Plus } from "lucide-react";
import { getMyPlayerId } from "@/lib/identity";
import { IdentityPicker } from "@/components/identity-picker";

export interface OverviewStat {
  elo: number;
  rank: number;
  /** 无比赛时为 null，显示「—」 */
  winRate: number | null;
}

interface OverviewSummaryProps {
  players: { id: number; name: string }[];
  stats: Record<number, OverviewStat>;
  /** 本周全员比赛场数（公共数据，不随身份变化） */
  weekMatchCount: number;
}

/** 深底个人摘要卡（bg-court，两主题恒深色，卡内浅色文字沿用 mock welcome-card 用色） */
export function OverviewSummary({
  players,
  stats,
  weekMatchCount,
}: OverviewSummaryProps) {
  const [myId, setMyId] = React.useState<number | null>(null);
  const [pickerMounted, setPickerMounted] = React.useState(false);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const me = myId === null ? undefined : players.find((p) => p.id === myId);
  const stat = me ? stats[me.id] : undefined;

  const statValue =
    "font-num text-[34px] leading-[1.4] font-[550] min-[761px]:text-[37px]";
  const statLabel = "text-[9px] text-[#c1cabd] min-[761px]:text-[10px]";
  const statUnit = "ml-1 text-sm font-normal";

  return (
    <section className="relative flex min-h-[225px] flex-col items-start overflow-hidden rounded-2xl bg-court p-[22px] text-[#f5f6ed] min-[761px]:min-h-[244px] min-[761px]:px-[30px] min-[761px]:py-7">
      {/* 细线球场装饰（mock .court-art） */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -right-[29px] -bottom-16 h-[270px] w-[255px] rotate-[-23deg] border border-[#d3f36b36]"
      >
        <div className="absolute inset-[19px] border border-[#d3f36b36]" />
        <div className="absolute inset-x-0 top-1/2 h-12 border-t border-b border-t-[#d3f36b70] border-b-[#d3f36b36]" />
        <div className="absolute left-1/2 h-full border-l border-[#d3f36b36]" />
      </div>

      <div className="relative flex w-full items-center justify-between gap-3">
        <h2 className="text-[23px] font-semibold">
          {me ? me.name : "尚未选择身份"}
        </h2>
        {stat && (
          <span className="text-[11px] text-[#c4d0bd]">
            俱乐部 #{stat.rank}
          </span>
        )}
      </div>

      <div className="relative mt-6 mb-6 grid w-full grid-cols-[1fr_1fr_1.2fr] gap-4 min-[761px]:gap-[26px]">
        <div className="flex flex-col">
          <span className={statLabel}>当前 ELO</span>
          <strong className={statValue}>{stat ? stat.elo : "—"}</strong>
        </div>
        <div className="flex flex-col">
          <span className={statLabel}>生涯胜率</span>
          <strong className={statValue}>
            {stat && stat.winRate !== null ? (
              <>
                {stat.winRate}
                <small className={statUnit}>%</small>
              </>
            ) : (
              "—"
            )}
          </strong>
        </div>
        <div className="flex flex-col">
          <span className={statLabel}>本周比赛 · 全员</span>
          <strong className={statValue}>
            {weekMatchCount}
            <small className={statUnit}>场</small>
          </strong>
        </div>
      </div>

      <div className="relative mt-auto flex flex-wrap items-center gap-2">
        {me ? (
          <>
            <Link
              href="/record"
              className="inline-flex min-h-11 items-center justify-center gap-2 rounded-[9px] bg-primary px-[17px] text-xs font-bold text-primary-foreground transition-colors hover:bg-primary/90"
            >
              <Plus className="size-4" strokeWidth={1.65} />
              记一场比赛
            </Link>
            <Link
              href={`/players/${me.id}`}
              className="inline-flex items-center gap-[7px] p-2.5 text-[11px] text-[#d0d8c9] transition-colors hover:text-[#f5f6ed]"
            >
              我的数据
              <ArrowRight className="size-4" strokeWidth={1.65} />
            </Link>
          </>
        ) : (
          <button
            type="button"
            onClick={() => setPickerMounted(true)}
            className="inline-flex min-h-11 items-center justify-center gap-2 rounded-[9px] bg-primary px-[17px] text-xs font-bold text-primary-foreground transition-colors hover:bg-primary/90"
          >
            选择身份
          </button>
        )}
      </div>

      {pickerMounted && (
        <IdentityPicker players={players} onSelect={(id) => setMyId(id)} />
      )}
    </section>
  );
}
