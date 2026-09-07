"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { getMyPlayerId } from "@/lib/identity";
import { cn } from "@/lib/utils";

interface SignupCardProps {
  /** 当前开放报名的场次日期 YYYY-MM-DD（恒为周三） */
  sessionDate: string;
  /** 参加总人数（含随行小伙伴） */
  totalPeople: number;
  /** 随行小伙伴人数 */
  guests: number;
  /** 已报名球员 id，用于判断当前身份是否已报名 */
  signedUpIds: number[];
}

/** 本周报名预览卡（mock signupPreview）；数据由 server 端 page 传入 */
export function SignupCard({
  sessionDate,
  totalPeople,
  guests,
  signedUpIds,
}: SignupCardProps) {
  const [myId, setMyId] = React.useState<number | null>(null);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const mine = myId !== null && signedUpIds.includes(myId);
  const [, month, day] = sessionDate.split("-").map(Number);

  return (
    <section className="flex flex-col rounded-2xl border border-border bg-card p-5 min-[761px]:px-[25px] min-[761px]:py-[23px]">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-bold text-card-foreground">本周报名</h2>
        <span
          className={cn(
            "inline-flex items-center rounded-[5px] px-[7px] py-1 text-[10px] font-bold",
            mine ? "bg-win-bg text-win" : "bg-secondary text-muted-foreground"
          )}
        >
          {mine ? "已报名" : "报名中"}
        </span>
      </div>

      <div className="mt-[19px] flex items-center gap-[17px]">
        <div className="flex shrink-0 flex-col items-center rounded-[10px] bg-secondary px-[17px] py-2">
          <span className="text-[9px] text-muted-foreground">{month} 月</span>
          <strong className="font-num text-[35px] leading-[1.2] font-semibold text-card-foreground">
            {String(day).padStart(2, "0")}
          </strong>
          <span className="text-[9px] text-muted-foreground">周三</span>
        </div>
        <div>
          <h3 className="text-base font-semibold text-card-foreground">
            周三羽毛球局
          </h3>
          <p className="mt-[5px] text-xs text-muted-foreground">18:00–20:00</p>
          <div className="mt-1.5 flex items-baseline gap-1">
            <strong className="font-num text-2xl font-semibold text-card-foreground">
              {totalPeople}
            </strong>
            <span className="text-xs text-muted-foreground">
              人参加 · 含 {guests} 位小伙伴
            </span>
          </div>
        </div>
      </div>

      <Link
        href="/signup"
        className="mt-5 inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-[9px] border border-border bg-card text-xs font-bold text-card-foreground transition-colors hover:bg-secondary"
      >
        {mine ? "查看我的报名" : "查看名单 · 去报名"}
        <ArrowRight className="size-4" strokeWidth={1.65} />
      </Link>
    </section>
  );
}
