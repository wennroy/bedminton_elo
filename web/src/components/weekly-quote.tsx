"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { quoteForWeek } from "@/lib/quote";

const navBtnClass =
  "grid size-[29px] place-items-center rounded-full border border-border bg-card text-foreground transition-colors hover:bg-secondary disabled:opacity-40 disabled:hover:bg-card min-[761px]:size-8";

export function WeeklyQuote() {
  const [offset, setOffset] = React.useState(0);
  // 上海时区按周确定，同一周服务端与客户端渲染结果一致
  const quote = quoteForWeek(new Date(), offset);

  return (
    <div className="relative flex items-end justify-between gap-3">
      <div>
        <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground">
          每周名句
          <span className="mt-1 block text-[9px] font-normal tracking-[0.5px] min-[761px]:ml-3.5 min-[761px]:mt-0 min-[761px]:inline min-[761px]:text-[10px]">
            {quote.dateLabel}
          </span>
        </div>
        <h1
          data-week={quote.weekStart}
          className="mt-[11px] mb-2 text-[23px] leading-[1.6] font-semibold min-[761px]:mt-[13px] min-[761px]:mb-[9px] min-[761px]:text-[29px]"
        >
          {quote.text}
        </h1>
        <div className="text-[9px] text-muted-foreground min-[761px]:text-[10px]">
          原创 · 每周一更新
        </div>
      </div>
      <div className="absolute top-0 right-0 flex shrink-0 gap-[5px] min-[761px]:static min-[761px]:gap-2">
        <button
          type="button"
          aria-label="查看上周名句"
          onClick={() => setOffset((o) => o - 1)}
          className={navBtnClass}
        >
          <ChevronLeft className="size-[15px]" strokeWidth={1.65} />
        </button>
        <button
          type="button"
          aria-label="查看下一周名句"
          disabled={offset >= 0}
          onClick={() => setOffset((o) => Math.min(0, o + 1))}
          className={navBtnClass}
        >
          <ChevronRight className="size-[15px]" strokeWidth={1.65} />
        </button>
      </div>
    </div>
  );
}
