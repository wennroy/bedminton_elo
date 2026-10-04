"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { BookOpen } from "lucide-react";
import type { RatingModel } from "@/lib/ratings/types";
import { cn } from "@/lib/utils";

/** 可切换的评分模型与展示名（配置未初始化时新版会如实显示不可用）。 */
export const RATING_MODEL_OPTIONS: ReadonlyArray<{
  value: RatingModel;
  label: string;
  title: string;
}> = [
  { value: "glicko2", label: "新版", title: "新版 Glicko-2 双打评分" },
  { value: "legacy", label: "Legacy", title: "旧版 ELO 评分" },
];

/** 查询参数中的 rating 值是否合法；非法值由服务端回退到默认模型。 */
export function isRatingModelParam(value: unknown): value is RatingModel {
  return value === "glicko2" || value === "legacy";
}

/**
 * 在现有查询串上切换 rating 参数并保留其余参数（weekly 的 week、
 * predict 的 pa1..pb2 预填阵容等）。纯函数，便于测试。
 */
export function applyRatingParam(
  currentQuery: string,
  mode: RatingModel
): string {
  const params = new URLSearchParams(currentQuery);
  params.set("rating", mode);
  return params.toString();
}

export interface RatingModeControlProps {
  /** 当前生效模型：页面按显式 rating 参数解析，非法/缺省回 activeModel。 */
  current: RatingModel;
}

/**
 * 评分模型切换：只改 rating 查询参数，页面其他选中状态原样保留；
 * 普通链接导航，不调用任何输入框 focus、不弹键盘。
 */
export function RatingModeControl({ current }: RatingModeControlProps) {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  return (
    <div className="flex items-center gap-1.5">
      <div
        role="group"
        aria-label="评分模型"
        className="inline-flex items-center rounded-lg border border-border bg-card p-0.5"
      >
        {RATING_MODEL_OPTIONS.map((option) => {
          const active = option.value === current;
          const query = applyRatingParam(
            searchParams.toString(),
            option.value
          );
          return (
            <Link
              key={option.value}
              href={`${pathname}?${query}`}
              title={option.title}
              aria-pressed={active}
              className={cn(
                "rounded-[7px] px-2.5 py-1 text-xs font-medium transition-colors",
                active
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {option.label}
            </Link>
          );
        })}
      </div>
      <Link
        href={
          current === "glicko2" ? "/methodology/glicko2" : "/methodology/legacy"
        }
        title="计分方式说明"
        className="inline-flex items-center gap-1 rounded-lg border border-border bg-card px-2 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
      >
        <BookOpen className="size-3" strokeWidth={1.65} />
        <span className="hidden min-[761px]:inline">计分方式</span>
      </Link>
    </div>
  );
}
