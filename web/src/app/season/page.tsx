import Link from "next/link";
import { RatingModeControl } from "@/components/rating-mode-control";
import { readRatingConfig } from "@/lib/rating-config";
import type { RatingModel } from "@/lib/ratings/types";
import {
  buildSeasonPageData,
  loadSeasonRatingParams,
  normalizeSeasonParam,
  WeeklyRatingUnavailableError,
} from "@/lib/season";
import { SeasonView } from "./season-view";

export const dynamic = "force-dynamic";

interface SeasonPageProps {
  searchParams: Promise<{ season?: string; rating?: string }>;
}

/** 显式 "glicko2"|"legacy" 优先；非法/缺省回 activeModel（无配置默认 legacy）。 */
function resolveSeasonModel(requested: string | undefined): RatingModel {
  if (requested === "glicko2" || requested === "legacy") return requested;
  return readRatingConfig()?.activeModel ?? "legacy";
}

export default async function SeasonPage({ searchParams }: SeasonPageProps) {
  const params = await searchParams;
  const ratingParam = params.rating;
  const rating = typeof ratingParam === "string" ? ratingParam : undefined;
  const model = resolveSeasonModel(rating);

  // Legacy 分支：赛季制度属于新版评分，空态说明 + 切换引导（不取数）。
  if (model === "legacy") {
    return (
      <div>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-xl font-bold text-foreground">赛季</h1>
          <RatingModeControl current="legacy" />
        </div>
        <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center">
          <p className="text-sm text-muted-foreground">
            赛季制度属于新版 Glicko-2 评分：自然季度划分、每周一结算与跨季软重置只在新版下统计。
          </p>
          <p className="text-xs text-muted-foreground">
            使用右上角开关切换到「新版」查看赛季统计。
          </p>
        </div>
      </div>
    );
  }

  const season = normalizeSeasonParam(params.season);
  let data;
  try {
    data = buildSeasonPageData(season);
  } catch (error) {
    if (error instanceof WeeklyRatingUnavailableError) {
      // glicko2 不可用：如实拒答展示，不伪造数据、不静默退回 legacy。
      return (
        <div>
          <h1 className="mb-4 text-xl font-bold text-foreground">赛季</h1>
          <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-loss/50 bg-loss-bg p-6 text-center">
            <p className="text-sm text-loss">
              新版评分暂不可用：{error.reason}
            </p>
            <Link
              href="/season?rating=legacy"
              className="text-xs text-muted-foreground underline underline-offset-2 transition-colors hover:text-foreground"
            >
              查看 Legacy ELO 页面
            </Link>
          </div>
        </div>
      );
    }
    throw error;
  }

  const ratingParams = loadSeasonRatingParams();

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-bold text-foreground">赛季</h1>
        <RatingModeControl current="glicko2" />
      </div>
      <SeasonView data={data} params={ratingParams} />
    </div>
  );
}
