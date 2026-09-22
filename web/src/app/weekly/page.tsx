import Link from "next/link";
import { notFound } from "next/navigation";
import {
  buildWeeklyStats,
  listWeekStarts,
  getWeekRange,
  WeeklyRatingUnavailableError,
} from "@/lib/weekly";
import { RatingModeControl } from "@/components/rating-mode-control";
import { WeeklyView } from "./weekly-view";

export const dynamic = "force-dynamic";

interface WeeklyPageProps {
  searchParams: Promise<{ week?: string; rating?: string }>;
}

export default async function WeeklyPage({ searchParams }: WeeklyPageProps) {
  const params = await searchParams;
  const weekStarts = listWeekStarts();
  if (weekStarts.length === 0) {
    return (
      <div>
        <h1 className="mb-4 text-xl font-bold text-foreground">周报</h1>
        <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          还没有比赛数据
        </div>
      </div>
    );
  }

  let week = params.week;
  if (!week || !/^\d{4}-\d{2}-\d{2}$/.test(week)) {
    week = weekStarts[weekStarts.length - 1];
  }

  const { weekStart } = getWeekRange(week);
  if (!weekStarts.includes(weekStart)) {
    notFound();
  }

  // 按 searchParams.rating 透传：显式 glicko2/legacy 优先，非法/缺省回 activeModel。
  const ratingParam = params.rating;
  const rating = typeof ratingParam === "string" ? ratingParam : undefined;

  let stats;
  try {
    stats = buildWeeklyStats(weekStart, { rating });
  } catch (error) {
    if (error instanceof WeeklyRatingUnavailableError) {
      // glicko2 不可用：如实拒答展示，不伪造胜率、不静默退回 legacy 数据。
      return (
        <div>
          <h1 className="mb-4 text-xl font-bold text-foreground">周报</h1>
          <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-loss/50 bg-loss-bg p-6 text-center">
            <p className="text-sm text-loss">新版评分暂不可用：{error.reason}</p>
            <Link
              href={`/weekly?week=${weekStart}&rating=legacy`}
              className="text-xs text-muted-foreground underline underline-offset-2 transition-colors hover:text-foreground"
            >
              查看 Legacy ELO 周报
            </Link>
          </div>
        </div>
      );
    }
    throw error;
  }

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-bold text-foreground">周报</h1>
        <RatingModeControl current={stats.ratingReport ? "glicko2" : "legacy"} />
      </div>
      <WeeklyView stats={stats} weekStarts={weekStarts} />
    </div>
  );
}
