import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Leaderboard } from "@/components/leaderboard";
import { HomeTrend } from "@/components/home-trend";
import { RatingBoundaryRefresh } from "@/components/rating-boundary-refresh";
import { RatingModeControl } from "@/components/rating-mode-control";
import { RatingStatus, toRatingStatusInput } from "@/components/rating-status";
import { leaderboardSummaries } from "@/lib/stats";
import { loadRatingView } from "@/lib/rating-view";
import { shanghaiLocalDateFromInstant } from "@/lib/ratings/calendar";
import { getWeekRange } from "@/lib/weekly";

export const dynamic = "force-dynamic";

interface TrendsPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

function getTodayString(): string {
  // 营业日以 Asia/Shanghai 为准，与评分引擎口径一致，不随服务器时区漂移
  return shanghaiLocalDateFromInstant(new Date().toISOString());
}

export default async function TrendsPage({ searchParams }: TrendsPageProps) {
  const sp = await searchParams;
  const result = loadRatingView({
    rating: typeof sp.rating === "string" ? sp.rating : undefined,
  });
  const { weekStart } = getWeekRange(getTodayString());

  const isLegacy = result.model === "legacy";
  const ratingQuery = isLegacy ? "" : "?rating=glicko2";

  // 当前区段（跨周/季界刷新区段会变化）：图表「当前季度/预估虚线」的锚点
  const currentSegment =
    result.model === "glicko2"
      ? (result.view.weekSegments.find(
          (segment) => segment.segmentId === result.currentSegmentId
        ) ?? null)
      : null;
  const currentSeason = currentSegment?.seasonId ?? null;

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-5">
      <RatingBoundaryRefresh
        nextBoundary={result.model === "glicko2" ? result.nextBoundary : null}
      />

      <div className="flex items-center justify-between gap-5">
        <div>
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
            CLUB STATISTICS
          </div>
          <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
            {isLegacy ? "全员 ELO 趋势" : "全员评分趋势"}
          </h1>
          <p className="mt-2 text-xs text-muted-foreground max-[760px]:text-[11px]">
            {isLegacy
              ? "按比赛日汇总 · 可选择成员对比"
              : "逐场预估 · 每周正式结算 · 可选择成员对比"}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2.5">
          <RatingModeControl current={result.model} />
          <Link
            href={`/players${ratingQuery}`}
            className="inline-flex min-h-11 shrink-0 items-center gap-2 rounded-[9px] border border-border bg-card px-4 text-xs font-bold text-card-foreground transition-colors hover:bg-secondary max-[760px]:min-h-9 max-[760px]:px-3 max-[760px]:text-[11px]"
          >
            球员档案
            <ArrowRight className="size-[15px]" strokeWidth={1.65} />
          </Link>
        </div>
      </div>

      <RatingStatus {...toRatingStatusInput(result)} />

      {isLegacy ? (
        <HomeTrend
          model="legacy"
          history={result.view.eloHistory}
          players={result.view.players}
          variant="full"
        />
      ) : (
        <HomeTrend
          model="glicko2"
          view={result.view}
          currentSegmentId={result.currentSegmentId ?? ""}
          currentSeason={currentSeason}
          now={result.asOf}
          variant="full"
          ratingQuery={ratingQuery}
        />
      )}

      <section className="rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
        <h2 className="mb-5 text-lg font-bold text-card-foreground">
          当前球员排行榜
        </h2>
        {isLegacy ? (
          <Leaderboard
            model="legacy"
            players={result.view.players}
            matches={result.view.matches}
            summaries={leaderboardSummaries(result.view, weekStart)}
          />
        ) : (
          <Leaderboard
            model="glicko2"
            rows={result.view.players}
            ratingQuery={ratingQuery}
          />
        )}
      </section>
    </div>
  );
}
