import { listMatchesByDate } from "@/lib/repo";
import { leaderboardSummaries } from "@/lib/stats";
import { loadRatingView } from "@/lib/rating-view";
import { getWeekRange } from "@/lib/weekly";
import {
  getActiveSessionDate,
  listSignups,
  signupSummary,
} from "@/lib/signup";
import { HomeTrend } from "@/components/home-trend";
import { WeekMatches } from "@/components/week-matches";
import { SignupCard } from "@/components/signup-card";
import { PredictCard } from "@/components/predict-card";
import { WeeklyQuote } from "@/components/weekly-quote";
import { RatingBoundaryRefresh } from "@/components/rating-boundary-refresh";
import { RatingModeControl } from "@/components/rating-mode-control";
import { RatingStatus, toRatingStatusInput } from "@/components/rating-status";
import {
  OverviewSummary,
  type OverviewStat,
} from "@/components/overview-summary";

export const dynamic = "force-dynamic";

interface HomePageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

function getTodayString(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

export default async function HomePage({ searchParams }: HomePageProps) {
  const sp = await searchParams;
  // 一次调用取统一投影：legacy 分支 view 为 StatsData（旧路径），
  // glicko2 分支 view 为 RatingView。
  const result = loadRatingView({
    rating: typeof sp.rating === "string" ? sp.rating : undefined,
  });
  const matches =
    result.model === "legacy" ? result.view.matches : listMatchesByDate();
  const today = getTodayString();
  const { weekStart } = getWeekRange(today);
  const weekMatchCount = matches.filter((m) => m.playedAt >= weekStart).length;

  const isLegacy = result.model === "legacy";
  const ratingQuery = isLegacy ? "" : "?rating=glicko2";

  // 个人摘要卡：生涯胜率按全部比赛计算，无比赛为 null（显示「—」）
  const record = new Map<number, { total: number; wins: number }>();
  for (const m of matches) {
    const aWon = m.scoreA > m.scoreB;
    for (const [ids, won] of [
      [[m.pa1, m.pa2], aWon],
      [[m.pb1, m.pb2], !aWon],
    ] as const) {
      for (const id of ids) {
        const r = record.get(id) ?? { total: 0, wins: 0 };
        r.total++;
        if (won) r.wins++;
        record.set(id, r);
      }
    }
  }

  const players =
    result.model === "legacy"
      ? result.view.players
      : result.view.players.map((p) => ({ id: p.playerId, name: p.name }));

  const winRateOf = (id: number): number | null => {
    const r = record.get(id);
    return r && r.total > 0 ? Math.round((r.wins / r.total) * 100) : null;
  };

  const overviewStats: Record<number, OverviewStat> = {};
  if (result.model === "legacy") {
    const summaries = leaderboardSummaries(result.view, weekStart);
    for (const p of players) {
      overviewStats[p.id] = {
        elo: summaries[p.id].elo,
        rank: summaries[p.id].rank,
        winRate: winRateOf(p.id),
      };
    }
  } else {
    for (const p of result.view.players) {
      overviewStats[p.playerId] = {
        elo: p.displayRating,
        rank: p.rank,
        status: p.status,
        winRate: winRateOf(p.playerId),
      };
    }
  }

  // 本周报名预览数据（lib/signup 为准）
  const sessionDate = getActiveSessionDate(new Date());
  const { count, totalPeople } = signupSummary(sessionDate);
  const signedUpIds = listSignups(sessionDate).map((s) => s.playerId);

  const panelClass = "rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]";

  // 当前区段：图表「当前季度/预估虚线」的锚点
  const currentSegment =
    result.model === "glicko2"
      ? (result.view.weekSegments.find(
          (segment) => segment.segmentId === result.currentSegmentId
        ) ?? null)
      : null;
  const currentSeason = currentSegment?.seasonId ?? null;

  return (
    <div className="flex flex-col gap-5 min-[761px]:gap-6">
      <RatingBoundaryRefresh
        nextBoundary={result.model === "glicko2" ? result.nextBoundary : null}
      />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          {/* 三行状态说明信息密度高，仅首页展开；其余页面用默认 false。 */}
          <RatingStatus {...toRatingStatusInput(result)} showLegend />
        </div>
        <RatingModeControl current={result.model} />
      </div>

      <WeeklyQuote />

      <div className="grid gap-5 min-[761px]:grid-cols-[1.25fr_1fr] min-[761px]:gap-[22px] min-[1191px]:grid-cols-[1.55fr_1fr]">
        <OverviewSummary
          players={players}
          stats={overviewStats}
          weekMatchCount={weekMatchCount}
          ratingLabel={isLegacy ? "当前 ELO" : "当前评分"}
        />
        <SignupCard
          sessionDate={sessionDate}
          totalPeople={totalPeople}
          guests={totalPeople - count}
          signedUpIds={signedUpIds}
        />
      </div>

      {isLegacy ? (
        <HomeTrend
          model="legacy"
          history={result.view.eloHistory}
          players={players}
          variant="compact"
        />
      ) : (
        <HomeTrend
          model="glicko2"
          view={result.view}
          currentSegmentId={result.currentSegmentId ?? ""}
          currentSeason={currentSeason}
          variant="compact"
          ratingQuery={ratingQuery}
        />
      )}

      <section className={panelClass}>
        <WeekMatches matches={matches} weekStart={weekStart} />
      </section>

      <PredictCard />
    </div>
  );
}
