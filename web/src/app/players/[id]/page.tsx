import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import {
  buildStatsData,
  playerFunStats,
  playerMatches,
  playerRelations,
  playerSummary,
} from "@/lib/stats";
import { INITIAL_RATING } from "@/lib/elo";
import { getWeekRange } from "@/lib/weekly";
import { ProfileHeader, MoreMetrics } from "@/components/fun-stats";
import { PlayerTrend, RecentForm } from "@/components/player-trend";
import { PlayerRelations } from "@/components/player-relations";
import { PlayerMatchHistory } from "@/components/player-match-history";
import { cn } from "@/lib/utils";

export const dynamic = "force-dynamic";

interface PlayerPageProps {
  params: Promise<{ id: string }>;
}

function todayString(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export default async function PlayerPage({ params }: PlayerPageProps) {
  const { id: idParam } = await params;
  const playerId = Number(idParam);
  if (!Number.isFinite(playerId)) {
    notFound();
  }

  const data = buildStatsData();
  const summary = playerSummary(playerId, data);
  if (!summary) {
    notFound();
  }

  const funStats = playerFunStats(playerId, data);
  const matches = playerMatches(playerId, data);
  const relations = playerRelations(playerId, data);

  // 排名：ELO 降序，同分按 id 升序（与排行榜口径一致）
  const eloOf = (id: number) =>
    Math.round(data.ratings.get(id)?.elo ?? INITIAL_RATING);
  const ordered = [...data.players].sort(
    (a, b) => eloOf(b.id) - eloOf(a.id) || a.id - b.id
  );
  const rank = ordered.findIndex((p) => p.id === playerId) + 1;

  // 本周变化（口径同首页）：当前 ELO − 本周一之前最后一个快照，无快照按初始分
  const { weekStart } = getWeekRange(todayString());
  let lastBeforeWeek: number | null = null;
  let hasHistory = false;
  const trendPoints: { date: string; elo: number }[] = [];
  for (const h of data.eloHistory) {
    if (h.playerId !== String(playerId)) continue;
    hasHistory = true;
    trendPoints.push({ date: h.date, elo: h.elo });
    if (h.date < weekStart) lastBeforeWeek = h.elo;
  }
  const currentElo = Math.round(summary.elo);
  const weekDelta = hasHistory
    ? currentElo - (lastBeforeWeek ?? INITIAL_RATING)
    : 0;

  const switcherPlayers = ordered.map((p) => ({
    id: p.id,
    name: p.name,
    elo: eloOf(p.id),
  }));

  const lastMatchDate = matches[0]?.date;
  const hasMatches = summary.totalMatches > 0;

  const metricCell =
    "px-5 py-[17px] min-[761px]:px-[25px] min-[761px]:py-[22px]";
  const metricLabel = "text-[11px] text-muted-foreground max-[760px]:text-[10px]";
  const metricValue =
    "mt-[5px] mb-1 flex items-center gap-[9px] font-num text-[38px] leading-[1.1] text-card-foreground min-[761px]:text-[43px]";
  const metricUnit = "text-[19px]";
  const metricSub = "text-[10px] text-muted-foreground max-[760px]:text-[9px]";

  return (
    <div className="flex flex-col gap-5 min-[761px]:gap-6">
      <div>
        <Link
          href="/players"
          className="inline-flex min-h-9 items-center gap-[7px] text-xs text-muted-foreground transition-colors hover:text-win"
        >
          <ArrowLeft className="size-[15px]" strokeWidth={1.65} />
          所有球员
        </Link>
      </div>

      <ProfileHeader
        id={playerId}
        name={summary.name}
        rank={rank}
        players={switcherPlayers}
      />

      <section
        aria-label="球员核心数据"
        className="grid grid-cols-2 rounded-2xl border border-border bg-card min-[761px]:grid-cols-[1.2fr_1fr_1fr_1fr]"
      >
        <div
          className={cn(
            metricCell,
            "border-border max-[760px]:border-r max-[760px]:border-b min-[761px]:border-r"
          )}
        >
          <div className={metricLabel}>当前 ELO</div>
          <div className={metricValue}>
            <span>{currentElo}</span>
            <span
              className={cn(
                "inline-flex items-center rounded-[5px] px-[7px] py-1 font-sans text-[10px] font-bold",
                weekDelta > 0
                  ? "bg-win-bg text-win"
                  : weekDelta < 0
                    ? "bg-loss-bg text-loss"
                    : "bg-secondary text-muted-foreground"
              )}
            >
              {weekDelta > 0 ? `+${weekDelta}` : weekDelta}
            </span>
          </div>
          <div className={metricSub}>
            较 {weekStart.slice(5).replace("-", ".")} 前 · 最高{" "}
            {funStats.peakElo}
          </div>
        </div>
        <div
          className={cn(
            metricCell,
            "border-border max-[760px]:border-b min-[761px]:border-r"
          )}
        >
          <div className={metricLabel}>生涯胜率</div>
          <div className={metricValue}>
            {hasMatches ? (
              <>
                <span>{summary.winRate}</span>
                <span className={metricUnit}>%</span>
              </>
            ) : (
              <span>—</span>
            )}
          </div>
          <div className={metricSub}>
            {summary.wins} 胜 / {summary.losses} 负
          </div>
        </div>
        <div className={cn(metricCell, "border-border max-[760px]:border-r min-[761px]:border-r")}>
          <div className={metricLabel}>累计出场</div>
          <div className={metricValue}>
            <span>{summary.totalMatches}</span>
            <span className={metricUnit}>场</span>
          </div>
          <div className={metricSub}>生涯双打比赛</div>
        </div>
        <div className={metricCell}>
          <div className={metricLabel}>当前状态</div>
          <div className={metricValue}>
            {funStats.currentStreakType === "none" ? (
              <span>—</span>
            ) : (
              <>
                <span>{funStats.currentStreak}</span>
                <span className={metricUnit}>
                  连{funStats.currentStreakType === "win" ? "胜" : "负"}
                </span>
              </>
            )}
          </div>
          <div className={metricSub}>
            {lastMatchDate
              ? `最近一场 · ${lastMatchDate.slice(5).replace("-", ".")}`
              : "还没有比赛"}
          </div>
        </div>
      </section>

      <div className="grid items-start gap-[18px] min-[761px]:grid-cols-[minmax(0,1.95fr)_minmax(260px,1fr)] min-[761px]:gap-[22px]">
        <PlayerTrend playerName={summary.name} points={trendPoints} />
        <RecentForm
          matches={matches}
          avgPointDiff={funStats.avgPointDiff}
          peakElo={funStats.peakElo}
          currentElo={currentElo}
        />
      </div>

      <PlayerRelations
        playerName={summary.name}
        partners={relations.partners}
        opponents={relations.opponents}
      />

      <PlayerMatchHistory matches={matches} playerName={summary.name} />

      <MoreMetrics
        mu={summary.mu}
        sigma={summary.sigma}
        longestWinStreak={funStats.longestWinStreak}
        peakElo={funStats.peakElo}
        peakEloDate={funStats.peakEloDate}
      />
    </div>
  );
}
