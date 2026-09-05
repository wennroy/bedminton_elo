import { listPlayers, listMatchesByDate } from "@/lib/repo";
import { buildStatsData } from "@/lib/stats";
import { getWeekRange } from "@/lib/weekly";
import { INITIAL_RATING } from "@/lib/elo";
import { Leaderboard } from "@/components/leaderboard";
import { HomeTrend, type PlayerSummaryLite } from "@/components/home-trend";
import { WeekMatches } from "@/components/week-matches";
import { CollapsibleSection } from "@/components/collapsible-section";
import { SignupCard } from "@/components/signup-card";
import { PredictCard } from "@/components/predict-card";

export const dynamic = "force-dynamic";

function getTodayString(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

export default async function HomePage() {
  const players = listPlayers();
  const matches = listMatchesByDate();
  const today = getTodayString();
  const { weekStart } = getWeekRange(today);
  const weekMatches = matches.filter((m) => m.playedAt >= weekStart);

  const { ratings, eloHistory } = buildStatsData();

  const eloOf = (id: number) =>
    Math.round(ratings.get(id)?.elo ?? INITIAL_RATING);
  const rankOf = new Map(
    [...players]
      .sort((a, b) => eloOf(b.id) - eloOf(a.id))
      .map((p, i) => [p.id, i + 1] as const)
  );

  // 本周涨跌 = 当前 ELO − 本周一之前最后一个快照(无快照则以初始分计)
  const lastBeforeWeek = new Map<number, number>();
  for (const h of eloHistory) {
    if (h.date < weekStart) lastBeforeWeek.set(Number(h.playerId), h.elo);
  }
  const hasHistory = new Set(eloHistory.map((h) => Number(h.playerId)));

  const summaries: Record<number, PlayerSummaryLite> = {};
  for (const p of players) {
    const elo = eloOf(p.id);
    summaries[p.id] = {
      elo,
      rank: rankOf.get(p.id) ?? 0,
      weekDelta: hasHistory.has(p.id)
        ? elo - (lastBeforeWeek.get(p.id) ?? INITIAL_RATING)
        : 0,
    };
  }

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <div className="flex flex-col gap-4">
        <SignupCard />
        <PredictCard />
        <HomeTrend history={eloHistory} summaries={summaries} />
        <CollapsibleSection title="排行榜" badge={`${players.length} 人`}>
          <Leaderboard players={players} matches={matches} />
        </CollapsibleSection>
        <CollapsibleSection title="本周战绩" badge={`${weekMatches.length} 场`}>
          <WeekMatches matches={weekMatches} />
        </CollapsibleSection>
      </div>
    </main>
  );
}
