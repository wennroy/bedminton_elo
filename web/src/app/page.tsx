import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { listPlayers, listMatchesByDate } from "@/lib/repo";
import { buildStatsData } from "@/lib/stats";
import { getWeekRange } from "@/lib/weekly";
import { INITIAL_RATING } from "@/lib/elo";
import {
  getActiveSessionDate,
  listSignups,
  signupSummary,
} from "@/lib/signup";
import { Leaderboard } from "@/components/leaderboard";
import { HomeTrend, type PlayerSummaryLite } from "@/components/home-trend";
import { WeekMatches } from "@/components/week-matches";
import { SignupCard } from "@/components/signup-card";
import { PredictCard } from "@/components/predict-card";
import { WeeklyQuote } from "@/components/weekly-quote";
import {
  OverviewSummary,
  type OverviewStat,
} from "@/components/overview-summary";

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
  const weekMatchCount = matches.filter((m) => m.playedAt >= weekStart).length;

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
  const overviewStats: Record<number, OverviewStat> = {};
  for (const p of players) {
    const r = record.get(p.id);
    overviewStats[p.id] = {
      elo: summaries[p.id].elo,
      rank: summaries[p.id].rank,
      winRate: r && r.total > 0 ? Math.round((r.wins / r.total) * 100) : null,
    };
  }

  // 本周报名预览数据（lib/signup 为准）
  const sessionDate = getActiveSessionDate(new Date());
  const { count, totalPeople } = signupSummary(sessionDate);
  const signedUpIds = listSignups(sessionDate).map((s) => s.playerId);

  const panelClass = "rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]";
  const panelHeadingClass =
    "mb-5 flex items-center justify-between gap-3.5";
  const textLinkClass =
    "inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-win";

  return (
    <div className="flex flex-col gap-5 min-[761px]:gap-6">
      <WeeklyQuote />

      <div className="grid gap-5 min-[761px]:grid-cols-[1.25fr_1fr] min-[761px]:gap-[22px] min-[1191px]:grid-cols-[1.55fr_1fr]">
        <OverviewSummary
          players={players}
          stats={overviewStats}
          weekMatchCount={weekMatchCount}
        />
        <SignupCard
          sessionDate={sessionDate}
          totalPeople={totalPeople}
          guests={totalPeople - count}
          signedUpIds={signedUpIds}
        />
      </div>

      <HomeTrend history={eloHistory} summaries={summaries} />

      <div className="grid items-start gap-5 min-[761px]:grid-cols-[1.25fr_1fr] min-[761px]:gap-[22px] min-[1191px]:grid-cols-[1.55fr_1fr]">
        <section className={panelClass}>
          <div className={panelHeadingClass}>
            <h2 className="text-lg font-bold text-card-foreground">
              球员排行榜
            </h2>
            <Link href="/players" className={textLinkClass}>
              全部球员
              <ArrowRight className="size-[15px]" strokeWidth={1.65} />
            </Link>
          </div>
          <Leaderboard
            players={players}
            matches={matches}
            summaries={summaries}
          />
        </section>
        <section className={panelClass}>
          <WeekMatches matches={matches} weekStart={weekStart} />
        </section>
      </div>

      <PredictCard />
    </div>
  );
}
