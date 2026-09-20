import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { Leaderboard } from "@/components/leaderboard";
import { HomeTrend } from "@/components/home-trend";
import { buildStatsData, leaderboardSummaries } from "@/lib/stats";
import { getWeekRange } from "@/lib/weekly";

export const dynamic = "force-dynamic";

function getTodayString(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

export default async function TrendsPage() {
  const data = buildStatsData();
  const { weekStart } = getWeekRange(getTodayString());
  const summaries = leaderboardSummaries(data, weekStart);

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-5">
      <div className="flex items-center justify-between gap-5">
        <div>
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
            CLUB STATISTICS
          </div>
          <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
            全员 ELO 趋势
          </h1>
          <p className="mt-2 text-xs text-muted-foreground max-[760px]:text-[11px]">
            按比赛日汇总 · 可选择成员对比
          </p>
        </div>
        <Link
          href="/players"
          className="inline-flex min-h-11 shrink-0 items-center gap-2 rounded-[9px] border border-border bg-card px-4 text-xs font-bold text-card-foreground transition-colors hover:bg-secondary max-[760px]:min-h-9 max-[760px]:px-3 max-[760px]:text-[11px]"
        >
          球员档案
          <ArrowRight className="size-[15px]" strokeWidth={1.65} />
        </Link>
      </div>

      <HomeTrend history={data.eloHistory} players={data.players} variant="full" />

      <section className="rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
        <h2 className="mb-5 text-lg font-bold text-card-foreground">
          当前球员排行榜
        </h2>
        <Leaderboard
          players={data.players}
          matches={data.matches}
          summaries={summaries}
        />
      </section>
    </div>
  );
}
