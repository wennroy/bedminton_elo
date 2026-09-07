import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { HomeTrend } from "@/components/home-trend";
import { buildStatsData } from "@/lib/stats";

export const dynamic = "force-dynamic";

export default async function TrendsPage() {
  const data = buildStatsData();

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
    </div>
  );
}
