import Link from "next/link";
import { EloChart } from "@/components/elo-chart";
import { Button } from "@/components/ui/button";
import { buildStatsData } from "@/lib/stats";
import { ChevronLeft } from "lucide-react";

export const dynamic = "force-dynamic";

export default async function TrendsPage() {
  const data = buildStatsData();

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <div className="mb-4 flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" asChild>
          <Link href="/" aria-label="返回">
            <ChevronLeft className="size-5" />
          </Link>
        </Button>
        <h1 className="text-xl font-bold text-foreground">ELO 趋势</h1>
      </div>

      <section className="flex flex-col gap-3">
        <p className="text-sm text-muted-foreground">
          每日 ELO 快照，展示所有球员评分变化。
        </p>
        <EloChart history={data.eloHistory} />
      </section>
    </main>
  );
}
