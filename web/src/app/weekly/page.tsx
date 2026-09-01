import { notFound } from "next/navigation";
import { buildWeeklyStats, listWeekStarts, getWeekRange } from "@/lib/weekly";
import { WeeklyView } from "./weekly-view";

export const dynamic = "force-dynamic";

interface WeeklyPageProps {
  searchParams: Promise<{ week?: string }>;
}

export default async function WeeklyPage({ searchParams }: WeeklyPageProps) {
  const params = await searchParams;
  const weekStarts = listWeekStarts();
  if (weekStarts.length === 0) {
    return (
      <main className="min-h-full bg-background px-4 pb-28 pt-4">
        <h1 className="mb-4 text-xl font-bold text-foreground">周报</h1>
        <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          还没有比赛数据
        </div>
      </main>
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

  const stats = buildWeeklyStats(weekStart);

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <WeeklyView stats={stats} weekStarts={weekStarts} />
    </main>
  );
}
