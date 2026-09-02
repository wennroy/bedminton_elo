import { notFound } from "next/navigation";
import Link from "next/link";
import { PlayerAvatar } from "@/components/player-avatar";
import { H2hTable } from "@/components/h2h-table";
import { PlayerMatchHistory } from "@/components/player-match-history";
import { FunStats } from "@/components/fun-stats";
import { Button } from "@/components/ui/button";
import {
  buildStatsData,
  headToHead,
  playerFunStats,
  playerMatches,
  playerSummary,
} from "@/lib/stats";
import { ChevronLeft, Trophy } from "lucide-react";

export const dynamic = "force-dynamic";

interface PlayerPageProps {
  params: Promise<{ id: string }>;
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

  const h2h = headToHead(playerId, data);
  const matches = playerMatches(playerId, data);
  const funStats = playerFunStats(playerId, data);

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <div className="mb-4 flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" asChild>
          <Link href="/" aria-label="返回">
            <ChevronLeft className="size-5" />
          </Link>
        </Button>
        <h1 className="text-xl font-bold text-foreground">球员主页</h1>
      </div>

      <section className="mb-6 flex flex-col items-center gap-3 rounded-2xl border border-border bg-card p-6 shadow-sm">
        <PlayerAvatar name={summary.name} size="lg" />
        <h2 className="text-2xl font-bold text-card-foreground">
          {summary.name}
        </h2>
        <div className="flex w-full gap-3">
          <div className="flex flex-1 flex-col items-center rounded-xl bg-muted p-3">
            <span className="text-xs text-muted-foreground">ELO</span>
            <span className="text-xl font-bold tabular-nums text-card-foreground">
              {Math.round(summary.elo)}
            </span>
          </div>
          <div className="flex flex-1 flex-col items-center rounded-xl bg-muted p-3">
            <span className="text-xs text-muted-foreground">TrueSkill</span>
            <span className="text-xl font-bold tabular-nums text-card-foreground">
              {Math.round(summary.mu)}
            </span>
            <span className="text-[10px] text-muted-foreground">
              σ {summary.sigma.toFixed(1)} · 区间 [
              {Math.round(summary.mu - 3 * summary.sigma)},{" "}
              {Math.round(summary.mu + 3 * summary.sigma)}]
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-sm text-muted-foreground">
          <Trophy className="size-4" />
          {summary.totalMatches} 场 · {summary.wins} 胜 {summary.losses} 负 · {" "}
          {summary.winRate}%
        </div>
      </section>

      <section className="mb-6 flex flex-col gap-3">
        <h3 className="text-lg font-bold text-foreground">趣味数据</h3>
        <FunStats stats={funStats} />
      </section>

      <section className="mb-6 flex flex-col gap-3">
        <h3 className="text-lg font-bold text-foreground">交锋记录</h3>
        <H2hTable records={h2h} />
      </section>

      <section className="flex flex-col gap-3">
        <h3 className="text-lg font-bold text-foreground">参赛历史</h3>
        <PlayerMatchHistory matches={matches} playerName={summary.name} />
      </section>
    </main>
  );
}
