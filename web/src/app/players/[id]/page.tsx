import { notFound } from "next/navigation";
import Link from "next/link";
import { PlayerAvatar } from "@/components/player-avatar";
import { H2hTable } from "@/components/h2h-table";
import { Button } from "@/components/ui/button";
import {
  buildStatsData,
  headToHead,
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
              {Math.round(summary.tsScore)}
            </span>
            <span className="text-[10px] text-muted-foreground">μ-3σ</span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-sm text-muted-foreground">
          <Trophy className="size-4" />
          {summary.totalMatches} 场 · {summary.wins} 胜 {summary.losses} 负 · {" "}
          {summary.winRate}%
        </div>
      </section>

      <section className="mb-6 flex flex-col gap-3">
        <h3 className="text-lg font-bold text-foreground">交锋记录</h3>
        <H2hTable records={h2h} />
      </section>

      <section className="flex flex-col gap-3">
        <h3 className="text-lg font-bold text-foreground">参赛历史</h3>
        {matches.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
            还没有参赛记录
          </div>
        ) : (
          <div className="space-y-2">
            {matches.map((m) => (
              <div
                key={m.id}
                className="rounded-2xl border border-border bg-card p-3 shadow-sm"
              >
                <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                  <span>{m.date}</span>
                  <span className={m.won ? "text-emerald-600" : "text-rose-600"}>
                    {m.won ? "胜" : "负"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-1 flex-col gap-1 text-sm">
                    <div className="flex items-center gap-1 text-card-foreground">
                      <span className="font-medium">
                        {summary.name} / {m.teammates.join(" / ")}
                      </span>
                    </div>
                    <div className="text-muted-foreground">
                      VS {m.opponents.join(" / ")}
                    </div>
                  </div>
                  <div className="text-lg font-bold tabular-nums text-card-foreground">
                    {m.scoreFor} : {m.scoreAgainst}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
