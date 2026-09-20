import Link from "next/link";
import { MatchBrowser } from "@/components/match-browser";
import { listMatchesByDate, listPlayers } from "@/lib/repo";

export const dynamic = "force-dynamic";

export default async function MatchesPage() {
  const matches = listMatchesByDate();
  const players = listPlayers();

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
            CLUB MATCHES
          </div>
          <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
            所有比赛
          </h1>
          <p className="mt-2 text-xs text-muted-foreground max-[760px]:text-[11px]">
            浏览俱乐部已记录的全部双打比赛，可按球员和日期筛选。
          </p>
        </div>
        <Link
          href="/"
          className="inline-flex min-h-9 shrink-0 items-center rounded-[9px] border border-border bg-card px-3 text-xs font-bold text-card-foreground transition-colors hover:bg-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          返回首页
        </Link>
      </div>

      <MatchBrowser matches={matches} players={players} />
    </div>
  );
}
