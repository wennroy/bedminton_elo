import { listMatchesByDate } from "@/lib/repo";
import { INITIAL_RATING } from "@/lib/elo";
import { loadRatingView } from "@/lib/rating-view";
import { RatingBoundaryRefresh } from "@/components/rating-boundary-refresh";
import { RatingModeControl } from "@/components/rating-mode-control";
import { RatingStatus, toRatingStatusInput } from "@/components/rating-status";
import {
  PlayerDirectory,
  type PlayerDirectoryEntry,
} from "@/components/player-directory";

export const dynamic = "force-dynamic";

interface PlayersPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

export default async function PlayersPage({ searchParams }: PlayersPageProps) {
  const sp = await searchParams;
  const result = loadRatingView({
    rating: typeof sp.rating === "string" ? sp.rating : undefined,
  });
  const matches =
    result.model === "legacy" ? result.view.matches : listMatchesByDate();

  const isLegacy = result.model === "legacy";
  const ratingQuery = isLegacy ? "" : "?rating=glicko2";

  // 每球员场次 / 胜场（全部比赛）
  const totals = new Map<number, { total: number; wins: number }>();
  for (const m of matches) {
    const aWon = m.scoreA > m.scoreB;
    for (const [ids, won] of [
      [[m.pa1, m.pa2], aWon],
      [[m.pb1, m.pb2], !aWon],
    ] as const) {
      for (const id of ids) {
        const r = totals.get(id) ?? { total: 0, wins: 0 };
        r.total++;
        if (won) r.wins++;
        totals.set(id, r);
      }
    }
  }

  let entries: PlayerDirectoryEntry[];
  if (result.model === "legacy") {
    const data = result.view;
    const eloOf = (id: number) =>
      Math.round(data.ratings.get(id)?.elo ?? INITIAL_RATING);

    // 排名恒按 ELO：ELO 降序，同分按 id 升序（与排行榜口径一致）
    const ordered = [...data.players].sort(
      (a, b) => eloOf(b.id) - eloOf(a.id) || a.id - b.id
    );

    const sparkByPlayer = new Map<number, number[]>();
    for (const h of data.eloHistory) {
      const id = Number(h.playerId);
      const arr = sparkByPlayer.get(id) ?? [];
      arr.push(h.elo);
      sparkByPlayer.set(id, arr);
    }

    entries = ordered.map((p, i) => {
      const t = totals.get(p.id) ?? { total: 0, wins: 0 };
      return {
        id: p.id,
        name: p.name,
        elo: eloOf(p.id),
        rank: i + 1,
        total: t.total,
        winRate: t.total > 0 ? Math.round((t.wins / t.total) * 100) : 0,
        sparkline: (sparkByPlayer.get(p.id) ?? []).slice(-9),
      };
    });
  } else {
    const view = result.view;
    // 每球员最近 9 个周正式结算分（展示取整），作为迷你走势
    const finalsByPlayer = new Map<number, number[]>();
    for (const point of [...view.points].sort((a, b) => a.order - b.order)) {
      if (point.kind !== "weekly_final") continue;
      const arr = finalsByPlayer.get(point.playerId) ?? [];
      arr.push(Math.round(point.r));
      finalsByPlayer.set(point.playerId, arr);
    }

    entries = view.players.map((p) => {
      const t = totals.get(p.playerId) ?? { total: 0, wins: 0 };
      return {
        id: p.playerId,
        name: p.name,
        elo: p.displayRating,
        rank: p.rank,
        total: t.total,
        winRate: t.total > 0 ? Math.round((t.wins / t.total) * 100) : 0,
        sparkline: (finalsByPlayer.get(p.playerId) ?? []).slice(-9),
        status: p.status,
      };
    });
    // 预按名次（同分按 id 稳定）排序；客户端切换排序时同分保持该口径
    entries.sort(
      (a, b) =>
        (a.rank ?? Number.POSITIVE_INFINITY) -
          (b.rank ?? Number.POSITIVE_INFINITY) || a.id - b.id
    );
  }

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-5">
      <RatingBoundaryRefresh
        nextBoundary={result.model === "glicko2" ? result.nextBoundary : null}
      />

      <div className="flex items-center justify-between gap-5">
        <div>
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
            PLAYER DIRECTORY
          </div>
          <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
            球员档案
          </h1>
        </div>
        <div className="flex shrink-0 items-center gap-2.5">
          <span className="inline-flex items-center rounded-[5px] bg-secondary px-[7px] py-1 text-[10px] font-bold text-muted-foreground">
            {entries.length} 位球友
          </span>
          <RatingModeControl current={result.model} />
        </div>
      </div>

      <RatingStatus {...toRatingStatusInput(result)} />

      <PlayerDirectory
        entries={entries}
        model={result.model}
        ratingQuery={ratingQuery}
      />
    </div>
  );
}
