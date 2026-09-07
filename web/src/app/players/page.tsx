import { buildStatsData } from "@/lib/stats";
import { INITIAL_RATING } from "@/lib/elo";
import {
  PlayerDirectory,
  type PlayerDirectoryEntry,
} from "@/components/player-directory";

export const dynamic = "force-dynamic";

export default async function PlayersPage() {
  const data = buildStatsData();
  const eloOf = (id: number) =>
    Math.round(data.ratings.get(id)?.elo ?? INITIAL_RATING);

  // 每球员场次 / 胜场（全部比赛）
  const totals = new Map<number, { total: number; wins: number }>();
  for (const m of data.matches) {
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

  const entries: PlayerDirectoryEntry[] = ordered.map((p, i) => {
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

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-5">
      <div className="flex items-center justify-between gap-5">
        <div>
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
            PLAYER DIRECTORY
          </div>
          <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
            球员档案
          </h1>
        </div>
        <span className="inline-flex items-center rounded-[5px] bg-secondary px-[7px] py-1 text-[10px] font-bold text-muted-foreground">
          {entries.length} 位球友
        </span>
      </div>

      <PlayerDirectory entries={entries} />
    </div>
  );
}
