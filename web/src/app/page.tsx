import { listPlayers, listMatchesByDate, type MatchWithNames } from "@/lib/repo";
import { Leaderboard } from "@/components/leaderboard";

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
  const todayMatches = matches
    .filter((m): m is MatchWithNames => m.playedAt === today)
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <Leaderboard
        players={players}
        matches={matches}
        todayMatches={todayMatches}
      />
    </main>
  );
}
