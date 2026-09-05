import { listPlayers, recomputeAllRatings } from "@/lib/repo";
import { PredictForm } from "./predict-form";

export const dynamic = "force-dynamic";

type Slot = number | null;

interface PredictPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

export default async function PredictPage({ searchParams }: PredictPageProps) {
  const players = listPlayers();
  const ratings = recomputeAllRatings();
  const sp = await searchParams;

  // 配对页带过来的预填阵容(如 /predict?pa1=1&pa2=2&pb1=3&pb2=4);
  // 非法、不存在或重复的 id 对应槽位置空,用户手动补选
  const validIds = new Set(players.map((p) => p.id));
  const seen = new Set<number>();
  const parse = (key: "pa1" | "pa2" | "pb1" | "pb2"): Slot => {
    const raw = sp[key];
    const n = typeof raw === "string" ? Number(raw) : NaN;
    if (!Number.isInteger(n) || !validIds.has(n) || seen.has(n)) return null;
    seen.add(n);
    return n;
  };
  const initialTeamA: [Slot, Slot] = [parse("pa1"), parse("pa2")];
  const initialTeamB: [Slot, Slot] = [parse("pb1"), parse("pb2")];

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <h1 className="mb-4 text-xl font-bold text-foreground">2v2 胜率预测</h1>
      <PredictForm
        players={players}
        ratings={ratings}
        initialTeamA={initialTeamA}
        initialTeamB={initialTeamB}
      />
    </main>
  );
}
