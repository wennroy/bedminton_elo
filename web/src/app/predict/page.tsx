import { listPlayers, recomputeAllRatings } from "@/lib/repo";
import { PredictForm } from "./predict-form";

export const dynamic = "force-dynamic";

export default async function PredictPage() {
  const players = listPlayers();
  const ratings = recomputeAllRatings();

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <h1 className="mb-4 text-xl font-bold text-foreground">2v2 胜率预测</h1>
      <PredictForm players={players} ratings={ratings} />
    </main>
  );
}
