import { listPlayers, recomputeAllRatings } from "@/lib/repo";
import { readRatingConfig } from "@/lib/rating-config";
import { loadRatingView } from "@/lib/rating-view";
import type { RatingModel } from "@/lib/ratings/types";
import { RatingBoundaryRefresh } from "@/components/rating-boundary-refresh";
import { RatingModeControl } from "@/components/rating-mode-control";
import { RatingStatus, toRatingStatusInput } from "@/components/rating-status";
import { PredictForm } from "./predict-form";

export const dynamic = "force-dynamic";

type Slot = number | null;

interface PredictPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

export default async function PredictPage({ searchParams }: PredictPageProps) {
  const players = listPlayers();
  const sp = await searchParams;

  // 与 loadRatingView 同一解析口径：显式合法值优先，非法/缺省回 activeModel。
  const requested = typeof sp.rating === "string" ? sp.rating : undefined;
  const model: RatingModel =
    requested === "glicko2" || requested === "legacy"
      ? requested
      : (readRatingConfig()?.activeModel ?? "legacy");

  // glicko2：状态条与边界刷新取一次统一投影；legacy 保持原 recompute 注入。
  const glicko2View =
    model === "glicko2"
      ? loadRatingView({ rating: "glicko2" })
      : null;
  const ratings = model === "legacy" ? recomputeAllRatings() : null;

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
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-bold text-foreground">2v2 胜率预测</h1>
        <RatingModeControl current={model} />
      </div>
      {glicko2View ? (
        <div className="mb-4">
          <RatingBoundaryRefresh
            nextBoundary={glicko2View.nextBoundary}
          />
          <RatingStatus {...toRatingStatusInput(glicko2View)} />
        </div>
      ) : null}
      <PredictForm
        players={players}
        ratings={ratings}
        model={model}
        initialTeamA={initialTeamA}
        initialTeamB={initialTeamB}
      />
    </div>
  );
}
