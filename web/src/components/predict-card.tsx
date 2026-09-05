import Link from "next/link";
import { ChevronRight } from "lucide-react";

export function PredictCard() {
  return (
    <Link
      href="/predict"
      className="flex items-center justify-between rounded-2xl border border-border bg-card p-4 shadow-sm transition-colors hover:bg-muted/30"
    >
      <div className="flex flex-col gap-1">
        <span className="font-medium text-card-foreground">
          🔮 胜率预测 · 任选 4 人，看看哪队更强
        </span>
      </div>
      <span className="flex items-center text-sm font-medium text-primary">
        去预测
        <ChevronRight className="size-4" />
      </span>
    </Link>
  );
}
