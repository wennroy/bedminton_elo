import Link from "next/link";
import { ArrowRight } from "lucide-react";

export function PredictCard() {
  return (
    <Link
      href="/predict"
      className="flex items-center justify-between gap-4 rounded-2xl border border-border bg-card p-5 transition-colors hover:bg-accent"
    >
      <div>
        <h2 className="text-base font-semibold text-card-foreground">
          胜率预测
        </h2>
        <p className="mt-1 text-xs text-muted-foreground">
          任选四位球员分边，看看哪队更强
        </p>
      </div>
      <span className="flex shrink-0 items-center gap-1 text-xs text-muted-foreground">
        去预测
        <ArrowRight className="size-4" strokeWidth={1.65} />
      </span>
    </Link>
  );
}
