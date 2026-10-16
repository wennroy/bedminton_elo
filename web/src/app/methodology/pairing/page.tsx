import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ArrowRight } from "lucide-react";
import { PairingSection } from "../pairing-section";

export const dynamic = "force-dynamic";

export default function MethodologyPairingPage() {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" asChild>
          <Link href="/schedule" aria-label="返回">
            <ChevronLeft className="size-5" />
          </Link>
        </Button>
        <h1 className="text-xl font-bold text-foreground">配对方式</h1>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">
        配对项目按出场公平、搭档与对手多样、实力接近排出共享的双打安排；人员变动时保留已打、续排待打。预测胜率所用的评分模型见计分方式说明。
      </p>
      <PairingSection />
      <section className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
        <p className="text-sm text-muted-foreground">想了解预测胜率怎么算？</p>
        <Link
          href="/methodology/glicko2"
          className="inline-flex shrink-0 items-center gap-1 text-xs font-bold text-card-foreground transition-colors hover:text-win"
        >
          查看计分方式说明
          <ArrowRight className="size-[15px]" strokeWidth={1.65} />
        </Link>
      </section>
    </div>
  );
}
