import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ArrowRight } from "lucide-react";
import { EloSection } from "../elo-section";
import { TrueSkillSection } from "../trueskill-section";

export const dynamic = "force-dynamic";

export default function MethodologyLegacyPage() {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" asChild>
          <Link href="/" aria-label="返回">
            <ChevronLeft className="size-5" />
          </Link>
        </Button>
        <h1 className="text-xl font-bold text-foreground">计分方式 · Legacy</h1>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">
        Legacy 保留旧版 ELO 计分作为历史对照，继续按原公式计算；TrueSkill
        只作对照指标与配对调度工具。以下说明两者的口径与公式；想看现行新版见页底链接。
      </p>
      <EloSection />
      <TrueSkillSection />
      <section className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
        <p className="text-sm text-muted-foreground">现行计分已升级。</p>
        <Link
          href="/methodology/glicko2"
          className="inline-flex shrink-0 items-center gap-1 text-xs font-bold text-card-foreground transition-colors hover:text-win"
        >
          想看新版 Glicko-2 计分方式
          <ArrowRight className="size-[15px]" strokeWidth={1.65} />
        </Link>
      </section>
    </div>
  );
}
