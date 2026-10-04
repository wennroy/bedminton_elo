import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";
import { MethodologyComparisonTable } from "./comparison-table";
import { Glicko2Section } from "./glicko2-section";
import { EloSection } from "./elo-section";
import { TrueSkillSection } from "./trueskill-section";

export const dynamic = "force-dynamic";

export default function MethodologyPage() {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" asChild>
          <Link href="/" aria-label="返回">
            <ChevronLeft className="size-5" />
          </Link>
        </Button>
        <h1 className="text-xl font-bold text-foreground">计分方式说明</h1>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">
        本站先后使用过多套评分。现行主计分是自定义的 Glicko-2
        双打扩展；Legacy ELO 按原公式继续计算，作为历史对照；TrueSkill
        只作对照指标与配对调度工具。本页说明三者的口径、公式和数值示例；赛季制度不在本页展开，详见
        <Link
          href="/season"
          className="underline underline-offset-2 transition-colors hover:text-foreground"
        >
          赛季报页
        </Link>
        。
      </p>
      <MethodologyComparisonTable />
      <Glicko2Section />
      <EloSection />
      <TrueSkillSection />
    </div>
  );
}
