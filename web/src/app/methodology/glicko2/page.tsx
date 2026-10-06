import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ArrowRight } from "lucide-react";
import { Glicko2Section } from "../glicko2-section";

export const dynamic = "force-dynamic";

export default function MethodologyGlicko2Page() {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" asChild>
          <Link href="/" aria-label="返回">
            <ChevronLeft className="size-5" />
          </Link>
        </Button>
        <h1 className="text-xl font-bold text-foreground">计分方式 · 新版</h1>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">
        现行主计分是自定义的 Glicko-2
        双打扩展：周内逐场预估、每周一正式结算，分数自带不确定性。旧版 ELO 见页底链接。
      </p>
      <Glicko2Section />
      <section className="flex items-center justify-between gap-3 rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
        <p className="text-sm text-muted-foreground">还在看旧版分数？</p>
        <Link
          href="/methodology/legacy"
          className="inline-flex shrink-0 items-center gap-1 text-xs font-bold text-card-foreground transition-colors hover:text-win"
        >
          想看 Legacy ELO 计分方式
          <ArrowRight className="size-[15px]" strokeWidth={1.65} />
        </Link>
      </section>
    </div>
  );
}
