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
      <section id="guo" className="scroll-mt-6 space-y-4 rounded-2xl border border-primary/25 bg-card p-5 min-[761px]:p-[25px]">
        <div><span className="text-[10px] font-bold tracking-widest text-win">8 人专用</span><h2 className="mt-1 text-lg font-bold text-foreground">郭老师分配方法</h2></div>
        <p className="text-sm leading-relaxed text-muted-foreground">8 人按创建项目时的实力分由高到低分成 A、B 两组，每组 4 人。组内三场安排为「甲乙 对 丙丁」「甲丙 对 乙丁」「甲丁 对 乙丙」，每个人与其余三人各搭档一次。两组共六场为一轮；名单固定，六场全部录入比分后才能开下一轮。</p>
        <ol className="list-decimal space-y-2 pl-5 text-sm leading-relaxed text-muted-foreground">
          <li>第 1 轮：实力较高四人在 A 组，其余四人在 B 组。</li>
          <li>第 2 轮：按首轮排名，A 组后二与 B 组前二交换。</li>
          <li>第 3 轮：按第二轮排名，A 组前二与 B 组后二同场，其余四人另一场，让不同水平的球友一起玩。</li>
          <li>打完三轮可结束；时间充裕再开第 4 轮，按第三轮结果恢复 A 后二与 B 前二交换。第三轮起 A、B 只表示场地。</li>
        </ol>
        <div className="rounded-xl bg-muted/50 p-4 text-sm leading-relaxed text-muted-foreground">
          <p className="font-semibold text-foreground">每轮排名：胜场 → 净胜分 → 超预期分 → 固定抽签顺序</p>
          <p className="mt-2">每场队内两人各自获得相同战绩。净胜分是本轮各场「己方得分 − 对方得分」的合计；胜率是本轮胜场除以已录分场数。三场没录齐前仅显示暂列排名。各轮重新计分，只在本组内比较。</p>
          <p className="mt-2">超预期分 = 实际胜场 − 对应场次的预测胜率之和。例如三场预测胜率为 60%、40%、50%，实际赢了两场，超预期分就是 2 − 1.5 = +0.5；只在胜场、净胜分都相同时参与排序。预测取生成该轮时的实力分快照，整轮固定。</p>
          <p className="mt-2">以上仍完全相同，使用创建时保存的抽签顺序，刷新不会重新抽签。由于四个人参加同样三场，完整一轮的总得分与净胜分排序一致，不能再用总得分破解净胜分并列。</p>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">每轮分组与当时的排名依据都会保存。之后修改或撤回旧比分会更新该轮战绩，已经生成的后续轮保持原分组。项目记分与其他比赛一样进入全站实力分计算，轮内排名单独计算。</p>
      </section>
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
