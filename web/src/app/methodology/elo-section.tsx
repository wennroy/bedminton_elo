import { Formula, Section, SubsectionTitle } from "./shared";

export function EloSection() {
  return (
    <Section
      title="Legacy：ELO"
      intro="旧版计分保留作历史对照，继续按原公式计算。每位选手只有一个分数，初始 1000，单一 K 值 16；每场比赛结束即时更新。与现行体系一样，比分差不进入分值，比分只用来判定胜负。"
    >
      <div className="flex flex-col gap-2">
        <SubsectionTitle>公式</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          期望胜率（logistic 形式，400 分尺度）：
        </p>
        <Formula>{`E = 1 / (1 + 10^((R_opp − R) / 400))
delta = 16 · (s − E)
s ∈ {1 胜, 0 负}`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          预测口径与结算口径不同。预测（predictElo）用两队均分：
        </p>
        <Formula>{`teamAvg = (r1 + r2) / 2
d = avg_B − avg_A
P(A 胜) = 1 / (1 + 10^(d / 400))`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          结算（replayMatches）则是逐人对对方队均：每人的 expected
          按「个人分 vs 对方队均」计算，delta = 16 × (s −
          expected)，四人各自独立应用。因此搭档的分数不进入更新，预测与结算口径不一致，四人的变化之和也不保证为零。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>数值示例</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          1000＋1000 对阵 1100＋1100：队均 1000 对 1100，P(A 胜) = 1 / (1 +
          10^0.25) ≈ 36.0%。对 1000 分选手，对方队均 1100，E ≈ 0.360：获胜 +16
          × (1 − 0.360) ≈ +10.2，落败 −16 × 0.360 ≈ −5.8。对 1100
          分选手，对方队均 1000，E ≈ 0.640：获胜 ≈ +5.8，落败 ≈
          −10.2。同一场里，预测用的是队均口径，落分用的却是个人对对方队均口径——两套口径不要混用。
        </p>
      </div>
    </Section>
  );
}
