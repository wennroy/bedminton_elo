import { Section, SubsectionTitle } from "./shared";

export function TrueSkillSection() {
  return (
    <Section
      title="TrueSkill"
      intro="TrueSkill 不是本站的主计分，它承担两个辅助角色：旧版排行榜上的对照指标，以及配对调度的平衡配对工具。"
    >
      <div className="flex flex-col gap-2">
        <SubsectionTitle>参数与直觉</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          初始 μ = 25，初始 σ = 8.333，β = σ / 2，不设平局。一句话直觉：μ
          是实力估计，σ 是估计的不确定性，σ 越大表示系统对 μ
          越没把握；新人和久未参加者都从不确定状态开始收敛。
        </p>
      </div>
      <div className="flex flex-col gap-2">
        <SubsectionTitle>本站用在哪</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>旧版排行榜提供 ELO／TrueSkill 切换标签，用它与 ELO 互相对照。</li>
          <li>个人页展示每位选手的 μ／σ。</li>
          <li>配对调度（/schedule）按 μ／σ 构造胜率接近 50% 的对阵。</li>
          <li>
            它不随停赛时长增加不确定性，也不参与主榜与赛季结算——这两件事由现行
            Glicko-2 体系负责。
          </li>
        </ul>
      </div>
    </Section>
  );
}
