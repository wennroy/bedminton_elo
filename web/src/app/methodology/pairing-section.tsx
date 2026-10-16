import Link from "next/link";
import { Formula, Section, SubsectionTitle } from "./shared";

export function PairingSection() {
  return (
    <Section
      title="现行：quota-local-v1 配对"
      intro="配对项目把「接下来几场谁和谁打」排成一份共享安排：保存在服务器上，所有人看到同一份数据，可以查看待打场次、标记已打、录入比分。人员临时变动时按新名单续排，已打的比赛原样保留。"
    >
      <div className="flex flex-col gap-2">
        <SubsectionTitle>场次状态</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>
            待打：生成后等待开打，可自由决定开打顺序；续排时只有待打场次会被替换。
          </li>
          <li>
            已打待录分：打完后由在场任何人「标记已打」，等待录入比分；续排时原样保留。
          </li>
          <li>已登记：比分已录入并计入评分；续排时同样原样保留。</li>
          <li>已被替换：续排时被换下的待打场次，只读存档，不再开打。</li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>出场公平</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          排阵的第一目标是出场均衡：每个人在本项目里的总出场（已打 +
          新安排）尽量接近。生成时把全部名额逐个分给「当前总出场最少」的人，并列时随机决定；名额不能整除时，有人多打一场是正常的。
        </p>
        <Formula>{`总名额 S = 4M，N 名球员
q = ⌊S / N⌋，r = S mod N
最均衡：r 人打 q + 1 场，其余打 q 场
出场方差下限 σ² = (r / N)·(1 − r / N)

例：N = 10、M = 8 → S = 32，q = 3、r = 2
→ 2 人打 4 场、8 人打 3 场，σ² 下限 = 0.16`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          r = 0 时方差可以为 0；r ≠ 0 时这个下限是数学最优，再巧妙的排法也无法让所有人场数完全相同。历史差距无法靠有限场数完全消除时不强行拉平——均衡是「从这一轮开始尽量公平」，不保证每个人都补到完全相同的场数。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>人员变动与续排</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>
            续排 =
            保留已打（已打待录分与已登记），按新名单重排待打；提交前页面会现算提示「保留
            X 场已打比赛，替换 Y 场待打比赛」。
          </li>
          <li>
            新人从加入时开始算公平：加入前别人已打的场次不算他落后，他的起点取继续在场者中的最低出场，不补旧账。
          </li>
          <li>
            暂时离开的人保留已打历史；重新加入时同样按当前最低起点继续，不追究离开期间的场次。
          </li>
          <li>
            名单不足 4
            人时可以先只保存名单、暂不安排比赛；不会删除球员，也不会删除任何已登记的历史。
          </li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>「减少已打对阵重复」</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          默认开启。它只统计本项目里已经打过的完整对阵（两边搭档组合完全相同才算同一场），排阵时优先安排没打过的组合。
        </p>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>这是软偏好而不是硬禁令：场数较多或人数较少时，必要的重复仍会发生。</li>
          <li>只影响未来的待打场次，不会隐藏或改动已打分组里的任何历史。</li>
          <li>勾选状态按项目记忆：续排时默认沿用上一次生成时的选择。</li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>配对质量</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          在出场均衡的前提下，排阵按以下偏好挑选方案（均为偏好，不是硬约束）：
        </p>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>搭档多样：尽量避免本批内反复与同一搭档组队。</li>
          <li>对手多样：尽量避免本批内反复遇到同一批对手。</li>
          <li>对阵不重复：同一套对阵组合（两边搭档完全相同）在本批内尽量不出现第二次。</li>
          <li>
            实力接近：按生成时的评分预测胜率，优先选择接近五五开的对阵；一方预测胜率超过
            70% 视为明显一边倒，这类方案排在后面、能免则免。
          </li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>配额怎么分：最小化平方和</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          把「尽量均衡」写成一个最小化问题：cᵢ 是有效出场（加入项目后的已打 +
          已承诺），xᵢ 是本批要新安排的场数：
        </p>
        <Formula>{`min  F = Σᵢ (cᵢ + xᵢ)²
约束 0 ≤ xᵢ ≤ M，Σᵢ xᵢ = 4M，每场四人互异`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          F 是凸函数且可按人拆分，所以逐名额贪心就是全局最优：每个名额都给当前边际代价
          ΔFᵢ = 2(cᵢ + xᵢ) + 1 最小的人，并列时随机打破。新人的 cᵢ
          取继续在场者的最小值，因此不会被加入前的旧账压后。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>对阵怎么挑：质量分 J</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          在配额确定的方案之间，用下面的分量比较优劣（C(n, 2) = n(n−1)/2
          表示重复对数，所有分量越小越好）：
        </p>
        <Formula>{`每场 m 预测 A 队胜率 p_m，d_m = 2·|p_m − 0.5|
B = mean(d_m²)                  实力差
P = Σ C(搭档次数, 2) / (2M)     本批搭档重复
O = Σ C(对手次数, 2) / (4M)     本批对手重复
E = Σ C(对阵次数, 2) / M        本批完整对阵重复
H = Σ 已打次数 × 本批次数 / M   已打对阵复现（勾选时计入）
J = 4·B + 2·P + 0.5·O + 2·E + 3·H`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          比较方案时先看明显失衡的场数（d_m &gt; 0.4，即一方胜率超过
          70%）、再看总超出幅度 Σ max(0, d_m −
          0.4)，最后才比 J——前两级只为把一边倒压到最少。「减少已打对阵重复」关闭时
          H 的权重取 0，其余分量始终生效。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>生成时胜率快照与可复现</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>
            生成对阵时会截取本项目名单内球员的当时评分存进项目，待打卡片上的「生成时预测 A
            队胜率」按这份快照计算，不随后续比赛变化；想看按最新分数的预测，用卡片上的「当前预测」链接。
          </li>
          <li>
            创建项目时记下当前生效的评分模型（新版 Glicko-2 或旧版
            ELO），此后该项目的预测胜率都按同一模型计算，续排也不会变；计算口径见
            <Link
              href="/methodology/glicko2"
              className="underline underline-offset-2 transition-colors hover:text-foreground"
            >
              计分方式说明
            </Link>
            。
          </li>
          <li>
            算法版本固定为
            quota-local-v1：同样的名单、场数与历史输入，排阵口径一致。高级选项里的随机种子留空则由服务器随机取新值；填入同一种子可复现同一份排阵结果，用于核对，不影响公平性。
          </li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>记分、防重复与撤回</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>
            从配对场次记分与手动记分使用同一套比分校验与评分更新，登记后正常计入排名。
          </li>
          <li>
            同一场只能登记一次：重复提交会被识别并拒绝，不会产生重复的比赛记录。
          </li>
          <li>
            撤回一场已登记的比赛会把它从评分中移除，同时把配对场次退回「已打待录分」，之后可以重新录入正确比分。
          </li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>口径</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>
            出场均衡与各项配对偏好只使用本项目内部的数据，不影响评分；评分只来自已登记的比赛。
          </li>
          <li>
            预测胜率仅供排阵参考，不是任何一方的承诺或让分依据；实际交手结果仍以登记的比分为准。
          </li>
        </ul>
      </div>
    </Section>
  );
}
