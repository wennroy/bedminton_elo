import Link from "next/link";
import { Formula, Section, SubsectionTitle, tableBodyCell, tableHeaderCell } from "./shared";

/**
 * 数值示例的每一个数字都由评分引擎（web/src/lib/ratings）按本页公式实算：
 * 甲 1060/112、乙 940/132 对 丙 1030/142、丁 990/161（RD 已含区段开始计入的
 * 一次过程方差，σ 均为 0.06），甲队 21:17 获胜。
 */
const EXAMPLE_ROWS = [
  {
    player: "甲",
    before: "1060 / 112",
    virtual: "1080",
    virtualRd: "252",
    expected: "0.4866",
    result: "胜",
    delta: "+16.82",
    after: "1076.8 / 110.7",
    win: true,
  },
  {
    player: "乙",
    before: "940 / 132",
    virtual: "960",
    virtualRd: "242",
    expected: "0.4866",
    result: "胜",
    delta: "+23.19",
    after: "963.2 / 129.6",
    win: true,
  },
  {
    player: "丙",
    before: "1030 / 142",
    virtual: "1010",
    virtualRd: "236",
    expected: "0.5135",
    result: "负",
    delta: "−26.75",
    after: "1003.3 / 139.0",
    win: false,
  },
  {
    player: "丁",
    before: "990 / 161",
    virtual: "970",
    virtualRd: "223",
    expected: "0.5136",
    result: "负",
    delta: "−34.61",
    after: "955.4 / 157.6",
    win: false,
  },
] as const;

export function Glicko2Section() {
  return (
    <Section
      title="现行：Glicko-2 双打"
      intro="每位选手的状态是三元组（r，RD，σ）：r 是实力估计，即页面展示的分数；RD（rating deviation）是不确定性的展示尺度，越大表示越没把握；σ 是内部坐标下以周为单位的波动性，描述实力本身的漂移速度。每场比赛只依据胜负更新——比分只决定谁赢，不进入分值。"
    >
      <div className="flex flex-col gap-2">
        <SubsectionTitle>现行参数</SubsectionTitle>
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className={tableHeaderCell}>参数</th>
              <th className={tableHeaderCell}>值</th>
              <th className={tableHeaderCell}>说明</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className={tableBodyCell}>新人初始 r</td>
              <td className={tableBodyCell}>1000</td>
              <td className={tableBodyCell}>展示分</td>
            </tr>
            <tr>
              <td className={tableBodyCell}>新人初始 RD</td>
              <td className={tableBodyCell}>180</td>
              <td className={tableBodyCell}>承认新人实力未知</td>
            </tr>
            <tr>
              <td className={tableBodyCell}>RD 下限</td>
              <td className={tableBodyCell}>60</td>
              <td className={tableBodyCell}>保留调整空间</td>
            </tr>
            <tr>
              <td className={tableBodyCell}>RD 上限</td>
              <td className={tableBodyCell}>250</td>
              <td className={tableBodyCell}>限制停赛后的不确定性</td>
            </tr>
            <tr>
              <td className={tableBodyCell}>初始波动性 σ</td>
              <td className={tableBodyCell}>0.06</td>
              <td className={tableBodyCell}>内部 x 坐标、每周单位</td>
            </tr>
            <tr>
              <td className={tableBodyCell}>波动性约束 τ</td>
              <td className={tableBodyCell}>0.3</td>
              <td className={tableBodyCell}>约束 σ 的更新幅度</td>
            </tr>
            <tr>
              <td className={tableBodyCell}>赛季相关</td>
              <td className={tableBodyCell}>900～1100 区间、保留率 0.75、季初 RD 下限 90</td>
              <td className={tableBodyCell}>
                仅列出备查，赛季制度详见
                <Link
                  href="/season"
                  className="underline underline-offset-2 transition-colors hover:text-foreground"
                >
                  赛季报页
                </Link>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>内部坐标</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          对外保留 1000 分中心；对内使用适配双打均分的坐标。1/2
          尺度同时作用于实力与不确定性，不能只缩分差而不缩不确定性：
        </p>
        <Formula>{`c = 173.7178
x = (r − 1000) / (2c)        φ = RD / (2c)
r = 1000 + 2c·x              RD = 2c·φ
σ 在 x 坐标下以周为单位保存，不与其他尺度的波动性混用`}</Formula>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>等效对手：搭档修正</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          更新 A1（搭档 A2，对手 B1、B2）时，把两名对手与搭档合成一个等效对手。搭档在均值公式中是减号，方差仍相加——其余三人的评分都有误差，他们的方差全部计入：
        </p>
        <Formula>{`x_virtual   = x_B1 + x_B2 − x_A2
φ_virtual = sqrt(φ_B1² + φ_B2² + φ_A2²)`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          等效对手只用于计算，不会被截断到真人分数的上下限，因此等效 RD
          可以超过真人的 RD 上限。例：1400＋800 对阵 1100＋1100，双方总实力相等，对外胜率恰为
          50%；更新 1400 分选手时等效对手是 1400，更新 800 分选手时是
          800——低分选手的获胜不会被单独记为爆冷。四人 RD
          相同时，两名队友的预估涨跌相同；RD 不同时则可以不同。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>对外胜率</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          预测一对互补概率，同时考虑四人的不确定性；不确定性越大，预测越向 50% 收缩：
        </p>
        <Formula>{`g(z) = 1 / sqrt(1 + 3z² / π²)
D = x_A1 + x_A2 − x_B1 − x_B2
U = sqrt(φ_A1² + φ_A2² + φ_B1² + φ_B2²)
P(A 胜) = logistic(g(U) · D)      logistic(z) = 1 / (1 + e^(−z))`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          个人更新里的期望胜率 E
          以等效对手为条件计算（自己的不确定性走更新公式，其余三人走等效对手），与对外
          P(A 胜) 口径不同。不能把四个 E 拼成展示胜率，也不能拿对外 P 替换核心内部的 E。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>结算节奏：周内 Estimated，周末 Final</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>区段：每周一 00:00（Asia/Shanghai）起算，左闭右开；比赛按实际日期归属，不按录入日期。</li>
          <li>
            周内：从区段开始的不可变正式基准出发逐场预估（Estimated）。区段开始只计入一次过程方差
            φ² ← φ² + h·σ²（h 为区段周数），随后冻结波动性，逐场顺序近似——不是每场重复一次完整的周更新。
          </li>
          <li>
            周末：从同一区段初基准出发，把整周比赛一次性批量跑标准
            Glicko-2，得到 Final，作为下一区段的正式基准；Estimated 只留作该周预估历史。
          </li>
          <li>
            周结算校准 = Final − 周末 Estimated，周一结算时单列展示。例如一周从 1000
            开始，周末预估 1020、正式结果 1017，显示「本周正式 +17；周结算校准
            −3」。校准可正可负，是结算事件，不伪装成某场输赢（交互示例，非回测结果）。
          </li>
          <li>无比赛的区段：实力与波动性不变，只增加不确定性（RD 封顶 250）。</li>
        </ul>
        <p className="text-sm leading-relaxed text-muted-foreground">
          周内冻结波动性下的单场更新（Estimated 路径）：
        </p>
        <Formula>{`g_i = g(φ_virtual)        E_i = logistic(g_i · (x_i − x_virtual))
I_i = g_i² · E_i · (1 − E_i)
G_i = g_i · (result_i − E_i)     result ∈ {1 胜, 0 负}
V_i = 1 / (1 / φ_i² + I_i)
x_i' = x_i + V_i · G_i           φ_i' = sqrt(V_i)`}</Formula>
        <p className="text-sm leading-relaxed text-muted-foreground">
          四人从同一份赛前快照同时写回；不能先更新 A
          队、再用变化后的状态更新 B 队。RD 边界在准备工作状态和每次更新后应用，均值更新使用未截断的
          V。
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>数值示例</SubsectionTitle>
        <p className="text-sm leading-relaxed text-muted-foreground">
          甲 1060／112、乙 940／132 对阵 丙 1030／142、丁
          990／161（分数／RD，RD 已含区段开始计入的一次过程方差，σ 均为
          0.06），甲队 21:17 获胜。表中数字由评分引擎按上述公式实算。
        </p>
        <p className="text-sm leading-relaxed text-muted-foreground">
          赛前对外胜率：D = (1060 + 940 − 1030 − 990) / 347.4356 ≈
          −0.0576，U ≈ 0.794，g(U) ≈ 0.916，P(A 胜) = logistic(−0.0527) ≈
          48.68%（乙队 51.32%）。
        </p>
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className={tableHeaderCell}>选手</th>
              <th className={tableHeaderCell}>赛前 r / RD</th>
              <th className={tableHeaderCell}>等效对手分</th>
              <th className={tableHeaderCell}>等效 RD</th>
              <th className={tableHeaderCell}>E</th>
              <th className={tableHeaderCell}>结果</th>
              <th className={tableHeaderCell}>Δr</th>
              <th className={tableHeaderCell}>赛后 r / RD</th>
            </tr>
          </thead>
          <tbody>
            {EXAMPLE_ROWS.map((row) => (
              <tr key={row.player}>
                <td className={tableBodyCell}>{row.player}</td>
                <td className={tableBodyCell}>{row.before}</td>
                <td className={tableBodyCell}>{row.virtual}</td>
                <td className={tableBodyCell}>{row.virtualRd}</td>
                <td className={tableBodyCell}>{row.expected}</td>
                <td className={tableBodyCell}>{row.result}</td>
                <td
                  className={`${tableBodyCell} font-num ${row.win ? "text-win" : "text-loss"}`}
                >
                  {row.delta}
                </td>
                <td className={tableBodyCell}>{row.after}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>乙的分数最低，但 RD 最大、调整空间最大，等效对手（960）又略高于自己，E 不足五成，因此获胜后涨得最多。</li>
          <li>等效 RD（223～252）超过真人 RD 上限 250 是正常的：三人方差相加，等效对手不做截断。</li>
          <li>
            四人 Δr 合计 −21.35 分，不为零：Glicko-2
            不保证零和，本扩展也不人为摊平，不保证多打就涨分。
          </li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>停赛、非零和与主榜</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>停赛不扣分：连续 t 周无比赛，φ² ← φ² + t·σ²，再应用 RD 上限；已有评分的缺席者不自动失去分数。</li>
          <li>主榜按实力均值 r 排序，不默认使用 r − k·RD，避免把不确定性增长悄悄变成扣分；RD 是可信程度提示，不是已验证严格校准的置信区间。</li>
          <li>不叠加连胜、出勤、净胜分等额外奖励；赛季软重置与周结算校准也不是出勤奖励。</li>
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <SubsectionTitle>口径与证据</SubsectionTitle>
        <ul className="flex flex-col gap-2 text-sm leading-relaxed text-muted-foreground">
          <li>
            本站采用的是自定义的 Glicko-2
            双打扩展：等效对手构造、双打尺度、周内顺序预估、周结算校准与赛季机制均为本项目设计，并非官方标准双打算法。
          </li>
          <li>
            真实历史回放（243 场双打、20 个样本周）：新版 Brier 0.2187 / log loss
            0.6269，优于同口径 naive Glicko-2 适配（0.2192 / 0.6291）与 legacy
            ELO（0.2318 / 0.6561）。这是单一俱乐部单一数据集上的回放结果，不是调参结论，不构成「已证明更准」。
          </li>
          <li>
            参考：Glicko-2
            <a
              href="https://www.glicko.net/glicko/glicko2.html"
              target="_blank"
              rel="noreferrer"
              className="underline underline-offset-2 transition-colors hover:text-foreground"
            >
              官方介绍
            </a>
            与
            <a
              href="https://www.glicko.net/glicko/glicko2.pdf"
              target="_blank"
              rel="noreferrer"
              className="underline underline-offset-2 transition-colors hover:text-foreground"
            >
              算法说明
            </a>
            （glicko.net）。
          </li>
        </ul>
      </div>
    </Section>
  );
}
