import { Section, tableBodyCell, tableHeaderCell } from "./shared";

/** 两套计分的速览对照：初值、更新时机、零和性、不确定性。 */
export function MethodologyComparisonTable() {
  const rows: readonly { dimension: string; glicko2: string; elo: string }[] = [
    {
      dimension: "初始值",
      glicko2: "1000 分，RD 180，σ 0.06",
      elo: "1000 分（单一分数）",
    },
    {
      dimension: "更新时机",
      glicko2:
        "周内逐场预估（Estimated）；每周一 00:00（Asia/Shanghai）切区段，周末从区段初基准批量正式结算（Final）",
      elo: "每场结束即时更新",
    },
    {
      dimension: "是否零和",
      glicko2: "不保证；四人之和可正可负，不人为摊平",
      elo: "不保证",
    },
    {
      dimension: "是否含不确定性",
      glicko2: "有：RD（展示尺度）与 σ（内部每周波动性）",
      elo: "无",
    },
    {
      dimension: "比分差",
      glicko2: "不进入分值，只定胜负",
      elo: "同左",
    },
    {
      dimension: "现状",
      glicko2: "主计分",
      elo: "保留对照，继续按原公式计算",
    },
  ];

  return (
    <Section title="速览对照">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className={tableHeaderCell}>维度</th>
            <th className={tableHeaderCell}>现行：Glicko-2 双打</th>
            <th className={tableHeaderCell}>Legacy：ELO</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.dimension}>
              <td className={tableBodyCell}>{row.dimension}</td>
              <td className={tableBodyCell}>{row.glicko2}</td>
              <td className={tableBodyCell}>{row.elo}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}
