# 折线图 tooltip 排序 + 周报导出进度 + TrueSkill 显示优化 + 周期档位

> 状态：已完成(2026-09-02 发版 v1.2.0,master b9b7400)

## 目标

1. 首页和趋势页折线图 hover/点击时，tooltip 里的球员按**该时刻的分数从高到低**动态排序（现在是按球员 ID 写死的顺序），排名变动一眼可见。
2. 周报导出不再"点了没反应"：按钮变分阶段进度条，生成完成后页内弹窗预览图片，可下载/长按保存。
3. TrueSkill 排行榜主显示改为 μ（新人从 25 分起步，不再是 0 分），并展示不确定度 σ。
4. 首页趋势图周期档：近2周→近4周、近4周→近12周。

## 非目标

- 不改 ELO / TrueSkill 算法参数与公式（μ0=25、σ0=8.333、K=16 等都不动）
- 不改 predict 页胜率计算、不改分享图 OG 样式
- 不发版本号，只更新 CHANGELOG Unreleased
- 折线图其他交互（聚焦选人、全选、对比/排名切换）不变

## 关键决定

- **tooltip 排序范围**：首页 `home-trend.tsx`（对比+排名两种模式）和 `/trends` 页 `elo-chart.tsx` 都修（用户定）。排序方向：对比模式按 ELO 值降序；排名模式按名次升序（第 1 名在最上）。
- **周期档位**：`RANGES` 改为 近4周(28d) / 近12周(84d) / 全部；默认选中仍是 28d 档（改后标签为"近4周"，行为与现状一致）。
- **导出交互**（用户定）：点击后按钮进入 loading：分阶段**假进度条**（计算数据…→渲染图片…，缓动趋向 90%，完成才到 100%；真实进度需把 OG 接口改流式，改动不成比例，不做）。fetch 拿到 blob 后开页内 Dialog 预览 `<img>`，弹窗内提供"下载 PNG"按钮 + "长按图片保存"提示；关闭弹窗 revoke objectURL。**不用 window.open**（异步完成后调用会被浏览器弹窗拦截）。
- **TrueSkill 显示**（用户定）：主数字显示 μ（取整），按 μ 降序排序；副行显示 `±σ`。球员详情页 TrueSkill 卡片同步改：主显示 μ，小字 `σ x.x · 区间 [μ-3σ, μ+3σ]`。附带已接受行为：未参赛者 μ=25 起步会排在中间（σ=8.3 大，视觉上能解释"没打几场"）。

## 假设

（空）

## 全局验收

`cd web && pnpm test && pnpm build`

## Tasks

- [x] T1 首页趋势图：tooltip 动态排序 + 周期档位 [独立]
  - 改动：`web/src/components/home-trend.tsx`（改）
  - 要点：Tooltip 加 `itemSorter`，闭包读 mode：排名模式 `(item) => Number(item.value)`（升序），对比模式 `(item) => -Number(item.value)`（降序）。注意 `item.value` 可能为 undefined（connectNulls 缺口日的球员不进 payload，但防御性兜底：`?? Infinity` / 取负前判空，避免 NaN 破坏排序）。`RANGES` 改为 `{ key: "4w", label: "近4周", days: 28 }`、`{ key: "12w", label: "近12周", days: 84 }`、`全部`；`RangeKey` 类型同步；`useState<RangeKey>("4w")` 默认值不变。
  - verify: `cd web && pnpm exec tsc --noEmit` + `pnpm dev` 后首页 hover 两种模式各目视一次（找一个 ELO 顺序中途发生过反转的日期区间确认 tooltip 顺序跟着变）`[人工]`

- [x] T2 趋势页 EloChart：tooltip 动态排序 [独立]
  - 改动：`web/src/components/elo-chart.tsx`（改）
  - 要点：同 T1 的对比模式：`itemSorter` 按值降序。该组件无排名模式，一处即可。
  - verify: `cd web && pnpm exec tsc --noEmit` + `/trends` 页 hover 目视 `[人工]`

- [x] T3 周报导出：进度条 + 弹窗预览 [独立]
  - 改动：`web/src/app/weekly/weekly-view.tsx`（改）
  - 要点：替换 `window.open` 为 async `handleExport`：① setState 进入 exporting，按钮内容换成进度条（div width 过渡：立即 15%，再缓动到 90%）+ 阶段文案（"计算数据…"→"渲染图片…"），按钮 disabled；② `fetch(/api/og/weekly?week=...)` → `res.blob()` → `URL.createObjectURL`；③ 进度 100% 后打开 Dialog（用现有 `components/ui/dialog.tsx`）显示图片 + 下载按钮（`<a href={url} download="周报-2026-08-24.png">`）+ "手机可长按图片保存"提示；④ 弹窗关闭时 `URL.revokeObjectURL`、状态复位；⑤ `!res.ok` / fetch 异常 → 复位并在按钮下方显示错误文案。与 T1/T2/T4 无共同文件。
  - verify: `cd web && pnpm exec tsc --noEmit` + `pnpm dev` 打开 `/weekly?week=2026-08-24` 点导出：进度条动画→弹窗出图→下载/关闭复位，各目视确认 `[人工]`

- [x] T4 TrueSkill 显示 μ ± σ [独立]
  - 改动：`web/src/components/leaderboard.tsx`（改）、`web/src/app/players/[id]/page.tsx`（改）、`web/src/lib/stats.ts`（如需要，改）
  - 要点：leaderboard 的 TrueSkill tab：`tsScore` 改为主显示 `Math.round(mu)`、排序改 `b.mu - a.mu`，副行 `μ-3σ` 文案改为 `±{sigma.toFixed(1)}`；未参赛球员用 TS_MU/TS_SIGMA 兜底（显示 25 ±8.3）。球员页：`summary.tsScore` 处改显示 μ——`stats.ts` 的 `PlayerSummary` 需加 `mu`/`sigma` 字段（`tsPlayers` map 里已有数据；若 `tsScore` 没有其他使用者可顺手删掉，先 grep 确认）。卡片小字：`σ {sigma.toFixed(1)} · 区间 [{Math.round(mu-3*sigma)}, {Math.round(mu+3*sigma)}]`。ELO tab 一行不动。
  - verify: `cd web && pnpm test && pnpm exec tsc --noEmit` + 首页切 TrueSkill tab、任意球员页，目视确认 `[人工]`

- [x] T5 CHANGELOG 更新 [顺序]
  - 改动：`CHANGELOG.md`（改）
  - 要点：Unreleased 的 What's New 加条目：折线图 hover 按当前分数排序、周期档调整为近4周/近12周；周报导出加进度条与弹窗预览；TrueSkill 榜显示 μ 与不确定度 σ、新人 25 分起步。措辞面向球友。在 T1–T4 全完成后做。
  - verify: 目视 CHANGELOG 段落 `[人工]`
