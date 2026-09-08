# 周报分享图 COURTSIDE 化

> 状态：进行中

## 目标

把 `/api/og/weekly` 生成的周报分享图（1080×1920 竖版 PNG）从旧的「白卡+领奖台+渐变装饰」风格重做为 COURTSIDE 浅色网页风：#f4f5f0 底、大数字 + 细分隔线榜行、lime accent 小面积点缀，与网站新 UI 同源。内容板块一项不少：三榜（出勤/战绩王/ELO 涨跌 top3）、最佳组合、本周趣闻、日期 + 俱乐部名 + 二维码。

## 非目标

- /weekly 网页版式重做（已在 T8 主题适配，页内榜单重排留待后续）。
- 图片尺寸（保持 1080×1920）、缓存/ETag 机制（no-cache + weeklyDataVersion 不动）、QR 生成逻辑。
- 外部字体文件引入（ImageResponse fonts 选项）。
- 发版与部署（CHANGELOG 只写到 Unreleased）。

## 关键决定

- 色调 → 浅色网页风：底 `#f4f5f0`、墨色 `#242923`、次文字 `#72796f`、分隔线 `#e2e5dc`、accent lime `#d3f36b`（上文字 `#263411`）；涨跌用 `--win #4e6c1d` / `--loss #ae5548`；全部字面值取自 `web/src/app/globals.css` 浅色组。
- 版式 → 允许重排（按 docs/ui-redesign-gap.md 建议）：大数字 + 细分隔线榜行替代领奖台柱与白卡片；趣闻改单列数据行；奖牌色若保留仅限名次徽标小面积。内容项不变。
- 设计先行 → T1 先在 mock/ 出静态 HTML 设计稿，用户确认后 T2 才翻译进 Satori；不直接盲改 route.tsx。
- Satori 约束 → 不解析 CSS 变量，所有令牌以字面值内联；元素必须显式 `display:flex`；领奖台式多列禁用 `flex:1`（历史坑：子树布局坍塌），用固定 width；linear-gradient/box-shadow/emoji 可用（现版已在用）。
- 字体 → 沿用 route.tsx 现有 FONT_FAMILY 栈（容器无 DIN 类字体），「大数字」观感靠字号/字重/letterSpacing，不引外部字体。
- 截断 → 保留现有 `truncate()` 显示宽度截断（CJK=2/ASCII=1）；设计稿与验收都必须包含罗马字长名用例（线上最长约 11 字符，如 "taco knight"），本地中文名 dummy 数据测不出来。
- 长名/空态 → 本周无比赛、趣闻缺项、榜单不足 3 人时版式不塌（现版已有空态分支，重排后保留等价处理）。

## 假设

（空）

## 全局验收

`cd web && pnpm lint && pnpm test && pnpm build`

## Tasks

- [x] T1 分享图设计稿（静态 HTML） [顺序]
  - 改动：`mock/weekly-share.html`（新）
  - 要点：按 1080×1920 等比（浏览器内可 scale 预览）画出确认后可直接翻译的版式：头部（🏸/俱乐部名「卷技术小分队」+ 第 N 周战报大标题 + 日期区间，lime accent 点缀）→ 三榜各 top3（大数字榜行：名次徽标 + 姓名 + 大数字，行间细分隔线；ELO 榜涨跌带 win/loss 色）→ 最佳组合横幅（可沿用深色 `--court #262f29` 横条作视觉锚点，与网页摘要卡/记分板同源）→ 本周趣闻单列数据行（🎯胶着/💥惨案/🔥连胜王/😱冷门，胜者先行比分大行 + 一行说明）→ 底部（QR 占位方块 + 「扫码查看完整排行榜」+ 域名 + 俱乐部副标题）。可 `<link>` 复用 `mock/styles.css` 的令牌变量；布局只用 flex（Satori 子集），不用 grid。设计稿里放两套数据示例：中文名 + 罗马字长名（11 字符级）各一组，确认截断与换行表现。
  - verify: [人工] 浏览器打开 `mock/weekly-share.html`，用户确认整体版式与细节（可当场改稿迭代）

- [x] T2 route.tsx 按设计稿重写 [顺序]
  - 改动：`web/src/app/api/og/weekly/route.tsx`（重写）
  - 要点：把 T1 确认稿逐块翻译成 Satori JSX，颜色/字号/间距全部字面值内联（对照 globals.css 浅色组，不用 var()）。保留不动：`truncate()`、ETag/no-cache/304 短路、`weeklyDataVersion`、QRCode.toDataURL、GET 参数校验、1080×1920。多列结构用固定 width 不用 flex:1；每个含子元素的 div 显式 display:flex。趣闻行沿用「胜者先行 + 比分」两行结构改单列数据行样式。`route.test.ts` 只断言状态码/headers/ETag，不应需要改；若设计微调导致断像素级假设（没有这类断言）再说。本地审图方式：`pnpm dev` 后访问 `/api/og/weekly?week=2026-09-01`（本地 dummy 库上周约 18 场，三榜+趣闻齐全）与 `?week=2026-09-07`（本周 6 场）各存一张 PNG。
  - verify: `cd web && pnpm test -- src/app/api/og/weekly && pnpm lint && pnpm build`；[人工] 两张本地 PNG 对照设计稿逐块核对（含罗马字长名截断——可临时把 dummy 球员改名或用长名用例核对）

- [ ] T3 CHANGELOG + 收尾 [顺序]
  - 改动：`CHANGELOG.md`（Unreleased 追加）
  - 要点：What's New 加一条面向球友的描述（如「周报分享图换上新外观：与网站同款排版配色，榜单数字更大更清爽」），不写技术细节。跑全局验收。
  - verify: `cd web && pnpm lint && pnpm test && pnpm build`；[人工] 最终 PNG 全要素检查：日期、俱乐部名、二维码、四个趣闻、三榜满员、最佳组合齐全的一张完整图
