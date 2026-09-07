# COURTSIDE 生产 UI 重设计

> 状态：进行中

## 目标

按 `mock/`（COURTSIDE 原型）重设计生产站 UI：响应式外壳（桌面侧栏 / 手机 4 入口底栏 + 头像菜单）、完整浅/深双主题、总览（含每周名句）、球员目录新页、球员分析、记分、报名、全员趋势六大区域落地到 Next.js 生产页。未出原型的页面（配对/预测/周报/我的/管理/更新日志）保留功能、适配新主题不破版，并产出后续重设计 gap 清单。

## 非目标

- 配对、预测、周报、我的、管理、更新日志的逐页重设计（本轮只做主题适配；gap 清单见 T8 产出）。
- LLM 每周名句生成与数据库持久化（本轮静态句子池）。
- 新增 Playwright / e2e 测试设施；不改数据库 schema、不改 API 字段。
- 发版与部署（CHANGELOG 只写到 Unreleased，release 另行）。
- 把 mock 的虚构数据、原型脚本复制进生产。

## 关键决定

- 范围 → mock 全覆盖 6 区域 + 外壳；未原型页面仅主题适配；T8 产出 gap 清单文档作为后续重设计输入。
- 深色 → 完整双主题：手动切换持久化、首次跟随系统、首屏绘制前初始化防闪白、记分板两主题恒深色、同步 `color-scheme` 与 `theme-color`。
- 名句 → 静态原创句子池，按上海时区周一确定性轮换，支持往期浏览；不接 LLM / DB。
- 移动导航 → 底栏 4 入口（总览/球员/记一场/报名）；身份切换挪右上角头像；配对/周报/预测/我的/更新日志/管理收进头像菜单；桌面侧栏列全部入口。
- 令牌 → mock 色值映射进 `globals.css` 的 shadcn/Tailwind 语义变量，不另起 CSS 体系；组件继续用 Tailwind 类。
- 图表 → 继续用 Recharts，不复制原型 SVG。
- 缩放 → 移除 `maximumScale: 1` / `userScalable: false`，允许页面缩放。
- TrueSkill → 排行榜 ELO/TrueSkill 切换保留；球员页 TrueSkill μ/σ/区间、最长连胜、峰值移入「更多指标」展开区，不删。
- 身份存储沿用 `badminton:myPlayerId`；主题新键 `badminton:theme`。
- 每场 ELO delta 由服务端重放输出（扩展 `elo.ts`），不用日期快照差冒充单场变化。
- 字体 → 平台中文栈（PingFang SC 等）+ DIN 类数字栈，去掉 Geist，无外部字体请求。

## 假设

（空）

## 全局验收

`cd web && pnpm lint && pnpm test && pnpm build`

## Tasks

- [ ] T1 主题令牌 + 应用外壳 [顺序]
  - 改动：`web/src/app/globals.css`（改）、`web/src/app/layout.tsx`（改）、`web/src/components/app-shell.tsx`（新）、`web/src/components/bottom-nav.tsx`（删，并入 app-shell）
  - 要点：把 mock `styles.css` 的色值写进 shadcn 变量（light: bg `#f4f5f0`/surface `#fff`/ink `#242923`/accent `#d3f36b` 等，dark 对应见 `mock/DESIGN.md` 表格）；另加自定义变量 `--court`(记分板深底)、`--win/--loss`、`--team-a/--team-b`、`--chart`、`--series-1..8`（两主题各一套，值见 mock `community.css` 顶部），并在 `@theme inline` 映射成 Tailwind 色名。layout 去掉 Geist 与 `maximumScale`/`userScalable:false`，`themeColor` 用 light/dark 两个 media 值；`<head>` 内联主题初始化脚本（读 `badminton:theme`，无则跟随 `prefers-color-scheme`，首屏前设 `documentElement` 的 `dark` class——dark 变体现有 `@custom-variant dark (&:is(.dark *))` 已就位）。AppShell 为 client 组件：≥761px 文字/图标侧栏（全部入口：总览/球员/全员趋势/每周报名/记分 + 配对/周报/预测/更新日志/管理），≤760px 底栏 4 入口（中间「记一场」accent 突出）；顶栏含品牌、主题切换按钮（lucide Sun/Moon）、头像菜单（身份切换复用 IdentityPicker 弹层 + 上述次要入口链接）。图标用 lucide-react。主体左右 padding 与 `max-width` 按 mock 三档响应式（≤760/761–1190/≥1191，max 1480px）。
  - verify: `cd web && pnpm lint && pnpm build`；[人工] 三档宽度无横向溢出、两主题切换与刷新持久化、首屏不闪白、手机可双指缩放

- [ ] T2 总览首页重设计 [顺序]
  - 改动：`web/src/lib/quote.ts`（新）、`web/src/lib/quote.test.ts`（新）、`web/src/components/weekly-quote.tsx`（新，client）、`web/src/components/overview-summary.tsx`（新）、`web/src/app/page.tsx`（改）、`web/src/components/signup-card.tsx`（改）、`web/src/components/leaderboard.tsx`（改）、`web/src/components/predict-card.tsx`（改，仅主题适配）、`web/src/components/week-matches.tsx`（改）
  - 要点：结构按 mock `communityOverview`：顶部每周名句 → 个人摘要卡+本周报名卡并排 → 全员趋势(沿用现有 HomeTrend，T3 重构) → 排行榜 + 我的最近比赛双栏；底部保留 PredictCard 入口。`quote.ts` 接口 `quoteForWeek(date: Date, offsetWeeks = 0): { weekStart: string; dateLabel: string; text: string }`：上海时区（UTC+8，参照 mock `mondayOf` 算法）归一到周一，静态池（10 条左右原创短句，可抄 mock 6 条并补）按周序号取模；测试覆盖周一边界、同周稳定、offset 往期。weekly-quote 组件含上一周/下一周（当周禁前进）浏览，useState offset。个人摘要卡深色（`--court` 底）：姓名、俱乐部排名、ELO/胜率/本周全员场次三个大数字、「记一场比赛」CTA、「我的数据」链接；未选身份时显示选择身份引导（复用 IdentityPicker 触发）。排行榜改表格样式（#/球员/胜率/ELO/近一周涨跌），**保留 ELO/TrueSkill 分段切换**，中屏隐藏胜率列，当前身份行高亮（`is-me` 绿底）；数据沿用 page.tsx 现有 weekDelta/rankOf 计算。「我的最近比赛」按身份过滤（未选身份回退本周全部，沿用 week-matches 数据）。励志/装饰文案只允许出现在名句区，其余标题直述功能。
  - verify: `cd web && pnpm test -- src/lib/quote && pnpm lint && pnpm build`；[人工] 首页两主题 + 手机/桌面布局对照 mock

- [ ] T3 全员 ELO 趋势重构 [独立]
  - 改动：`web/src/components/home-trend.tsx`（重写）、`web/src/app/trends/page.tsx`（改）、`web/src/components/elo-chart.tsx`（删，/trends 改用重构后的组件）
  - 要点：按 mock `clubTrendPanel`：默认全员选中（不再默认只看自己）、周期 近4周/近12周/全部、模式 ELO积分/排名 双分段、成员 chips（色点+姓名+对勾，选中态不只靠颜色；全选/只看自己/清空；全不选显示明确空态不自动恢复）、颜色用 T1 的 `--series-*` 按球员稳定索引、浅深两套。右侧（手机图下两列）日期读数栏：hover/触碰/键盘聚焦某日 → 显示该日选中成员 ELO，按当日分数降序，排名模式显示名次（名次=全员当日名次，筛选不抬排名；纵轴名次 1 在上）；Recharts 实现可用自定义 tooltip/activeLabel 驱动读数栏。窗口起点沿用 cutoff 前最后一个快照（mock `clubDates` 逻辑：cutoff 日插入前值）。首页 compact 模式与 /trends 大图复用同一组件（prop 区分），/trends 页标题区按 mock `trendsPage`。排名同分口径与排行榜一致（ELO 降序，同分按 id 升序稳定）。
  - verify: `cd web && pnpm lint && pnpm test && pnpm build`；[人工] 成员选择/排名模式/日期读数/空态，两主题与三档宽度

- [ ] T4 球员目录 /players 新页 [独立]
  - 改动：`web/src/app/players/page.tsx`（新，server，`export const dynamic = "force-dynamic"`）、`web/src/components/player-directory.tsx`（新，client）
  - 要点：按 mock `playersPage`：搜索框（按姓名过滤，无结果保留搜索框+空提示）、排序分段（按 ELO/胜率/场次；排名定义恒按 ELO）、卡片网格（手机 2 列、桌面 3 列）：avatar、姓名（当前身份带「我」pill）、排名 `#NN`、场次·胜率、ELO 大数字、近 9 个快照 sparkline（小组件内联 SVG 或 Recharts 微型图）。server 页用 `buildStatsData()` 组装每卡 {id,name,elo,rank,total,winRate,sparkline: number[]}；排名同分口径与排行榜一致。routes.test.ts 会自动校验新页的 `force-dynamic`。
  - verify: `cd web && pnpm lint && pnpm test && pnpm build`；[人工] 搜索/排序/空态，对照 mock 卡片布局

- [ ] T5 球员分析页重设计 [独立]
  - 改动：`web/src/lib/elo.ts`（改，加 `computeMatchEloDeltas`）、`web/src/lib/stats.ts`（改：`PlayerMatchRecord` 加 `delta: number`；`playerFunStats` 加搭档/对手全量列表）、`web/src/lib/stats.test.ts`（改）、`web/src/lib/elo.test.ts`（改）、`web/src/app/players/[id]/page.tsx`（重写）、`web/src/components/player-trend.tsx`（新）、`web/src/components/player-relations.tsx`（新）、`web/src/components/player-match-history.tsx`（重写）、`web/src/components/fun-stats.tsx`（改/删，内容并入新区块）
  - 要点：结构按 mock `profile`（依次回答：现在如何→最近如何→和谁配合/交锋→具体发生了什么）：头部（大 avatar、姓名、俱乐部排名 badge、「切换球员」弹层复用 picker 交互）→ 指标条 4 格（当前 ELO+本周变化 pill、生涯胜率、累计出场、当前连胜/负）→ 个人趋势（player-trend：单人 Recharts，周期 4/12/全部只过滤图表不改生涯统计，cutoff 前最后 ELO 作区间起点，标注区间涨跌）→ 近期手感（近 8 场胜负点，左早右晚；近 8 场胜率、生涯场均净胜分双标签）→ 搭档与对手双面板（hero：搭档≥3 场胜率最高 / 对手≥3 场我方胜率最低，复用现有口径；各 top3 条形行；「全部」弹层列完整列表含 <3 场标注「样本较少」）→ 比赛记录（全部/获胜/失利筛选、每场显示 ELO delta、保留现有每页 10 场自动加载只调视觉；手机行布局：日期提行顶、delta 右上角）。TrueSkill μ/σ/区间、最长连胜、峰值及日期收进「更多指标」展开区。`computeMatchEloDeltas(matches): Record<number, number>[]`（每场 → 球员 string id → 该场 delta），复用 replay 结构按序对齐 `data.matches`；`playerMatches` 用其填 `delta`（不得用日期快照差）。stats.test.ts 补：单场 delta 四人守恒（总增量≈0 或按 K 验证符号）、搭档全量列表含样本数。
  - verify: `cd web && pnpm test -- src/lib/stats src/lib/elo && pnpm lint && pnpm build`；[人工] 球员页各区块对照 mock、弹层 Escape/焦点

- [ ] T6 记分页重设计 [独立]
  - 改动：`web/src/components/record-form.tsx`（重写）、`web/src/app/record/page.tsx`（小改）、`web/src/components/elo-delta-card.tsx`（改，融入成功弹层样式）
  - 要点：按 mock `recordPage` + `DESIGN.md §记分`：深色 court-board（两主题恒深色，`--court` 底）、A/B 队双列（A 队黄绿 `--team-a`、B 队浅蓝 `--team-b`，不代表胜负）、4 个阵容槽位（点槽开 picker 弹层：搜索、已占其他位置的禁用、可移除当前位）、比分大数字直接输入 0–99 + ±按钮（≥44px）、「交换两边」阵容比分一起换、「清空阵容」保留比分并提示、校验（四人未满/重复/平局/非法分数时禁用并说明原因）、确认弹层（双方阵容+比分，返回修改不写入）、提交中禁用防重复、成功弹层（四人前后积分与变化、「再记一场」保留阵容比分重置 21:0、「查看我的数据」进录入者档案）。**保留**：`/api/matches` 请求字段不变、`initialSlots` 配对预填、新增球员流程、未选身份时 IdentityPicker 自动弹出与 `enteredBy`、成功后 `router.refresh()`。桌面右侧说明栏（三步录入说明 + 我的最近一场），手机隐藏侧栏。
  - verify: `cd web && pnpm test -- src/app/api/matches && pnpm lint && pnpm build`；[人工] 记分全流程：选人→直输比分→换边→确认→积分反馈→再记一场，两主题下记分板均深色

- [ ] T7 报名页重设计 [独立]
  - 改动：`web/src/app/signup/page.tsx`（改）、`web/src/app/signup/signup-form.tsx`（重写）
  - 要点：按 mock `signupPage`：session-banner（大日期块 `--court` 底 + 场次信息 + 三个计数：参加总人数=partySize 总和、报名成员数=记录数、随行小伙伴=差值）、名单按报名先后（序号、成员可点进球员档案、「本人参加/携带 1 位小伙伴」、人数大数字）、操作栏（桌面与名单并排、手机置名单前）：身份行（当前身份+切换，复用 IdentityPicker）、「自己来/带一位小伙伴」选项卡（带对勾圆点）、未报名→确认报名；已报名→状态行 + 取消按钮（确认弹层：保留/确认取消）+「切换人数会自动更新报名」。**保留**：`/api/signups` POST/DELETE 字段、`getActiveSessionDate` 周三 20:00 切换、管理员移除他人报名按钮（`getAdminKey`）。空名单、未选身份（引导去头像菜单或弹 IdentityPicker）状态按 DESIGN.md 状态表。
  - verify: `cd web && pnpm test -- src/lib/signup && pnpm lint && pnpm build`；[人工] 报名/改人数/取消/身份切换/管理员移除，手机操作栏在名单前

- [ ] T8 收尾：未原型页适配 + gap 清单 + CHANGELOG [顺序]
  - 改动：`web/src/app/{me,admin,changelog,predict,schedule,weekly}/**`（仅清理硬编码颜色，如 `bg-white`、`emerald-*`、`rose-*`、`amber-*` → 语义变量/`-win`/`-loss`）、`web/src/components/{schedule-form,predict-form,weekly-view,collapsible-section}.tsx`（同）、`CHANGELOG.md`（Unreleased 追加）、`docs/ui-redesign-gap.md`（新）
  - 要点：全站 grep 硬编码色（`grep -rn "emerald-\\|rose-\\|amber-\\|bg-white" web/src`）逐个换语义类；逐页肉眼过两主题不破版。CHANGELOG 按根 AGENTS.md 写到 `## [Unreleased]` 的 `### What's New`（面向球友：新外观、深色模式、球员目录、每周名句、报名与趋势新界面等，不写技术细节）。`docs/ui-redesign-gap.md` 列后续重设计 gap：配对、预测、周报及分享图、管理、更新日志、我的页——各自现状、与 COURTSIDE 风格的差距、建议优先级与设计要点（参照 `mock/DESIGN.md` 的「下一轮扩展范围」）。最后跑全局验收并做全站人工巡检。
  - verify: `cd web && pnpm lint && pnpm test && pnpm build`；[人工] 全站两主题巡检（6 改造页 + 6 未原型页）

## 后续重设计 gap（初稿，T8 核实落档）

- 配对 `/schedule`：双队卡片 + 预填记分入口，A/B 队色与记分板一致。
- 预测 `/predict`：未原型；需设计与总览/球员页一致的指标排版。
- 周报 `/weekly` + OG 分享图：同数字排版/配色/细球场线，保留日期、俱乐部名、二维码。
- 管理 `/admin`：未原型。
- 更新日志 `/changelog`：继续直解析根 CHANGELOG，视觉适配。
- 我的 `/me`：头像菜单已承载身份切换，/me 页定位待重新设计（或并入菜单）。
