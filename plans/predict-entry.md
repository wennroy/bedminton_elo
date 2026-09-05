# 胜率预测入口与口径统一

> 状态：进行中

## 目标

`/predict` 页面已完整实现"选 2v2 看胜率"，但全站没有任何入口（幽灵页面）。
本次为它加两处入口（首页卡片 + 配对页每场联动），并把全站用户可见的
胜率数字统一为 ELO 单口径。做完后：用户从首页或配对页能自然进入预测页，
看到的胜率数字和排行榜的 ELO 分是同一套逻辑。

## 非目标

- 不重做预测页的选人交互（维持两队点选）
- 不动配对算法本身（scheduler 内部仍用 TrueSkill 做平衡，仅展示口径变）
- 底部导航不加项（5 位已满）
- 不加 1v1 预测、不在结果区加 h2h 历史交锋
- 不动 trends/profiles 页的 TrueSkill μ±σ 展示（那是评分展示，非胜率）

## 关键决定

- 复用现有 `/predict` 页面，只加入口和打磨，推倒重做被否决
- 入口两处：首页卡片（仿 SignupCard 模式）+ 配对页每场「胜率预测」链接
- 预测结果区只留 ELO 单模型，删掉 TrueSkill 胜率条
- 配对页卡片上的胜率数字连带改为 ELO 口径——全站用户可见胜率只有一个数
- 预填参数沿用 `/record` 的既有惯例：`?pa1=&pa2=&pb1=&pb2=`

## 假设

- 首页卡片位置：SignupCard 之后、HomeTrend 之前（SignupCard 是时效性内容，优先）
- 非法预填参数（id 不存在/重复）的容错：对应槽位置空，不整页报错

## 全局验收

`pnpm -C web lint && pnpm -C web test && pnpm -C web build`

## Tasks

- [x] T1 预测页：URL 预填 + 结果区单 ELO 化 [顺序]
  - 改动：`web/src/app/predict/page.tsx`（改）、`web/src/app/predict/predict-form.tsx`（改）
  - 要点：page.tsx 照抄 `record/page.tsx` 的 searchParams 模式（Next 15 中
    searchParams 是 Promise，要 await；页面保持 `force-dynamic`）。解析
    pa1/pa2/pb1/pb2，只接受存在于 players 且互不重复的 id，非法值对应槽位置空。
    PredictForm 加可选 props `initialTeamA`/`initialTeamB` 初始化 useState。
    删除 tsPrediction 与 TrueSkill 的 PredictionBar，结果区只留 ELO 一条，
    清掉 trueskill import；底部文案「基于当前双评分系统计算」改为 ELO 单口径。
  - verify: `pnpm -C web lint src/app/predict && pnpm -C web test`
    ＋[人工] `/predict?pa1=1&pa2=2&pb1=3&pb2=4` 预填正确且只显示一条胜率；
    `/predict?pa1=999` 不报错

- [x] T2 首页加预测卡片 [独立]
  - 改动：`web/src/components/predict-card.tsx`（新）、`web/src/app/page.tsx`（改）
  - 要点：仿 `signup-card.tsx`——静态 Link 卡片、无数据依赖，文案如
    「🔮 胜率预测 · 任选 4 人，看看哪队更强」+ 右侧「去预测 ›」。
    在 page.tsx 中插在 `<SignupCard />` 之后、`<HomeTrend />` 之前。
  - verify: `pnpm -C web lint src/components/predict-card.tsx src/app/page.tsx && pnpm -C web test`

- [x] T3 配对结果每场加「胜率预测」入口 [顺序]
  - 改动：`web/src/app/schedule/schedule-form.tsx`（改）
  - 要点：现结果卡片整体是 `<Link href=/record?...>`，HTML 不允许 `<a>` 嵌套
    `<a>`——把外层 Link 改为 div，卡片主体区域保留 Link→`/record`（记分），
    胜率行「A 队胜率 x% ›」单独包成 Link→
    `/predict?pa1=${a1}&pa2=${a2}&pb1=${b1}&pb2=${b2}`。视觉保持现有卡片样式，
    胜率行加 hover 态表明可点。依赖 T1 先落地（否则链接过去无预填）。
  - verify: `pnpm -C web lint src/app/schedule/schedule-form.tsx && pnpm -C web test`
    ＋[人工] 生成配对后点胜率行进 `/predict` 且 4 人预填正确；点卡片主体仍进 `/record`

- [x] T4 schedule API 胜率改 ELO 口径 [独立]
  - 改动：`web/src/app/api/schedule/route.ts`（改）、`web/src/app/api/schedule/route.test.ts`（新）
  - 要点：`winRate` 改用 `predictElo(a1, a2, b1, b2, eloRatings).teamAWin`；
    eloRatings 从同一个 `recomputeAllRatings()` 结果构造 `Record<string, number>`
    （key 为 `String(id)`，缺省 1000）。`createPlayer` 仍被 optimizeSchedule
    使用故保留，删 `predictTeamOutcomeWin` import。新测试参照
    `api/players/route.test.ts` 模式（DATABASE_URL 指向临时库 + closeDb +
    describe.sequential），先造球员和比赛，再断言返回的 winRate 与
    predictElo 计算值一致。
  - verify: `pnpm -C web test -- schedule`

- [ ] T5 全站终验与走查 [顺序]
  - 改动：无（仅验证）
  - 要点：跑全局验收命令；手动走查清单：首页卡片→/predict；配对页生成→
    点胜率行→/predict 预填；预填后换队/清空正常；配对页胜率数字与 /predict
    同阵容数字一致（都是 ELO 口径）。
  - verify: `pnpm -C web lint && pnpm -C web test && pnpm -C web build` ＋[人工] 走查清单全过

- [ ] T6 更新 CHANGELOG.md [顺序]
  - 改动：`CHANGELOG.md`（仓库根目录，改）
  - 要点：按既有格式补 Unreleased 条目：预测页入口（首页卡片 + 配对页联动）、
    全站胜率统一 ELO 口径。不 bump 版本号（release 脚本负责）。
  - verify: `pnpm -C web test`（changelog.test.ts 校验格式）
