# 预测页展示 ELO 预测变化
> 状态：原计划及 UAT 调整已完成（本地验证通过）

## 目标

/predict 预测结果从「只有一条 A 队胜率」升级为「胜率 + 每位球员的 ELO 预测变化」：
选好 4 人后，每人能看到这场赢了多少分、输扣多少分。结果面板重新设计，
胜率条与 ELO 变化融为一体，不再是胜率条下面硬接一张表。

## 非目标

- 配对页（智能配对结果里的每场「A 队胜率」行）不动
- 胜率本身的口径不动（继续用 predictElo 队均 vs 队均）
- TrueSkill 的 μ/σ 变化不做
- 分享图、周报等其他页面不动

## 关键决定

- 展示内容：每位球员两个数——「赢 +X / 输 −Y」。比分不影响 ELO（无净胜分加成），
  所以只有胜负两种结果，不需要让用户填预测比分
- delta 口径：必须与实际记分一致，用 replayMatches 的逐人公式
  （个人 ELO vs 对方队均分，K=16），**不是** predictElo 的队均口径——
  否则预测页显示 +12、真录完实际 +9，数字对不上
- 呈现：重设计结果面板。方向：双向胜率条（A 端 lime --team-a、B 端浅蓝 --team-b，
  两端标百分比），下方按队两列、每人一行「赢 +X / 输 −Y」，涨跌色用全站既有
  WIN/LOSS 令牌。先出 HTML 设计稿确认，再写代码（延续 mock/weekly-share.html 流程）
- 数字取整：Math.round 显示（与周报 ELO 涨跌榜一致）
- 范围：只改 /predict 页（用户已确认）

## 假设

（空）

## 全局验收

`cd /Users/wennroy/proj/bedminton_elo/web && pnpm test && pnpm build`

## Tasks

- [x] T1 设计稿：mock/predict-elo-change.html [顺序] [人工]
  - 改动：`mock/predict-elo-change.html`（新）
  - 要点：重设计「预测结果」面板：顶部双向胜率条（A 端 lime、B 端浅蓝，两端
    标百分比与队名），下方按队两列、每人一行「名字 + 赢 +X / 输 −Y」，
    涨绿跌红。复用 mock/styles.css 的 COURTSIDE 令牌。做两份预览：
    中文名 + 罗马字长名（taco knight / badminton boi），长名不撞数字。
    只做结果面板，不做选人区
  - verify: 人工——浏览器打开设计稿，确认版式后勾选

- [x] T2 elo.ts 导出 predictEloDeltas + 测试 [顺序]
  - 改动：`web/src/lib/elo.ts`（改）、`web/src/lib/elo.test.ts`（改）
  - 要点：`predictEloDeltas(a1, a2, b1, b2, ratings)` 返回
    `Record<string, { win: number; loss: number }>`。公式与 replayMatches
    循环体逐人那段完全一致：expected = 1/(1+10^((对方队均 − 个人)/400))，
    win = K*(1−expected)，loss = −K*expected。测试复用现有「predictElo 与
    重放一致性」套路：构造几场 golden 比赛，断言 predictEloDeltas 的输出
    等于 replayMatches 实放该场产生的 deltas
  - verify: `cd /Users/wennroy/proj/bedminton_elo/web && pnpm test -- src/lib/elo`

- [x] T3 predict-form.tsx 结果面板重写 [顺序]
  - 改动：`web/src/app/predict/predict-form.tsx`（改）
  - 要点：按 T1 确认稿实现结果面板；eloPrediction 旁调 predictEloDeltas，
    数字 Math.round；仍只在 ready（两队各 2 人）时显示；脚注文案保留
    「仅供参考」之意并说明数字为赢/输的 ELO 变化。组件不引新依赖
  - verify: `cd /Users/wennroy/proj/bedminton_elo/web && pnpm build`

- [x] T4 CHANGELOG 条目 [独立]
  - 改动：`CHANGELOG.md`（改）
  - 要点：[Unreleased] What's New 加一条，面向球友：「胜率预测升级：
    除了 A 队胜率，还能看到每位球员这场的 ELO 预测变化——赢多少、输多少」
  - verify: `cd /Users/wennroy/proj/bedminton_elo/web && pnpm test -- src/lib/changelog`


## UAT 调整（2026-09-15）

### 反馈与决定

- 用户反馈：滚动查看时看不到胜率；A/B 队不需要两份选人界面。
- 采用用户指定交互：点击上方四个选手框之一，再点击下方共用名单填入或替换。
  默认激活首个空位，填入后从当前位依次寻找下一个空位；选满后保留当前位。
- 已选球员在名单标出所属队伍并禁用，避免重复占位；替换释放原球员，清空重置四位。
- 阵容与简洁的双向胜率条一起吸顶，滚动名单时仍可查看胜率、切换目标位置。
  ELO 明细移到名单前，选满四人即显示。胜率公式、逐人分数公式保持不变。
- 本地 375×667 复查未发现页面最底部被导航完全遮挡；原布局的问题是结果位于两份
  完整名单之后。另发现 A 队胜率文字使用固定深色，改用支持深浅主题的颜色令牌。

### Tasks

- [x] T5 共用球员名单、四个可选择位置、单位置替换、防重复、自动补空位。
- [x] T6 阵容与胜率吸顶，ELO 明细前移，修正暗色模式胜率可读性。
- [x] T7 CHANGELOG 追加 UAT 调整。
- [x] T8 验证：手机 / 平板 / 桌面滚动、替换 / 清空 / URL 预填、暗色模式，
  并运行 `pnpm test`、`pnpm build` 及修改组件的 ESLint。

### 验证结果

- Chromium 浏览器：320×568、375×667、390×844、768×1024、1440×900；
  滚到底的胜率条在可视区域内、未被导航遮挡，无横向溢出。
- 交互：指定 B 队第二位后填人、自动补空位、选满显示胜率与 ELO、替换单人且保留
  其他位置、旧球员释放、防重复、清空、完整预填及重复/无效 URL 参数均通过。
- 检查深浅主题截图，无浏览器 JavaScript 错误。
- `pnpm test`：16 个测试文件，127 项全部通过。
- `pnpm build` 与 `pnpm exec eslint src/app/predict/predict-form.tsx`：通过。
