# What's New 时间线 + 体验优化

> 状态：进行中

## 目标

建立 CHANGELOG/版本号机制并把「What's New」搬上网页(时间线展示每个版本的新功能);同时完成四项体验优化:本周战绩补全时间信息、个人页参赛历史渐进加载、个人页趣味数据、配对结果按用户持久化且可一键跳转记分。

## 非目标

- 不做服务端配对持久化 / 跨设备同步(本机 localStorage 即可)
- 不加底部导航 tab(更新日志入口放「我的」页)
- 参赛历史不做 API 分页(单球员场次量级小,客户端渐进展示足够)
- 本次不实际发版:不改 Unreleased 标题、不打 tag、不动 package.json version
- 不改 ELO / TrueSkill 算法,不动 admin 页

## 关键决定

- Base 分支:origin/master(远端没有 main 分支,用户说的 origin/main 即 master)。新分支 `whats-new-ux`,在当前 worktree 直接 `git checkout -b whats-new-ux origin/master`(工作区干净)
- 版本起点:追记 `[1.0.0] - 2026-08-31`(上线日)总结已有功能,其上是 `[Unreleased]` 段
- CHANGELOG 单一数据源 = 仓库根目录 `CHANGELOG.md`(Keep a Changelog 格式 + SemVer);网页运行时 fs 读取解析,不另建数据源;Docker 用 volume 把根目录文件挂进容器,不改 build context
- 每个版本段内含 `### What's New` 小节,面向球友的自然语言 bullets —— 网页时间线只渲染这一部分
- 更新日志页 = `/changelog`,入口卡片放「我的」页(与「管理员登录」并列)
- 趣味数据四项全做:当前/最长连胜、黄金搭档(搭档≥3 场中胜率最高)、头号克星(交手≥3 场中胜率最低)、ELO 历史峰值 + 场均净胜分
- 配对持久化:localStorage,key = `badminton:schedule:<playerId>`(未选身份用 `anon`);生成成功写入,「清空」或再次「生成配对」才覆盖
- 配对场次卡片可点击 → 跳 `/record?pa1=..&pa2=..&pb1=..&pb2=..` 预填阵容

## 假设

(无 —— 全部决策已锁定)

## 全局验收

`pnpm --dir web test && pnpm --dir web build`

## Tasks

- [x] T1 建分支 + CHANGELOG.md + 根目录 AGENTS.md [顺序]
  - 改动:`CHANGELOG.md`(新)、`AGENTS.md`(新,仓库根目录)
  - 要点:先 `git checkout -b whats-new-ux origin/master`。CHANGELOG 用 Keep a Changelog 格式:`# Changelog` 头 + 格式说明;`## [Unreleased]` 下放空 `### What's New`(注释提示发版前填写);`## [1.0.0] - 2026-08-31` 的 What's New 用自然语言总结已上线功能(ELO 排行榜与走势、快速记分即时涨跌、智能配对、每周战报分享图、个人主页、身份选择、人人可加球员)。AGENTS.md 记录:SemVer 约定;开发期改动记 Unreleased、发版时改标题为 `[x.y.z] - 日期` 并补 What's New;What's New 面向球友不写技术细节;网页 /changelog 自动解析本文件,勿另建数据源;发版时同步 `web/package.json` 的 version。注意与 `web/AGENTS.md`(next dev 自动生成)区分,提一句即可。
  - verify: `grep -q "## \[Unreleased\]" CHANGELOG.md && grep -q "## \[1.0.0\] - 2026-08-31" CHANGELOG.md && grep -q "What's New" AGENTS.md && echo OK`

- [x] T2 /changelog 时间线页 + 「我的」页入口 + compose 挂载 [顺序]
  - 改动:`web/src/lib/changelog.ts`(新)、`web/src/lib/changelog.test.ts`(新)、`web/src/app/changelog/page.tsx`(新)、`web/src/app/me/page.tsx`(改)、`web/docker-compose.yml`(改)
  - 要点:changelog.ts 两个函数:`parseChangelog(markdown: string): ChangelogEntry[]`(`{version, date: string|null, whatsNew: string[]}`,只认 `## [x.y.z] - YYYY-MM-DD` / `## [Unreleased]` 段内 `### What's New` 的 `- ` bullets,其他小节忽略)和 `readChangelog()`(依次试 `process.env.CHANGELOG_PATH`、`cwd/CHANGELOG.md`、`cwd/../CHANGELOG.md`,都不存在返回 [];容器 cwd=/app 命中挂载文件,本地 dev cwd=web 命中上一级)。page.tsx 是 server component,**必须** `export const dynamic = "force-dynamic"`(routes.test.ts 守卫 + 每次请求重读文件);UI = 竖向时间线(圆点+竖线+版本卡片),Unreleased 用琥珀色「未发布」徽章,空态显示「暂无更新记录」。me/page.tsx 是 client component,入口卡片照抄「管理员登录」那块的 Link 结构,图标用 Sparkles,文案「更新日志 / 看看有什么新功能」。docker-compose.yml volumes 加一行 `- ../CHANGELOG.md:/app/CHANGELOG.md:ro`(服务器上 compose 在 web/ 下,`..` 即仓库根目录;部署时 git pull 后文件已存在,`up -d` 重建即生效,无需重新 build)。
  - verify: `pnpm --dir web test -- changelog routes`

- [x] T3 本周战绩显示对局日期和记录时间 [独立]
  - 改动:`web/src/lib/format.ts`(新)、`web/src/lib/format.test.ts`(新)、`web/src/components/week-matches.tsx`(改)
  - 要点:数据已全(WeekMatches 的 MatchWithNames 含 playedAt/createdAt),只改展示。纯函数 `formatMatchMeta(playedAt: string, createdAt: string, enteredByName: string): string`,输出如 `8月30日 对局 · 凯哥 录入于 21:43`;createdAt 可能是 sqlite `YYYY-MM-DD HH:MM:SS` 格式,解析前把空格换成 `T` 再 `new Date()`(Safari 兼容)。week-matches.tsx 替换第 155-157 行的「录入:X」一行,分组头不动。
  - verify: `pnpm --dir web test -- format`

- [x] T4 个人页参赛历史渐进加载(默认 10 条) [顺序]
  - 改动:`web/src/components/player-match-history.tsx`(新)、`web/src/app/players/[id]/page.tsx`(改)
  - 要点:把 page.tsx 第 79-117 行的内联渲染整体搬进新 client 组件,props = `{ matches: PlayerMatchRecord[], playerName: string }`(全量传入,不分页 API)。useState visibleCount 初始 10;底部哨兵 div 挂 IntersectionObserver 进视口 +10,同时保留「加载更多」按钮兜底(IntersectionObserver 不可用时);全部加载完显示「已显示全部 N 场」。空态文案保持「还没有参赛记录」。
  - verify: `pnpm --dir web exec tsc --noEmit`(注:eslint 配置本身是坏的,nextVitals is not iterable,既有问题不在本次修)+ [人工] dev server 打开一个 >10 场的球员主页:默认 10 条,滚到底自动追加,直至「已显示全部」

- [ ] T5 个人页趣味数据区块 [顺序]
  - 改动:`web/src/lib/stats.ts`(改,+playerFunStats)、`web/src/lib/stats.test.ts`(改,+用例)、`web/src/components/fun-stats.tsx`(新)、`web/src/app/players/[id]/page.tsx`(改,在概览卡和交锋记录之间插入)
  - 要点:`playerFunStats(playerId: number, data: StatsData): PlayerFunStats`,返回 `{ currentStreak: number, currentStreakType: "win"|"loss"|"none", longestWinStreak: number, bestPartner: {id,name,wins,total,winRate}|null, nemesis: {id,name,wins,losses,total,winRate}|null, peakElo: number, peakEloDate: string|null, avgPointDiff: number }`。data.matches 已按 played_at/created_at/id 升序,可直接算连胜;bestPartner/nemesis 都要求交手/搭档 ≥3 场,不足为 null;peakElo 取 data.eloHistory 中该球员最大值及首次达成日期;avgPointDiff = 平均(我方得分−对方得分)保留 1 位小数。fun-stats.tsx 用 2 列卡片网格:当前连胜(🔥,连败用玫瑰色)、最长连胜、黄金搭档(名字+胜率+场次)、头号克星(名字+胜率+场次)、ELO 峰值(+日期)、场均净胜分(+/-);null 项显示「数据不足」。测试直接构造 StatsData 字面量,纯函数不碰 db;stats.test.ts 已有构造模式可参考。
  - verify: `pnpm --dir web test -- stats`

- [x] T6 配对持久化 + 点击场次跳转记分预填 [独立]
  - 改动:`web/src/lib/schedule-storage.ts`(新)、`web/src/lib/schedule-storage.test.ts`(新)、`web/src/app/schedule/schedule-form.tsx`(改)、`web/src/app/record/page.tsx`(改)、`web/src/components/record-form.tsx`(改)
  - 要点:schedule-storage.ts 定义并导出 `ScheduledMatch`/`ScheduleResult` 类型(schedule-form 现在本地的类型搬过来复用)+ `StoredSchedule { playerIds, matches, seed, lambda, result, savedAt }`;`parseStoredSchedule(raw: string): StoredSchedule | null`(校验形状,坏数据返回 null,纯函数可测);`loadSchedule/saveSchedule/clearSchedule(userId: number | null)` 包 localStorage,key = `badminton:schedule:${userId ?? "anon"}`。schedule-form 挂载时按 getMyPlayerId() 恢复 selected/matches/seed/lambda/result;handleGenerate 成功后 saveSchedule;clear() 加 clearSchedule。结果卡片改成 Link → `/record?pa1=${a1}&pa2=${a2}&pb1=${b1}&pb2=${b2}`(id 本就是字符串),加 ChevronRight 和「点击去记分」提示。record/page.tsx(server,已 force-dynamic)读 `searchParams`(Promise,需 await)里 4 个 id,转 number 并对 listPlayers() 校验存在后作为 `initialSlots` 传给 RecordForm;RecordForm 加可选 prop `initialSlots?: [Slot,Slot,Slot,Slot]` 初始化 slots state(不用 useSearchParams,免 Suspense 坑)。
  - verify: `pnpm --dir web test -- schedule-storage` + [人工] 配对页:生成 → 刷新仍在 → 点场次跳 /record 四人已预填 → 清空后刷新不再恢复
