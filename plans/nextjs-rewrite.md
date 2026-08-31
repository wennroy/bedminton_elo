# 卷技术小分队🏸 Next.js 重写

> 状态：进行中

## 目标

把现有 Streamlit 羽毛球 ELO 应用重写为手机优先的 Next.js 全栈网站:球友在球场局间 10 秒录完一场双打比分,录完立刻看到 ELO 涨跌;排行榜/个人主页/周报让大家愿意持续使用。本地 dev 用 mock 数据先行验证效果,测试通过后经 GitHub 同步到服务器 Docker 部署,接管现有域名与数据。

## 非目标

- 单打功能(只要双打,一场=一局)
- 分差计分(只看胜负,标准 ELO)
- PWA/加到主屏、真实照片头像(自动彩色字头像)
- 用户注册/密码体系(免注册,点选身份 + localStorage)
- 待办比赛流程(配对只生成表,比分走统一快速录入)
- 整表重写式数据管理(改为按场编辑/删除)
- 删除 legacy Python 文件(已统一收进 `legacy/` 目录作移植参照,上线后再清理)

## 关键决定

澄清阶段(拷问轮)定案,一行一条;review 按此验收:

- 核心场景 → 局间实时记分为主,记分牌式 UI:大按钮、头像选人、零打字
- 记分权限 → 任何人可记任何场;每条记录存 `entered_by` 追溯
- 身份 → 免注册,首访点选「我是谁」,localStorage 记住;管理员口令才能改/删历史
- 访问 → 公网域名 + HTTPS(服务器已有 Apache2 反代),手机浏览器优先,桌面端自适应即可
- 技术栈 → Next.js 15(App Router)+ TypeScript 全栈;SQLite 沿用,`DATABASE_URL` 环境变量指向 db 文件
- 核心钩子 → 排行榜首页 + 录入后即时 ELO 涨跌反馈 + 当周战报(自然周)
- 视觉 → 浅色清爽工具风,中文界面,沿用「卷技术小分队🏸」名称
- 数据 → 在服务器上,`DATABASE_URL` 指向;新版 schema 用球员 id 外键(见 T3),容器启动时跑幂等迁移脚本把旧的 name 字符串转成 id
- 配对功能 → 保留但简化:一键生成配对表(模拟退火),无待办流程;温度/种子藏「高级选项」
- 评分系统 → ELO + TrueSkill 双系统都保留,排行榜可切换,默认 ELO
- 双打 ELO bug → 修复 `legacy/main.py:209-231` 的运算符优先级 bug,改为 `Δ = K * (S - E)`;评分本就全量重放 matches,历史零成本自动重算
- 比赛形式 → 只要双打;一场=一局比分(如 21:18);比分不能相同
- 周报 → 独立页面(出勤榜/战绩王/ELO 涨跌榜/最佳组合)+ 一键生成分享图(next/og)
- 球员管理 → 任何人可现场加人;自动彩色字头像(姓名首字 + 稳定配色);管理员可改名/合并/删除
- 数据功能 → 个人主页(参赛历史 + head-to-head)、ELO 趋势图、2v2 胜率预测器,全部保留
- 部署 → 现有服务器 + Docker 单容器(docker compose),Apache 反代接管现有域名;流程:本地 dev 验证 → push GitHub → 服务器 pull + 重建
- 纠错 → 任何人可撤回录入后 10 分钟内的比赛(`created_at` 判定);超时仅管理员可改
- 管理员 → `ADMIN_PASSWORD` 环境变量;设置页输入一次后设备记住;可编辑/删除任意比赛、球员改名/合并/删除
- K 值/初始分 → 沿用 K_DOUBLES=16、INITIAL_RATING=1000;TrueSkill 照 `legacy/trueskill_utils.py` 原样移植(mu=25, sigma=25/3, beta=sigma/2),保证分数连续性
- 配对算法 → 照 `legacy/random_utils.py` 移植:模拟退火 iters=5000、T0=1.0、decay=0.995、floor=1e-4,loss = α·出场方差 + λ·组合重复熵 + (1−λ)·平均胜负差·2,胜率用 TrueSkill 预测
- 包管理 → pnpm;图表 → Recharts;分享图 → next/og(ImageResponse);DB 驱动 → better-sqlite3(不用 ORM)
- 目录 → Next.js 应用在 `web/` 子目录,legacy Python 统一收在 `legacy/` 目录作移植参照
- mock 数据 → 10 个中文名球员 + 过去 8 周约 150 场比赛,固定种子生成,保证排行榜/趋势图/周报都有内容可看

## 假设

(空)

## 全局验收

`cd web && pnpm typecheck && pnpm vitest run && pnpm build`

## Tasks

- [x] T1 Next.js 脚手架 + 应用骨架 [顺序]
  - 改动:`web/`(全新 Next.js 15 + TS + Tailwind + shadcn/ui 脚手架)、`web/src/components/identity-picker.tsx`(新)、`web/src/components/player-avatar.tsx`(新)、`web/src/components/bottom-nav.tsx`(新)、`web/src/lib/identity.ts`(新)、`web/src/app/layout.tsx`(新)、`web/src/app/page.tsx`(占位)
  - 要点:浅色主题;手机优先布局,底部导航(首页/记分/配对/周报/我的)。`identity.ts` 提供 `getMyPlayerId()/setMyPlayerId()` 读写 localStorage;`IdentityPicker` 首访弹窗列出球员网格(自动头像)点选;`PlayerAvatar` 用姓名首字 + 名字 hash 稳定配色。页面标题「卷技术小分队🏸」。
  - verify: `cd web && pnpm build && pnpm dev` [人工:手机尺寸视口下看骨架、导航、身份选择弹窗]

- [x] T2 算法移植:ELO(修复版)+ TrueSkill + 模拟退火配对 [顺序]
  - 改动:`web/src/lib/elo.ts`(新)、`web/src/lib/trueskill.ts`(新)、`web/src/lib/scheduler.ts`(新)、`web/scripts/gen_golden.py`(新)、`web/test/golden/*.json`(新)、`web/src/lib/*.test.ts`(新)
  - 要点:以 `legacy/main.py`(calculate_elo/calculate_trueskill/predict)、`legacy/trueskill_utils.py`、`legacy/random_utils.py` 为移植 spec。ELO 双打必须修 bug:`Δ_p = K*(S − E_p)`,`E_p = 1/(1+10^((R_对手队均 − R_p)/400))`。`gen_golden.py` 用现有 Python 代码对固定输入生成期望值 JSON(双打 ELO 用**修复后**的期望值,在脚本里写明修复逻辑);vitest 逐条对齐。scheduler 移植模拟退火全部常数(iters=5000, T0=1.0, decay=0.995, floor=1e-4)与三种邻域操作;随机数用可注入 seed 的 PRNG(如 mulberry32),保证测试可复现。
  - verify: `cd web && python3 scripts/gen_golden.py && pnpm vitest run src/lib`

- [ ] T3 数据层 + schema 迁移 + mock seed [顺序]
  - 改动:`web/src/lib/db.ts`(新)、`web/src/lib/schema.sql`(新)、`web/scripts/migrate-legacy.ts`(新)、`web/scripts/seed-mock.ts`(新)、`web/src/lib/repo.ts`(新)、`web/src/lib/repo.test.ts`(新)
  - 要点:新 schema:`players(id, name UNIQUE, created_at)`;`matches(id, pa1, pa2, pb1, pb2, score_a, score_b, played_at, entered_by, created_at)` 全部用球员 id 外键。`db.ts` 用 better-sqlite3 打开 `process.env.DATABASE_URL`(默认 `./badminton.db`),启动时执行 `schema.sql`(IF NOT EXISTS)。`migrate-legacy.ts` 幂等:检测旧表结构(matches 里是 name 字符串)→ 建 players → name 映射 id 重写 matches → 旧表改名 `matches_legacy` 备份;用「legacy 是否已迁移」标记位防重复跑。`repo.ts` 是唯一数据访问层:`listPlayers/addPlayer/renamePlayer/mergePlayers/addMatch/listMatchesByDate/deleteMatch/getMatch/recomputeAllRatings`(全量重放,调 T2 的 elo/trueskill,结果存内存或直接算,不落缓存表)。`seed-mock.ts` 固定种子生成 10 球员 + 8 周约 150 场。
  - verify: `cd web && pnpm vitest run src/lib/repo.test.ts && pnpm tsx scripts/seed-mock.ts && sqlite3 "$(DATABASE_URL:-./badminton.db)" 'select count(*) from matches;'`(应 ≈150)

- [ ] T4 核心闭环:首页排行榜 + 快速记分 + 当天比赛 + 10 分钟撤回 [顺序]
  - 改动:`web/src/app/page.tsx`(改)、`web/src/app/record/page.tsx`(新)、`web/src/app/api/matches/route.ts`(新,POST/DELETE)、`web/src/app/api/matches/[id]/route.ts`(新)、`web/src/components/leaderboard.tsx`(新)、`web/src/components/record-form.tsx`(新)、`web/src/components/elo-delta-card.tsx`(新)、`web/src/app/api/matches/route.test.ts`(新)
  - 要点:首页 = 排行榜(ELO/TrueSkill 切换 tabs,默认 ELO)+ 底部「当天比赛」列表(10 分钟内场次显示撤回按钮)。记分流程:选 4 人(头像网格,分 A/B 两队)→ 大号数字步进器输比分(默认 21:x,校验不相等)→ 提交 → `EloDeltaCard` 展示四人 ELO 涨跌(+动画)。POST 先写库再 `recomputeAllRatings`,响应带四人前后分数;DELETE 仅当 `Date.now() - created_at < 10min` 或带管理员口令。API 测试用临时 db 文件跑通 POST/DELETE/撤回窗口。
  - verify: `cd web && pnpm vitest run src/app/api/matches && pnpm dev` [人工:手机视口录一场 21:18,确认排行榜即时变化、EloDeltaCard 展示、10 分钟内可撤回]

- [ ] T5 个人主页 + ELO 趋势图 + 2v2 胜率预测器 [独立]
  - 改动:`web/src/app/players/[id]/page.tsx`(新)、`web/src/components/h2h-table.tsx`(新)、`web/src/app/trends/page.tsx`(新)、`web/src/components/elo-chart.tsx`(新)、`web/src/app/predict/page.tsx`(新)、`web/src/lib/stats.ts`(新)、`web/src/lib/stats.test.ts`(新)
  - 要点:`stats.ts` 提供 `headToHead(playerId)`(对每人胜-负)、`eloHistory()`(每日快照,供 Recharts 折线)、`playerSummary(playerId)`。个人主页:头像 + 当前双评分 + 参赛历史 + h2h 表。预测器页选 4 人,调 T2 的 `predict_elo`/`predict_trueskill` 移植版,展示两队胜率。
  - verify: `cd web && pnpm vitest run src/lib/stats.test.ts && pnpm dev` [人工:看个人主页 h2h、趋势图、预测器结果合理]

- [ ] T6 配对生成页 [独立]
  - 改动:`web/src/app/schedule/page.tsx`(新)、`web/src/app/api/schedule/route.ts`(新)
  - 要点:选在场球员(≥4,头像多选)+ 总场次 → 调 T2 `scheduler.ts` 生成配对表展示(每场可显示 TrueSkill 预测胜率)。「高级选项」折叠面板:随机种子、温度 λ。不写入任何待办表,生成即展示。
  - verify: `cd web && pnpm vitest run src/lib/scheduler.test.ts && pnpm dev` [人工:选 6 人 4 场,生成结果出场均衡]

- [ ] T7 周报页 + 一键分享图 [独立]
  - 改动:`web/src/app/weekly/page.tsx`(新)、`web/src/app/api/og/weekly/route.tsx`(新)、`web/src/lib/weekly.ts`(新)、`web/src/lib/weekly.test.ts`(新)
  - 要点:`weekly.ts` 按自然周(周一 00:00 起)聚合:出勤榜、战绩王(胜场最多)、ELO 涨跌榜、最佳组合(胜率最高且 ≥3 场)。周报页可切换历史周;「生成分享图」打开 `/api/og/weekly?week=YYYY-MM-DD`,用 next/og ImageResponse 输出 1080×1350 卡片(浅色风,标题「卷技术小分队🏸 第 N 周战报」)。
  - verify: `cd web && pnpm vitest run src/lib/weekly.test.ts && pnpm dev` [人工:mock 数据下看本周/上周周报,打开分享图链接确认渲染]

- [ ] T8 管理页 + Docker + 部署文档 [顺序]
  - 改动:`web/src/app/admin/page.tsx`(新)、`web/src/lib/admin.ts`(新)、`web/src/app/api/admin/**`(新)、`web/Dockerfile`(新)、`web/docker-compose.yml`(新)、`README.md`(改)
  - 要点:设置页输入 `ADMIN_PASSWORD` 后 localStorage 记住,请求带 `x-admin-key` 头;管理页可编辑/删除任意比赛、球员改名/合并(merge 把 B 的所有记录改指 A 后删 B)。Dockerfile:node:22-slim 多阶段(better-sqlite3 用预编译二进制,避免 alpine musl 编译),`DATABASE_URL=/data/badminton.db` 挂载卷,启动时先跑 `migrate-legacy.ts`。compose 暴露 `127.0.0.1:3000` 供 Apache 反代。README 写清:本地 `pnpm dev` → 测试 → push GitHub → 服务器 `git pull && docker compose up -d --build` → Apache ProxyPass 配置示例。
  - verify: `cd web && pnpm build && docker build -t bedminton-web . && docker run --rm -e ADMIN_PASSWORD=test -p 3000:3000 bedminton-web` [人工:容器起来后录一场 + 管理员删除一场]
