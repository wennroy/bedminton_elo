# 周三局报名平台

> 状态：进行中

## 目标

每周三 18:00–20:00 的固定局有线上报名处：球友在 `/signup` 页选「我是谁」一键报名，可带一位小伙伴（人数 1/2）；首页顶部卡片显示本期报名情况并引流；每周三 20:00 后页面自动滚动为下周三的局，零管理成本。

## 非目标

- 推送通知（微信群 / 企业微信 / PushPlus 等任何渠道）——后续独立 plan
- 与配对页联动（如配对默认勾选已报名球员）
- 历史场次回顾、出勤统计
- 人数上限 / 候补队列、场地等固定文案、管理员手动开局或改时间
- 为小伙伴建球员档案（+1 永远只是计数）

## 关键决定

- **页面位置**：独立 `/signup` 页 + 首页顶部「周三局」卡片入口；底部导航不动（已满）。
- **报名身份**：复用现有免登录体系（localStorage 选「我是谁」），报名时选人数 1（自己来）或 2（带一位小伙伴）；+1 小伙伴不填名字、不建档案，榜单数据不受污染。
- **局的生成**：lazy 滚动，无 sessions 表、无定时任务——session 就是「周三的日期」本身，`signups.session_date` 直接存日期；周三 20:00（Asia/Shanghai）前展示当周场，之后任何访问自动切到下周三。随时可报名/取消，不设人数上限。
- **名单公开**：谁报了名所有人可见（接龙性质）。
- **管理员代管理**：admin key 用户在名单每行看到「移除」和改人数按钮（复用现有 admin.ts 体系）；API 沿用全站免登录信任模型，不强制服务端鉴权（与记分/撤回一致）。
- **页面文案**：只显示「每周三 18:00–20:00」+ 本期日期 + 名单，无地点说明。
- **时区**：会话日期计算必须显式按 Asia/Shanghai（容器内 TZ 不可靠，用 UTC+8 偏移法，勿依赖服务器本地时区）。
- **推送存档结论**（本期不做）：普通微信群无官方机器人 API，第三方 hook 有封号风险；将来做推送首选企业微信群机器人 webhook，备选 PushPlus/Server酱 推个人微信。

## 假设

（空）

## 全局验收

`cd web && pnpm test && pnpm build`

## Tasks

- [x] T1 signups 数据层：schema + lib + 测试 [顺序]
  - 改动：`web/src/lib/schema.sql`（改）、`web/src/lib/signup.ts`（新）、`web/src/lib/signup.test.ts`（新）、`web/src/lib/repo.test.ts`（改：legacy 迁移测试在 drop players 前先 `DROP TABLE IF EXISTS signups`，否则 FK 引用报错 foreign key mismatch）
  - 要点：schema.sql 追加 `CREATE TABLE IF NOT EXISTS signups (id INTEGER PRIMARY KEY AUTOINCREMENT, session_date TEXT NOT NULL, player_id INTEGER NOT NULL REFERENCES players(id), party_size INTEGER NOT NULL DEFAULT 1 CHECK (party_size IN (1,2)), created_at TEXT NOT NULL DEFAULT (datetime('now')), UNIQUE(session_date, player_id))`（db.ts 每次开库自动 exec schema.sql，无需迁移脚本）。lib 函数风格对齐 repo.ts（可选 `db?` 参数便于内存库测试）：
    - `getActiveSessionDate(now: Date): string` —— 用 `new Date(now.getTime() + 8*3600_000)` 得上海时间，再读 `getUTC*()`：若是周三且小时 <20 → 返回当天，否则返回下一个周三（`(3-day+7)%7 || 7` 天后），格式 YYYY-MM-DD。
    - `listSignups(sessionDate, db?)` —— join players 拿名字，按 created_at 升序。
    - `upsertSignup(sessionDate, playerId, partySize, db?)` —— `INSERT ... ON CONFLICT(session_date, player_id) DO UPDATE SET party_size=excluded.party_size`。
    - `removeSignup(sessionDate, playerId, db?)`。
    - `signupSummary(sessionDate, db?)` —— `{ count, totalPeople }`（totalPeople = sum(party_size)）。
    - 测试覆盖：周三 19:59 → 当天、周三 20:00 → +7 天、周四 → 下周三、upsert 改人数、remove、唯一约束。
  - verify: `cd web && pnpm test -- signup`

- [x] T2 /api/signups 路由 [顺序]
  - 改动：`web/src/app/api/signups/route.ts`（新）
  - 要点：结构对齐 `api/matches/route.ts`（NextResponse、force-dynamic、try/catch 500）。GET → `{ sessionDate, signups: [{playerId, name, partySize, createdAt}], totalPeople }`，sessionDate 恒为 `getActiveSessionDate(new Date())`（不支持查历史）。POST body `{playerId, partySize}`：校验 playerId 存在于 players、partySize ∈ {1,2}，upsert 到当期 session，之后 `revalidatePath("/signup")` 和 `"/"`。DELETE body `{playerId}`：从当期 session 移除，同样 revalidate。不做 admin key 校验（见关键决定）。
  - verify: `cd web && pnpm exec tsc --noEmit && pnpm test` + `pnpm dev` 后 `curl -s localhost:3000/api/signups` 及 POST/DELETE 各一次看返回 `[人工]`

- [x] T3 /signup 报名页 [独立]
  - 改动：`web/src/app/signup/page.tsx`（新）、`web/src/app/signup/signup-form.tsx`（新）
  - 要点：page.tsx 是 server component，**必须 `export const dynamic = "force-dynamic"`**（routes.test.ts 强制，否则 build 时静态化踩旧坑）；渲染标题「周三局报名」、副标题「每周三 18:00–20:00 · 本期 M月D日（周三）」（由 getActiveSessionDate 格式化）、名单（`名字 ×2` 样式，顶部总计「已报 N 人 · 含小伙伴共 M 人」）、以及 client 组件 SignupForm。SignupForm：用 `getMyPlayerId()` 读身份，未选 → 提示去「我的」页选身份（可复用 identity-picker 模式）；未报名 → 人数切换 1/2 + 「报名」按钮；已报名 → 显示「已报名 ×N」，可切人数（再 POST 即 upsert）+「取消报名」；`getAdminKey()` 非空时每行显示「移除」按钮。所有操作后 `router.refresh()`。视觉对齐现有页面（px-4 pb-28 pt-4、card 风格）。
  - verify: `cd web && pnpm exec tsc --noEmit` + `pnpm dev` 目视走完：报名 → 改人数 → 取消 → admin 移除他人 `[人工]`

- [x] T4 首页「周三局」卡片 [独立]
  - 改动：`web/src/app/page.tsx`（改）、`web/src/components/signup-card.tsx`（新）
  - 要点：SignupCard 为 server 组件（在 page.tsx 直接渲染，传 summary 或内部自取数据均可，注意保持 force-dynamic 页面内直出）；放在 HomeTrend 上方。内容：「🏸 周三局 · M月D日（周三）18:00–20:00」+「已报名 N 人（含小伙伴共 M 人）」+ 右侧「去报名 →」Link 到 /signup；0 人时也正常显示（拉人）。与 T3 无共同文件。
  - verify: `cd web && pnpm exec tsc --noEmit` + `pnpm dev` 首页目视卡片与跳转 `[人工]`

- [x] T5 CHANGELOG 更新 [顺序]
  - 改动：`CHANGELOG.md`（改）
  - 要点：Unreleased 的 What's New 加条目（占位注释前）：「周三局报名页上线：选好『我是谁』一键报名，可标记带一位小伙伴；首页新增本期报名卡片」。措辞面向球友。T1–T4 完成后做。
  - verify: 目视 CHANGELOG 段落 `[人工]`
