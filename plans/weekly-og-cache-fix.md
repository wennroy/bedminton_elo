# 周报分享图缓存失效修复

> 状态：已完成

## 目标

修复 bug：生成过一次的周报分享图，在录入了新比赛后再次点击「生成分享图」
仍是旧图。修完后：每次点击都会校验后台数据是否有更新——无更新时走缓存
秒回（HTTP 304，不重渲染不重新下载），有更新时重新渲染出最新图。

## 根因（已查实，执行时不必再查）

`next/og` 的 `ImageResponse` 在生产环境硬编码响应头
`Cache-Control: public, immutable, no-transform, max-age=31536000`
（node_modules/next/dist/compiled/@vercel/og/index.node.js 约 20124 行，
`...options.headers` 在其后展开，**可被 options.headers 覆盖**）。
而分享图 URL 恒为 `/api/og/weekly?week=<周一日期>`，浏览器对同一 URL
直接命中一年期 immutable 缓存，永不再请求服务器。线上
`curl -sI "https://bedminton.wennroy.com/api/og/weekly?week=2026-09-01"`
已证实该响应头。

注意：`immutable` 缓存条目按完整 URL（含 query）作 key 且不再回源，
所以**只改响应头救不了已中招的浏览器**——必须同时换 URL key
（客户端 fetch 加一个一次性破缓存参数）。

## 非目标

- 不做服务器端 PNG 文件缓存（304 短路已够，渲染是唯一重活）
- 不动 /weekly 页面本身（已是 `force-dynamic`，页面数据总是新的）
- 不动其他接口的缓存策略（全站只有这一个 OG 路由）
- 不动 Apache 反代配置（它只透传响应头，无 mod_cache）

## 关键决定

- 方案 = HTTP 协商缓存：服务端算出本周报内容指纹作 ETag，响应头改为
  `Cache-Control: no-cache`（存储但每次用前回源校验）+ `ETag`；
  请求带匹配的 `If-None-Match` 时直接 304，**跳过 Satori 渲染**
- 指纹 = `sha1(JSON.stringify(WeeklyStats)).slice(0,16)`——WeeklyStats
  是图片内容的全部决定因素（比赛/比分/球员名/ELO 全在里面），
  由 DB 状态确定性推出，天然覆盖一切数据更新
- 客户端一次性破缓存：fetch URL 加常量参数 `&v=2`，让浏览器手里的
  旧 immutable 条目（key 为无参 URL）永不再命中；此后新鲜度由 ETag 保证
- 版本号 v1.4.1（bugfix patch），用 scripts/release.sh 发版并部署上线

## 假设

（无）

## 全局验收

`pnpm -C web lint && pnpm -C web test && pnpm -C web build`

## Tasks

- [x] T1 周报内容指纹 helper + 单测 [独立]
  - 改动：`web/src/lib/weekly.ts`（改）、`web/src/lib/weekly.test.ts`（改）
  - 要点：新增 `export function weeklyDataVersion(stats: WeeklyStats): string`，
    实现为 `createHash("sha1").update(JSON.stringify(stats)).digest("hex").slice(0, 16)`（node:crypto）。
    lib/weekly.ts 只被服务端（page/OG route）以值导入，weekly-view 对它只有
    type-only import，加 node:crypto 不会泄进客户端包。
    测试用现有 computeWeeklyStats 纯函数路径：同输入两次指纹相同；
    加一场比赛 → 变；改球员名 → 变；改比分 → 变。
  - verify: `pnpm -C web test -- lib/weekly`

- [x] T2 OG 路由改协商缓存 + 路由测试 [顺序]（依赖 T1 的 helper）
  - 改动：`web/src/app/api/og/weekly/route.tsx`（改）、
    `web/src/app/api/og/weekly/route.test.ts`（新）
  - 要点：GET 顺序改为：校验 week 参数（保持现有 400）→
    `buildWeeklyStats(weekStart)` → `weeklyDataVersion(stats)` →
    `const etag = \`"${version}"\``；若
    `request.headers.get("if-none-match") === etag` 则
    `return new Response(null, { status: 304, headers: { ETag: etag, "Cache-Control": "no-cache" } })`
    （在 QRCode/ImageResponse 之前短路，省掉渲染）。
    否则照常渲染，ImageResponse options 传
    `headers: { "Cache-Control": "no-cache", ETag: etag }`（覆盖 immutable 默认值）。
    stats 已在 GET 里算过，把它传给 WeeklyCard 而不是让 WeeklyCard 内部再
    build 一次（小重构：WeeklyCard 加 stats prop，删掉内部 buildWeeklyStats 调用）。
    测试照 `api/schedule/route.test.ts` 的临时库模式（tmpdir + DATABASE_URL +
    closeDb + describe.sequential）：造 4 人 1 场 → If-None-Match 匹配当前指纹
    断言 304 且 body 为空；带过期 tag 断言 200 + content-type image/png +
    cache-control 为 no-cache + etag 存在（200 路径会真渲染，next/og 的 node
    构建在 vitest 下可用，断言 status/headers 即可，不必消费 blob）。
  - verify: `pnpm -C web test -- api/og/weekly && pnpm -C web lint src/app/api/og/weekly`
  - 执行补记：直接 import .tsx 路由进 vitest 需要 `vitest.config.ts` 加
    `esbuild: { jsx: "automatic" }`（tsconfig 的 jsx:preserve 只对 Next 生效，
    esbuild 默认 classic 会报 React is not defined）——已随 T2 一并修复。

- [x] T3 客户端一次性破缓存参数 [独立]
  - 改动：`web/src/app/weekly/weekly-view.tsx`（改）
  - 要点：handleExport 里 fetch URL 改为
    `/api/og/weekly?week=${stats.weekStart}&v=2`，上方加注释：旧版本响应带
    immutable 一年缓存且按 URL 作 key，加常量参数换 key 规避存量缓存条目，
    此后新鲜度由服务端 ETag 协商保证。其余不动。
  - verify: `pnpm -C web lint src/app/weekly/weekly-view.tsx && pnpm -C web test`

- [x] T4 全局验收 + 生产模式冒烟 [顺序]（依赖 T1-T3）
  - 改动：无（仅验证）
  - 要点：跑全局验收；然后临时库冒烟（参考 v1.4.0 做法：tmpdir 建库 +
    schema.sql + 4 人 1 场，`DATABASE_URL=/tmp/xxx.db npx next start -p 3210`
    用 .next 生产构建跑）：
    1. `curl -sI "localhost:3210/api/og/weekly?week=<该场所在周一>&v=2"` →
       cache-control 为 no-cache、有 etag、**不再是 immutable**
    2. 带上一步 etag 发 `curl -sI -H 'If-None-Match: "<etag>"' ...` → 304
    3. 往临时库再录一场后重启服务（或换 DB 内容），同 URL → 200 且 etag 变了
    4. 旧的无 v 参数 URL 仍能 200（回归）
    冒烟完删除临时库。
  - verify: `pnpm -C web lint && pnpm -C web test && pnpm -C web build` ＋[人工] 上述 4 条 curl 全过

- [x] T5 更新 CHANGELOG.md [顺序]
  - 改动：`CHANGELOG.md`（仓库根目录，改）
  - 要点：Unreleased 的 What's New 下加面向球友的一条，如
    「修复：周报分享图在录入新比赛后重新生成仍是旧图，现在会按最新数据出图」。
    不 bump 版本号（release 脚本负责）。
  - verify: `pnpm -C web test`（changelog.test.ts 校验格式）

- [x] T6 发布 v1.4.1 并部署上线 [顺序]（依赖 T1-T5）＋[人工]
  - 改动：`CHANGELOG.md`、`web/package.json`（均由 scripts/release.sh 改）
  - 要点：主会话执行，不派 subagent。
    1. `scripts/release.sh 1.4.1` → 提交（风格参照 git log，如
       `Release v1.4.1`）→ `git push origin master`
    2. 服务器：`ssh wennroy` 后 `cd ~/proj/bedminton_elo && git pull --ff-only origin master`
       （GitHub 偶发 TLS 挂死：用
       `git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=15 pull --ff-only origin master`
       快速失败重试），然后 `cd web && docker compose up -d --build`
    3. **CHANGELOG.md 是单文件 bind mount，git pull 换新 inode 后容器内仍是旧内容，
       必须再 `docker compose restart`**（无需 --build 之外的额外操作，restart 可与
       up -d --build 合并成一次 up 后的 restart；若容器本来就重建了则已生效，
       以 /changelog 实际显示为准）
    4. 线上验证：
       `curl -sI "https://bedminton.wennroy.com/api/og/weekly?week=<最近周一>&v=2"`
       → cache-control: no-cache 且有 etag；带该 etag 的 If-None-Match 请求 → 304；
       https://bedminton.wennroy.com/changelog 显示 1.4.1
  - verify: [人工] 上述线上 curl 三条全过 + /changelog 显示 1.4.1
