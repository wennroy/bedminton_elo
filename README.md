# 卷技术小分队 🏸 羽毛球 ELO 排行榜

基于 Next.js 15 + TypeScript + Tailwind CSS + shadcn/ui 的羽毛球双打 ELO 记分与排行榜应用。支持实时排行榜、比赛录入、个人主页、趋势图、胜率预测、配对调度与每周战报。

数据层使用 `better-sqlite3`，部署时通过 Docker 在服务器本地运行， Apache 反向代理对外提供服务。

## 目录结构

```
web/          Next.js 应用
legacy/       旧版 Python 实现（已归档，仅作历史参考）
```

## 本地开发

```bash
cd web
corepack enable
pnpm install
pnpm dev
```

打开 <http://localhost:3000>。

### 测试

```bash
pnpm test        # vitest
```

### 填充 mock 数据

```bash
npx tsx scripts/seed-mock.ts
# 强制覆盖已有数据
npx tsx scripts/seed-mock.ts --force
```

## 环境变量

| 变量 | 说明 | 示例 |
|---|---|---|
| `DATABASE_URL` | SQLite 数据库路径，容器内固定为 `/data/badminton.db` | `/data/badminton.db` |
| `ADMIN_PASSWORD` | 管理后台口令，必填 | `your-secure-password` |

本地开发时可在 `web/.env` 中设置；生产环境通过 `web/.env` 或服务器环境变量注入。

## 部署流程

服务器需已安装 Docker 与 Docker Compose。

### 0. 配置 Docker Hub 加速器(国内服务器必做)

编辑 `/etc/docker/daemon.json`(没有就新建),加入腾讯云镜像加速器:

```json
{
  "registry-mirrors": ["https://mirror.ccs.tencentyun.com"]
}
```

```bash
sudo systemctl restart docker
```

### 1. 拉代码并配置

```bash
git pull
cd web
cp .env.example .env
# 编辑 .env,设置 ADMIN_PASSWORD
```

### 2. 接入旧数据(重要)

**先备份旧库**,再把它放到挂载目录:

```bash
cp /path/to/旧库/badminton.db ~/badminton.db.bak   # 备份
mkdir -p data
cp ~/badminton.db.bak data/badminton.db           # 旧库就位
```

容器启动时会自动运行 `migrate-legacy.ts`:检测到旧表结构(球员是姓名字符串)就幂等迁移到新 schema,旧表保留为 `matches_legacy` 备份。**迁移只迁双打记录**,单打行会跳过并打印数量。想用别的路径就在 `.env` 里改 `DATABASE_URL`。

### 3. 构建并启动

```bash
docker compose up -d --build
docker compose logs -f web   # 确认 migration 日志与启动成功
```

数据卷挂载在 `web/data`,容器重启后数据持久化。

### 4. 从旧 Streamlit 切换

新版默认监听 `127.0.0.1:8503`(与现有 Apache 配置一致,Apache 无需改动)。切换时先停掉旧 Streamlit 释放 8503 端口,再 `docker compose up -d --build`。要换端口在 `.env` 里设 `PORT`。

## Apache 反向代理示例

假设应用监听 `127.0.0.1:8503`，Apache 配置片段：

```apache
<VirtualHost *:80>
    ServerName badminton.example.com

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8503/
    ProxyPassReverse / http://127.0.0.1:8503/

    # Next.js 不使用 WebSocket，无需额外 ws 代理
</VirtualHost>
```

启用所需模块：

```bash
sudo a2enmod proxy proxy_http
sudo systemctl restart apache2
```

## 实力分与评分模式

- 页面顶部可切换「新版 / Legacy」：新版为 Glicko-2 双打实力分，Legacy 为旧版 ELO，两套分数并存、互不影响，Legacy 视图保留全部旧分数与曲线。
- 新版周内每场比赛结束立即给出分数变化反馈（标注「预估 Estimated」）；每周一统一结算该周正式分（Final）。
- 周一结算会把整周的预估统一校准，**分数可能上调也可能下调**——这是评分机制的正常行为，不代表录入有误；个人页可查看每周每场明细与校准量。
- 每三个月一个赛季：新赛季开始时分数软重置（900～1100 区间保留，两端超出部分保留 75%），不确定性回升，久未参赛者回到更公平的起跑线。
- 默认展示的评分模型由服务端配置决定（`web/scripts/rating-config.ts`，持久化在数据库 `meta` 表，重启不丢）；未配置时站点按 Legacy 运行。
- `/season` 赛季报按自然季度汇总榜单：期初→期末分数涨跌、出勤、战绩王、最佳组合、赛季趣闻，并简介赛季制度（进行中赛季统计截至当天）。
- `/methodology` 计分方式说明页对照讲解 Glicko-2 与 Legacy ELO 的计算公式和数值示例，并说明 TrueSkill 在前端展示中的角色，方便球友看懂分数的来历。

## legacy/ 目录说明

`legacy/` 为旧版 Python 实现，包含原始 ELO / TrueSkill 计算逻辑与数据文件。新版 `web/scripts/migrate-legacy.ts` 会在容器首次启动时自动检测旧表结构并把双打比赛迁移到新 schema，迁移状态写入 SQLite `meta` 表，不会重复执行。
