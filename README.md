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

```bash
git pull
cd web
cp .env.example .env
# 编辑 .env，设置 ADMIN_PASSWORD
docker compose up -d --build
```

数据卷挂载在 `web/data`，容器重启后数据持久化。

## Apache 反向代理示例

假设应用监听 `127.0.0.1:3000`，Apache 配置片段：

```apache
<VirtualHost *:80>
    ServerName badminton.example.com

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:3000/
    ProxyPassReverse / http://127.0.0.1:3000/

    # Next.js 不使用 WebSocket，无需额外 ws 代理
</VirtualHost>
```

启用所需模块：

```bash
sudo a2enmod proxy proxy_http
sudo systemctl restart apache2
```

## legacy/ 目录说明

`legacy/` 为旧版 Python 实现，包含原始 ELO / TrueSkill 计算逻辑与数据文件。新版 `web/scripts/migrate-legacy.ts` 会在容器首次启动时自动检测旧表结构并把双打比赛迁移到新 schema，迁移状态写入 SQLite `meta` 表，不会重复执行。
