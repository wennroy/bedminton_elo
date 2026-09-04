# 部署

生产站点 <https://bedminton.wennroy.com>，架构：**Apache 反代 → 127.0.0.1:8503 → docker 容器（Next.js standalone）**。旧 Streamlit 版已停用（`streamlit_3.service` stop+disable），仅作回滚后备。

## 服务器布局（`ssh wennroy`）

| 内容 | 位置 |
| --- | --- |
| 代码仓库（master 分支） | `~/proj/bedminton_elo` |
| compose / Dockerfile | `~/proj/bedminton_elo/web/` |
| **线上数据库** | `web/data/badminton.db`（容器内 `/data/badminton.db`，volume 挂载） |
| 环境变量 | `web/.env`（`ADMIN_PASSWORD`、`PORT`、`DATABASE_URL`；**只存在于服务器，不入库**） |
| 旧 Streamlit 库 | `~/proj/bedminton_elo/badminton.db`（未动）；另有备份 `~/badminton.db.bak-20260901` |

容器 `restart: unless-stopped`，服务器重启会自动拉起。

## 日常发版流程

1. 本地：`scripts/release.sh <x.y.z>`（改写 CHANGELOG Unreleased 段 + 同步 `web/package.json` version），提交并 push 到 master。
2. 服务器上更新并重建：

```bash
ssh wennroy
cd ~/proj/bedminton_elo
git pull --ff-only origin master
cd web
docker compose up -d --build
```

3. 验证（示例）：

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://bedminton.wennroy.com/
curl -s -o /dev/null -w '%{http_code}\n' https://bedminton.wennroy.com/changelog
```

注意：

- **GitHub 偶发 TLS 断流甚至挂死**：挂死时按 Ctrl-C，用
  `git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=15 pull` 快速失败后重试。
- **只改了 CHANGELOG.md 不需要重新 build**，但因为 compose 用的是单文件 bind mount（按 inode 绑定，`git pull` 会替换文件产生新 inode），容器里看到的仍是旧内容——必须 `docker compose restart` 才生效。
- 数据库 schema 变更无需迁移脚本：`web/src/lib/db.ts` 每次开库自动执行 `schema.sql`（`CREATE TABLE IF NOT EXISTS`）。
- 没有 sudo 权限的命令（如 Apache 配置、systemctl）需要用户本人执行。

## 回滚

```bash
cd ~/proj/bedminton_elo/web && docker compose down
sudo systemctl start streamlit_3   # 需用户本人执行
```

注意：回滚后新站录入的数据在 `web/data/badminton.db` 里，旧站看不到。

## 操作数据库前

先备份再动：

```bash
cp ~/proj/bedminton_elo/web/data/badminton.db{,.bak-$(date +%Y%m%d)-<说明>}
```
