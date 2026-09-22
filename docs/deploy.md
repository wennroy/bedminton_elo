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

先备份再动。应用开库使用 **WAL 模式**（`web/src/lib/db.ts`），提交过的数据可能仍在 `badminton.db-wal` 里，**只复制主 `badminton.db` 文件不是完整备份**。使用 SQLite 在线 backup（一致性快照），容器内自带 `better-sqlite3`，无需在服务器装额外工具：

```bash
cd ~/proj/bedminton_elo/web
docker compose exec web node -e "const db=require('better-sqlite3')('/data/badminton.db'); db.backup('/data/badminton.db.bak-' + new Date().toISOString().slice(0,10) + '-<说明>').then(() => { console.log('backup done'); db.close(); });"
```

备份文件落在 `web/data/`（与数据库同目录）。服务器装有 sqlite3 CLI 时等价命令：`sqlite3 ~/proj/bedminton_elo/web/data/badminton.db ".backup '~/proj/bedminton_elo/web/data/badminton.db.bak-<说明>'"`。

## 新版评分（Glicko-2）配置、切换与回退

> 现状：站点默认模型仍为 **Legacy**；页面顶部「新版 / Legacy」切换控件已上线，用户可显式查看新版。以下步骤用于生产切换默认模型，需单独的发版指令执行，本说明只准备操作步骤。

配置存放与语义：

- 评分配置持久化在数据库 `meta` 表（单键 `ratings.config.v1`），容器重建、重启都不会丢失；**未初始化配置时站点自动按 Legacy 运行**，缺配置不影响启动。
- 首次 `init` 时冻结首赛季起点：默认取初始化当天（上海时区）之后第一个季度起点，可 `--first-season-start YYYY-MM-DD` 显式指定（须为季度首日）。之后的读取、重启、月份变化都不会移动该起点；更换参数版本时未显式指定也会保留原值。
- `activate` 只切换默认模型，不删比赛、不动新版历史与最近成功快照；目标与当前一致时为幂等 no-op。

发布前步骤（按顺序）：

1. **备份数据库**（见上一节，必须是在线一致性备份）。
2. **固定配置**（先 `--dry-run` 预览差异，不写库）：

   ```bash
   cd ~/proj/bedminton_elo/web
   docker compose exec web tsx scripts/rating-config.ts init --db /data/badminton.db --dry-run
   docker compose exec web tsx scripts/rating-config.ts init --db /data/badminton.db
   docker compose exec web tsx scripts/rating-config.ts status --db /data/badminton.db
   ```

   记录输出中的 `firstSeasonStart`（冻结值），回放报告与切换都使用它。
3. **回放校验报告（发布前必须完成）**：用生产库匿名只读快照在本地生成，JSON 摘要录入 `docs/ratings-validation.md`；数据集与带姓名报告不入库。

   ```bash
   cd web && tsx scripts/backtest-ratings.ts --db <生产库匿名只读快照路径> --as-of <ISO 时点> --first-season-start <配置冻结值> --out <报告输出路径>
   ```

4. **切换默认模型到新版**，随后检查页面与分享图：

   ```bash
   docker compose exec web tsx scripts/rating-config.ts activate --model glicko2 --db /data/badminton.db
   ```

   检查项：首页 / 趋势 / 排行榜 / 个人页的新版数据正常渲染，周报网页与分享图（OG 图片）同周数值一致；顶部切换回 Legacy 后旧数据不变。
5. **回退（只切默认，不删数据）**：

   ```bash
   docker compose exec web tsx scripts/rating-config.ts activate --model legacy --db /data/badminton.db
   ```

   比赛数与原始赛果不受影响，新版配置与历史保留，确认后可再切回。

版本说明：本次变更为重要新功能，发版预计按大版本 **2.0.0** 执行（发版时以 `scripts/release.sh` 与 CHANGELOG 实际核对为准）。
