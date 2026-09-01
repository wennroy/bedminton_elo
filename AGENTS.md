# AGENTS.md

## Changelog 与 Versioning

- 版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)：大版本 = 重要新功能或不兼容变化，小版本 = 普通功能更新，补丁 = 修复。
- 所有更新记录在仓库根目录 `CHANGELOG.md`（[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式）——这是**唯一数据源**。
- **开发期**：把用户可感知的变化追加到 `## [Unreleased]` 段。可以直接起草在 `### What's New` 下，也可以用 `### Added / Changed / Fixed` 小节记细节（网页不展示这些小节）。
- **发版前**（大版本必须，小版本建议）：把 `## [Unreleased]` 改名为 `## [x.y.z] - YYYY-MM-DD`，确保段内有 `### What's New` —— 用面向球友的自然语言描述新功能和变化，不写技术实现细节；顶部新开一个空的 `## [Unreleased]`；同步 `web/package.json` 的 `version`。
- 网站的 `/changelog` 页面在运行时直接解析根目录 `CHANGELOG.md` 的 What's New 小节渲染时间线。**不要**为它另建数据文件或数据库表；保持标题格式（`## [x.y.z] - YYYY-MM-DD`、`## [Unreleased]`、`### What's New`）不变，否则解析会坏。
- Docker 部署通过 volume 把根目录 `CHANGELOG.md` 挂进容器（`../CHANGELOG.md:/app/CHANGELOG.md:ro`），无需重新 build 镜像即可更新日志内容。
- 注意：`web/AGENTS.md` 是 `next dev` 自动生成的文件，与本文件无关，不要混用。
