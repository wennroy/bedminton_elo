#!/usr/bin/env bash
# 发版脚本(对应根目录 AGENTS.md 的发版约定):
#   scripts/release.sh <x.y.z>
# 会做的事:
#   1. CHANGELOG.md:`## [Unreleased]` 改名为 `## [x.y.z] - <今天>`,顶部新开空 Unreleased 段
#      (要求 Unreleased 的 What's New 下至少有一条 "- " 条目,否则拒绝执行)
#   2. web/package.json 的 version 同步为 x.y.z
# 不做 git 提交;完成后会打印建议的提交 / 推送 / 服务器操作命令。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHANGELOG="$ROOT/CHANGELOG.md"
PKG="$ROOT/web/package.json"
PLACEHOLDER='<!-- 发版前在这里填写面向球友的新功能描述 -->'

VERSION="${1:-}"
VERSION="${VERSION#v}" # 允许 v1.2.0 或 1.2.0
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "用法: $0 <x.y.z>   例如: $0 1.2.0" >&2
  exit 1
fi

[[ -f "$CHANGELOG" ]] || { echo "错误: 找不到 $CHANGELOG" >&2; exit 1; }
[[ -f "$PKG" ]] || { echo "错误: 找不到 $PKG" >&2; exit 1; }

if grep -q "^## \[$VERSION\]" "$CHANGELOG"; then
  echo "错误: CHANGELOG 里已存在 ## [$VERSION],不能重复发版" >&2
  exit 1
fi
if ! grep -q '^## \[Unreleased\]$' "$CHANGELOG"; then
  echo "错误: CHANGELOG 缺少 ## [Unreleased] 段,格式可能已被破坏" >&2
  exit 1
fi

# Unreleased 段(到下一个 ## 标题为止)必须包含 What's New 小节和至少一条条目
UNRELEASED_BODY="$(awk '/^## \[Unreleased\]$/{f=1;next} /^## \[/{f=0} f' "$CHANGELOG")"
if ! grep -q "^### What's New" <<< "$UNRELEASED_BODY"; then
  echo "错误: Unreleased 段缺少 ### What's New 小节" >&2
  exit 1
fi
if ! grep -q '^- ' <<< "$UNRELEASED_BODY"; then
  echo "错误: Unreleased 的 What's New 还没有条目,先把本期变化写进去再发版" >&2
  exit 1
fi

TODAY="$(date +%F)"
OLD_VERSION="$(grep -m1 '"version"' "$PKG" | sed 's/[^0-9.]//g')"

# 1) CHANGELOG:替换 Unreleased 标题为「新空 Unreleased + ## [x.y.z] - 今天」,
#    旧段内容原样保留到新版本号下,只去掉占位注释行
awk -v ver="$VERSION" -v date="$TODAY" -v whatsnew="### What's New" -v placeholder="$PLACEHOLDER" '
  /^## \[Unreleased\]$/ && !done {
    print "## [Unreleased]"
    print ""
    print whatsnew
    print ""
    print placeholder
    print ""
    print "## [" ver "] - " date
    done = 1
    inold = 1
    next
  }
  inold && /^## \[/ { inold = 0 }
  inold && $0 == placeholder { next }
  { print }
' "$CHANGELOG" > "$CHANGELOG.tmp"
mv "$CHANGELOG.tmp" "$CHANGELOG"

# 2) package.json version(用临时文件写法,BSD/GNU sed 都适用)
sed 's/"version": "[^"]*"/"version": "'"$VERSION"'"/' "$PKG" > "$PKG.tmp"
mv "$PKG.tmp" "$PKG"

echo "✅ CHANGELOG.md:## [Unreleased] → ## [$VERSION] - $TODAY(顶部已新开空 Unreleased)"
echo "✅ web/package.json:$OLD_VERSION → $VERSION"
echo ""
echo "接下来:"
echo "  git add CHANGELOG.md web/package.json"
echo "  git commit -m \"Release v$VERSION\""
echo "  git push origin HEAD:master"
echo "  # 服务器(CHANGELOG 是单文件挂载,restart 即可,不用 --build):"
echo "  #   ssh wennroy 'cd ~/proj/bedminton_elo && git pull --ff-only origin master && cd web && docker compose restart'"
