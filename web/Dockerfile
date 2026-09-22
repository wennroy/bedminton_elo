# syntax=docker/dockerfile:1
FROM node:22-slim AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
# 国内服务器:npm 安装与 corepack 下载 pnpm 都走 npmmirror
ENV npm_config_registry=https://registry.npmmirror.com
ENV COREPACK_NPM_REGISTRY=https://registry.npmmirror.com
RUN corepack enable

FROM base AS deps
WORKDIR /app
# better-sqlite3 prebuilt binary download may fail (GitHub releases unreachable);
# fall back to source compile — needs the toolchain.
# Debian 源也切腾讯云镜像,否则国内 apt-get 极慢
RUN sed -i 's|deb.debian.org|mirrors.cloud.tencent.com|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends python3 make g++ && rm -rf /var/lib/apt/lists/*
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

FROM base AS builder
WORKDIR /app
COPY . .
# Replace host node_modules (possibly different platform) with the deps stage.
COPY --from=deps /app/node_modules ./node_modules
ENV NEXT_TELEMETRY_DISABLED=1
# DATABASE_URL must not trigger a real DB connection during build; db.ts is lazy.
RUN pnpm build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV DATABASE_URL=/data/badminton.db
# ADMIN_PASSWORD should be provided at runtime via compose env.
RUN npm install -g tsx
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/scripts/migrate-legacy.ts ./scripts/migrate-legacy.ts
# 评分配置 CLI（docker compose exec web tsx scripts/rating-config.ts status/init/activate）。
# 只复制其纯 TS 依赖链（相对导入运行时解析；ratings/types 仅类型导入也一并带上），
# 不含测试文件。
COPY --from=builder /app/scripts/rating-config.ts ./scripts/rating-config.ts
COPY --from=builder /app/src/lib/rating-config.ts ./src/lib/rating-config.ts
COPY --from=builder /app/src/lib/ratings/calendar.ts ./src/lib/ratings/calendar.ts
COPY --from=builder /app/src/lib/ratings/config.ts ./src/lib/ratings/config.ts
COPY --from=builder /app/src/lib/ratings/types.ts ./src/lib/ratings/types.ts
COPY --from=builder /app/src/lib/db.ts ./src/lib/db.ts
COPY --from=builder /app/src/lib/schema.sql ./src/lib/schema.sql
EXPOSE 3000
CMD ["sh", "-c", "tsx scripts/migrate-legacy.ts && node server.js"]
