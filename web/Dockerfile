# syntax=docker/dockerfile:1
FROM node:22-slim AS base
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

FROM base AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
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
COPY --from=builder /app/src/lib/db.ts ./src/lib/db.ts
COPY --from=builder /app/src/lib/schema.sql ./src/lib/schema.sql
EXPOSE 3000
CMD ["sh", "-c", "tsx scripts/migrate-legacy.ts && node server.js"]
