import { defineConfig } from "vitest/config";
import path from "path";

/**
 * E2E 旅程测试独立配置：真实服务器（next start）+ 真实 Chromium。
 * 与单测分离——pnpm test 跑 719 单测/集成，pnpm test:e2e 跑旅程。
 * 串行执行（fileParallelism: false）：每个旅程独占服务器端口与浏览器。
 */
export default defineConfig({
  esbuild: {
    jsx: "automatic",
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    globals: true,
    environment: "node",
    pool: "forks",
    fileParallelism: false,
    include: ["tests/e2e/**/*.test.ts"],
    testTimeout: 240000,
    hookTimeout: 240000,
  },
});
