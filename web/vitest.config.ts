import { defineConfig, configDefaults } from "vitest/config";
import path from "path";

export default defineConfig({
  // tsconfig 的 jsx: "preserve" 是给 Next 用的;vitest 的 esbuild 默认 classic
  // 转换会注入 React.createElement 导致 "React is not defined",这里对齐为
  // automatic runtime,才能直接 import .tsx 路由(如 api/og/weekly)做测试。
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
    // E2E 旅程（真实服务器+浏览器）只由 vitest.e2e.config.ts 承载，
    // 单测跑法（pnpm test）不收集，避免拖重与端口占用。
    exclude: [...configDefaults.exclude, "tests/e2e/**"],
  },
});
