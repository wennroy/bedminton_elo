import { defineConfig } from "vitest/config";
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
  },
});
