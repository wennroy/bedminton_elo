import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // dev 与 build 分目录,避免 build 踩坏 dev 的缓存
  distDir: process.env.NODE_ENV === "development" ? ".next-dev" : ".next",
  output: "standalone",
  eslint: {
    ignoreDuringBuilds: true,
  },
  serverExternalPackages: ["better-sqlite3"],
};

export default nextConfig;
