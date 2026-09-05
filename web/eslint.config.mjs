import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { FlatCompat } from "@eslint/eslintrc";

const __dirname = dirname(fileURLToPath(import.meta.url));

// eslint-config-next@15 的 core-web-vitals/typescript 入口仍是 legacy
// eslintrc 格式,需经 FlatCompat 包装;直接 spread 会在加载期抛
// "nextVitals is not iterable"。(Next 16 起才是原生 flat config)
const compat = new FlatCompat({ baseDirectory: __dirname });

const eslintConfig = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: [".next/**", ".next-dev/**", "out/**", "build/**", "next-env.d.ts"],
  },
];

export default eslintConfig;
