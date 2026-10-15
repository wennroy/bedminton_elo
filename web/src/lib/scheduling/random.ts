/**
 * 标准无符号 Mulberry32。
 *
 * 注意：scheduler.ts 里的旧实现用算术右移 `>>`，输出到不了 [0.5,1)
 * （采样上限约 0.49999，还可能为负）。本实现全部用无符号右移 `>>>`，
 * 输出严格落在 [0,1)。已用权威参考实现逐位核对（见 random.test.ts）。
 */

export type PRNG = () => number;

const UINT32_DIVISOR = 4294967296; // 2^32
const MAX_SEED = 0xffffffff;

export function createRng(seed: number): PRNG {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > MAX_SEED) {
    throw new Error(`seed 必须是 0..${MAX_SEED} 的整数，实际 ${seed}`);
  }
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), t | 1);
    r = (r ^ (r + Math.imul(r ^ (r >>> 7), r | 61))) >>> 0;
    return ((r ^ (r >>> 14)) >>> 0) / UINT32_DIVISOR;
  };
}

/**
 * 从 masterSeed 派生第 restartIndex 次重启的确定性种子（无符号 32 位）。
 * splitmix 风格 finalizer：同输入必同输出，不同 restartIndex 分布独立。
 */
export function deriveSeed(masterSeed: number, restartIndex: number): number {
  if (!Number.isSafeInteger(masterSeed) || masterSeed < 0 || masterSeed > MAX_SEED) {
    throw new Error(`masterSeed 必须是 0..${MAX_SEED} 的整数，实际 ${masterSeed}`);
  }
  if (!Number.isSafeInteger(restartIndex) || restartIndex < 0) {
    throw new Error(`restartIndex 必须是非负安全整数，实际 ${restartIndex}`);
  }
  let h = (masterSeed >>> 0) ^ Math.imul(restartIndex + 1, 0x9e3779b9);
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad);
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97);
  h = (h ^ (h >>> 15)) >>> 0;
  return h >>> 0;
}
