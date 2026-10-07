import type { LocalDate } from "./types";

const YEAR_LEN = 4;

/**
 * 跨年边界:有序日期序列中年份相对前一项变大的下标集合
 * (同日多事件只标记首个;语义:「从这里开始年份变更」)。
 */
export function yearBoundaryIndices(dates: readonly LocalDate[]): ReadonlySet<number> {
  const boundaries = new Set<number>();
  for (let i = 1; i < dates.length; i++) {
    if (dates[i].slice(0, YEAR_LEN) !== dates[i - 1].slice(0, YEAR_LEN)) {
      boundaries.add(i);
    }
  }
  return boundaries;
}

/**
 * X 轴刻度抽稀:按容器宽度限制最大刻度数并均匀间隔选取,跨年边界始终保留。
 * recharts 的 minTickGap 自动抽稀不保证边界点幸存(边界日期可能被跳过,
 * 年份标注随之消失),所以由调用方显式给出 ticks。
 * 边界点插入在所属区间头部刻度之后,局部间距可能变小,最多增加「跨年次数」个刻度。
 */
export function pickAxisTicks<T>(
  values: readonly T[],
  {
    width,
    isBoundary,
    minGap = 44,
  }: {
    width: number;
    isBoundary: (value: T) => boolean;
    minGap?: number;
  }
): T[] {
  const n = values.length;
  const maxTicks = Math.max(2, Math.floor(width / minGap));
  if (n <= maxTicks) return [...values];
  const step = Math.ceil(n / maxTicks);
  const picked: T[] = [];
  for (let i = 0; i < n; i += step) {
    picked.push(values[i]);
    for (let j = i + 1; j < Math.min(i + step, n); j++) {
      if (isBoundary(values[j])) picked.push(values[j]);
    }
  }
  return picked;
}
