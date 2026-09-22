import { revalidatePath } from "next/cache";

/**
 * 评分相关的 revalidate 路径集合：所有比赛/球员写入口（录入、撤回、改分、
 * 合并、改名、删球员）统一调用 revalidateRatingPages()，保证每个消费
 * 评分数据或比赛事实的页面在新写入后同步失效——只刷首页不够，
 * 周界/季界后各页也必须拿到重放后的新投影。
 *
 * 每条路在集合里的原因：
 * - "/"：首页展示当前评分榜与最近比赛结果。
 * - "/trends"：趋势图直接消费评分历史投影（逐场 Estimated/周 Final/重置）。
 * - "/players"：球员目录消费当前分、RD 与排名。
 * - "/players/[id]"（page）：个人档案页消费单人历史曲线与峰值；
 *   动态段一条 revalidatePath 覆盖全部档案页，无需逐个 id。
 * - "/weekly"：周报消费周 Final/Estimated、区段与事实统计。
 * - "/predict"：预测页消费当前评分状态与胜率。
 * - "/schedule"：赛程/配对优化消费同一评分工作状态与概率。
 * - "/matches"：全部比赛浏览页消费比赛事实列表。
 *
 * 刻意不含 "/signup"：签到与评分无关，仍由 signups 路由自行刷新。
 */
export type RatingRevalidateEntry =
  | { readonly path: string; readonly type?: undefined }
  | { readonly path: string; readonly type: "page" | "layout" };

export const RATING_REVALIDATE_PATHS: readonly RatingRevalidateEntry[] = [
  { path: "/" },
  { path: "/trends" },
  { path: "/players" },
  { path: "/players/[id]", type: "page" },
  { path: "/weekly" },
  { path: "/predict" },
  { path: "/schedule" },
  { path: "/matches" },
];

/** 一次失效全部评分相关页面；写入口在成功变更后调用。 */
export function revalidateRatingPages(): void {
  for (const entry of RATING_REVALIDATE_PATHS) {
    if (entry.type === undefined) {
      revalidatePath(entry.path);
    } else {
      revalidatePath(entry.path, entry.type);
    }
  }
}
