"use client";

import { useRatingBoundaryRefresh } from "@/lib/use-rating-boundary-refresh";

/**
 * 极薄客户端宿主：页面是服务端组件，无法直接调 hook；该组件消费
 * loadRatingView 的 nextBoundary，到周/季边界或页面重新可见时
 * router.refresh()。legacy/unavailable 传 null 不设定时。渲染 null。
 */
export function RatingBoundaryRefresh({
  nextBoundary,
}: {
  nextBoundary: string | null;
}) {
  useRatingBoundaryRefresh(nextBoundary);
  return null;
}
