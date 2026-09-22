"use client";

import * as React from "react";
import { useRouter } from "next/navigation";

/**
 * 距 nextBoundary 还剩多少毫秒后应刷新；nextBoundary 为 null（legacy、
 * unavailable）或非法 ISO 时返回 null（调用方不设定时）。已到期的边界回 0，
 * 表示应尽快刷新一次。纯函数，便于测试。
 */
export function boundaryRefreshDelayMs(
  nextBoundary: string | null,
  nowMs: number
): number | null {
  if (nextBoundary === null) return null;
  const target = Date.parse(nextBoundary);
  if (Number.isNaN(target)) return null;
  return Math.max(target - nowMs, 0);
}

/**
 * lib/ 下首个客户端 hook：消费 loadRatingView 的 nextBoundary。
 * 到达周/季边界或页面重新可见时 router.refresh()，由服务端按当时 asOf
 * 重新结算——不在客户端结算、不触碰任何输入框焦点：
 * - nextBoundary 为 null（legacy/unavailable）时不设定时；
 * - 同一次挂载中，同一个 nextBoundary 值只自动刷新一次：stale 旧快照的
 *   边界可能已经过去，到期即 0 毫秒，去重避免到期定时器循环刷新；
 * - 退出时清理定时器与 visibilitychange 监听。
 */
export function useRatingBoundaryRefresh(nextBoundary: string | null): void {
  const router = useRouter();
  const firedForRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    if (nextBoundary === null) return;
    if (boundaryRefreshDelayMs(nextBoundary, Date.now()) === null) return;

    let timer: number | null = null;
    const clearTimer = () => {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
    };
    const refresh = () => {
      if (firedForRef.current === nextBoundary) return;
      firedForRef.current = nextBoundary;
      router.refresh();
    };
    const arm = () => {
      const delay = boundaryRefreshDelayMs(nextBoundary, Date.now());
      if (delay === null) return;
      clearTimer();
      timer = window.setTimeout(refresh, delay);
    };

    // 边界已过（如 stale 旧快照）：挂载时尝试恢复一次；同一边界不重复。
    if (boundaryRefreshDelayMs(nextBoundary, Date.now()) === 0) refresh();
    else arm();

    const handleVisibilityChange = () => {
      if (document.visibilityState !== "visible") return;
      if (firedForRef.current === nextBoundary) return;
      const delay = boundaryRefreshDelayMs(nextBoundary, Date.now());
      if (delay === null) return;
      if (delay === 0) refresh();
      else arm();
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      clearTimer();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [nextBoundary, router]);
}
