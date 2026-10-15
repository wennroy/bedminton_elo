"use client";

import * as React from "react";
import { getMyPlayerId } from "@/lib/identity";

/**
 * 当前所选球员身份（挂载时读取一次）。
 * AppShell 已在无身份时强制弹出选择，这里兜底读取供操作按钮禁用判断。
 */
export function useMyPlayerId(): number | null {
  const [id, setId] = React.useState<number | null>(null);
  React.useEffect(() => {
    setId(getMyPlayerId());
  }, []);
  return id;
}
