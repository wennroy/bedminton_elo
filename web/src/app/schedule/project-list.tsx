"use client";

import * as React from "react";
import Link from "next/link";
import { ChevronRight, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMyPlayerId } from "./use-my-player-id";
import type { ProjectSummary } from "./types";

const PAGE_SIZE = 30;

/**
 * sqlite datetime('now') 存的是 UTC "YYYY-MM-DD HH:MM:SS"（无时区标记），
 * 统一补 Z 按 UTC 解析后转本地墙钟展示；解析失败原样返回。
 */
export function formatProjectUpdatedAt(value: string): string {
  const iso = value.includes("T") ? value : value.replace(" ", "T");
  const withZone = /(Z|[+-]\d{2}:?\d{2})$/.test(iso) ? iso : `${iso}Z`;
  const d = new Date(withZone);
  if (Number.isNaN(d.getTime())) return value;
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${d.getMonth() + 1}月${d.getDate()}日 ${hh}:${mm}`;
}

interface ProjectListProps {
  initialProjects: ProjectSummary[];
  initialNextCursor: number | null;
}

export function ProjectList({ initialProjects, initialNextCursor }: ProjectListProps) {
  const myId = useMyPlayerId();
  const [projects, setProjects] = React.useState(initialProjects);
  const [nextCursor, setNextCursor] = React.useState(initialNextCursor);
  const [loadingMore, setLoadingMore] = React.useState(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  async function loadMore() {
    if (nextCursor === null || loadingMore) return;
    setLoadingMore(true);
    setLoadError(null);
    try {
      const res = await fetch(
        `/api/schedule/projects?limit=${PAGE_SIZE}&cursor=${nextCursor}`
      );
      if (!res.ok) {
        setLoadError("加载失败，请重试");
        return;
      }
      const data = await res.json();
      setProjects((prev) => [...prev, ...data.projects]);
      setNextCursor(data.nextCursor ?? null);
    } catch {
      setLoadError("网络错误，请重试");
    } finally {
      setLoadingMore(false);
    }
  }

  const newProjectButton = (
    <Button asChild={myId !== null} disabled={myId === null} size="lg">
      {myId !== null ? (
        <Link href="/schedule/new">
          <Plus />
          新建配对项目
        </Link>
      ) : (
        <span title="请先在右上角选择球员身份">新建配对项目</span>
      )}
    </Button>
  );

  if (projects.length === 0) {
    return (
      <div className="flex flex-col items-center gap-4 rounded-2xl border border-dashed border-border bg-card px-6 py-14 text-center shadow-sm">
        <p className="text-sm font-medium text-card-foreground">
          还没有配对项目
        </p>
        <p className="max-w-sm text-xs leading-relaxed text-muted-foreground">
          选好今晚的球员，生成几场势均力敌的双打对阵；项目保存在服务器上，
          所有球友都能查看进度、标记已打并录入比分。
        </p>
        {myId === null && (
          <p className="text-xs text-muted-foreground">
            请先在右上角选择球员身份，再新建项目。
          </p>
        )}
        {newProjectButton}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">
          共 {projects.length} 个项目{nextCursor !== null ? "（还有更早的）" : ""}
        </p>
        {newProjectButton}
      </div>

      <ul className="flex flex-col gap-3">
        {projects.map((p) => (
          <li key={p.id}>
            <Link
              href={`/schedule/${p.id}`}
              className="block rounded-2xl border border-border bg-card p-4 shadow-sm transition-colors hover:border-primary/50"
            >
              <div className="flex items-center justify-between gap-3">
                <h2 className="min-w-0 truncate text-base font-bold text-card-foreground">
                  {p.name}
                </h2>
                <span className="flex shrink-0 items-center gap-1 text-xs text-muted-foreground">
                  {formatProjectUpdatedAt(p.updatedAt)}
                  <ChevronRight className="size-3.5" />
                </span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span>{p.activePlayers} 人在名单</span>
                <span>待打 {p.pendingCount}</span>
                <span>已打待录分 {p.playedUnscoredCount}</span>
                <span>已登记 {p.recordedCount}</span>
              </div>
            </Link>
          </li>
        ))}
      </ul>

      {nextCursor !== null && (
        <div className="flex flex-col items-center gap-2">
          <Button
            variant="outline"
            onClick={loadMore}
            disabled={loadingMore}
            className="w-full sm:w-auto"
          >
            {loadingMore ? "加载中…" : "加载更早的项目"}
          </Button>
          {loadError && <p className="text-xs text-destructive">{loadError}</p>}
        </div>
      )}
    </div>
  );
}
