// @vitest-environment jsdom
/**
 * /schedule 项目列表：
 * 渲染名称/更新时间/人数/三态计数；空态引导新建；
 * 游标分页「加载更早」；未选身份时新建按钮禁用并提示。
 */
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

vi.mock("next/link", async () => {
  const React = await import("react");
  return {
    default: ({
      href,
      children,
      ...rest
    }: {
      href: string;
      children: React.ReactNode;
    }) => React.createElement("a", { href, ...rest }, children),
  };
});

import {
  ProjectList,
  formatProjectUpdatedAt,
} from "./project-list";
import type { ProjectSummary } from "./types";

function summary(overrides: Partial<ProjectSummary> = {}): ProjectSummary {
  return {
    id: 1,
    name: "周五晚场",
    model: "legacy",
    revision: 0,
    createdBy: "1",
    createdAt: "2026-10-10 12:00:00",
    updatedAt: "2026-10-10 13:30:00",
    activePlayers: 6,
    pendingCount: 3,
    playedUnscoredCount: 1,
    recordedCount: 2,
    ...overrides,
  };
}

describe("ProjectList（/schedule 项目列表）", () => {
  beforeEach(() => {
    window.localStorage.setItem("badminton:myPlayerId", "1");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("渲染名称、更新时间与三态计数，行链接到详情", () => {
    render(
      <ProjectList
        initialProjects={[
          summary(),
          summary({
            id: 2,
            name: "周日下午",
            activePlayers: 4,
            pendingCount: 0,
            playedUnscoredCount: 0,
            recordedCount: 4,
          }),
        ]}
        initialNextCursor={null}
      />
    );
    expect(screen.getByText("周五晚场")).toBeTruthy();
    expect(screen.getByText("周日下午")).toBeTruthy();
    expect(screen.getByText("6 人在名单")).toBeTruthy();
    expect(screen.getByText("待打 3")).toBeTruthy();
    expect(screen.getByText("已打待录分 1")).toBeTruthy();
    expect(screen.getByText("已登记 2")).toBeTruthy();
    expect(screen.getByText("4 人在名单")).toBeTruthy();
    expect(screen.getByText("待打 0")).toBeTruthy();
    expect(screen.getByText("已登记 4")).toBeTruthy();
    const row = screen.getByText("周五晚场").closest("a");
    expect(row?.getAttribute("href")).toBe("/schedule/1");
  });

  it("空状态引导新建，已选身份时新建入口可用", () => {
    render(<ProjectList initialProjects={[]} initialNextCursor={null} />);
    expect(screen.getByText("还没有配对项目")).toBeTruthy();
    const cta = screen.getByRole("link", { name: /新建配对项目/ });
    expect(cta.getAttribute("href")).toBe("/schedule/new");
  });

  it("未选身份时新建按钮禁用并提示先选择身份", () => {
    window.localStorage.clear();
    render(<ProjectList initialProjects={[]} initialNextCursor={null} />);
    const cta = screen.getByRole("button", { name: /新建配对项目/ });
    expect(cta).toHaveProperty("disabled", true);
    expect(screen.getByText(/请先在右上角选择球员身份/)).toBeTruthy();
  });

  it("有下一页时点击「加载更早」按游标追加", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () =>
        Promise.resolve({
          projects: [summary({ id: 3, name: "更早的项目" })],
          nextCursor: null,
        }),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ProjectList
        initialProjects={[summary()]}
        initialNextCursor={1}
      />
    );
    expect(screen.queryByText("更早的项目")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "加载更早的项目" }));

    await screen.findByText("更早的项目");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/schedule/projects?limit=30&cursor=1"
    );
    expect(screen.queryByRole("button", { name: "加载更早的项目" })).toBeNull();
  });

  it("formatProjectUpdatedAt：UTC sqlite 时间转本地墙钟", () => {
    const formatted = formatProjectUpdatedAt("2026-10-10 13:30:00");
    expect(formatted).toMatch(/^\d{1,2}月\d{1,2}日 \d{2}:\d{2}$/);
    // 已带时区/无法解析的输入原样或安全处理
    expect(formatProjectUpdatedAt("not-a-date")).toBe("not-a-date");
  });
});
