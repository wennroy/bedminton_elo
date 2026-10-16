// @vitest-environment jsdom
/**
 * /methodology/pairing 配对方式说明书：
 * 关键章节与出场公平数值例渲染；返回与计分方式交叉链接正确；
 * 配对列表页与新建页均有说明书入口链接（详情页断言见 project-detail.test.tsx）。
 */
import * as React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

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

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), refresh: vi.fn() }),
}));

vi.mock("@/lib/schedule-projects", () => ({
  listProjects: () => ({ projects: [], nextCursor: null }),
}));

vi.mock("@/lib/repo", () => ({
  listPlayers: () => [],
}));

import MethodologyPairingPage from "./page";
import SchedulePage from "../../schedule/page";
import NewScheduleProjectPage from "../../schedule/new/page";

describe("/methodology/pairing 配对方式说明书", () => {
  afterEach(() => {
    cleanup();
  });

  it("渲染关键章节与出场公平数值例", () => {
    render(<MethodologyPairingPage />);

    expect(
      screen.getByRole("heading", { name: "配对方式" })
    ).toBeTruthy();
    expect(screen.getByText("现行：quota-local-v1 配对")).toBeTruthy();
    for (const title of [
      "场次状态",
      "出场公平",
      "人员变动与续排",
      "「减少已打对阵重复」",
      "配对质量",
      "配额怎么分：最小化平方和",
      "对阵怎么挑：质量分 J",
      "生成时胜率快照与可复现",
      "记分、防重复与撤回",
    ]) {
      expect(screen.getByText(title)).toBeTruthy();
    }
    // 数值例：10 人 8 场 → 2 人 4 场、8 人 3 场，方差下限 0.16
    expect(screen.getByText(/2 人打 4 场、8 人打 3 场，σ² 下限 = 0\.16/)).toBeTruthy();
    // 公式：配额平方和目标与边际代价、质量分 J 与权重
    expect(screen.getByText(/F = Σᵢ \(cᵢ \+ xᵢ\)²/)).toBeTruthy();
    expect(screen.getByText(/ΔFᵢ = 2\(cᵢ \+ xᵢ\) \+ 1/)).toBeTruthy();
    expect(
      screen.getByText(/J = 4·B \+ 2·P \+ 0\.5·O \+ 2·E \+ 3·H/)
    ).toBeTruthy();
  });

  it("返回链接指向 /schedule，交叉链接指向计分方式", () => {
    render(<MethodologyPairingPage />);

    expect(screen.getByRole("link", { name: "返回" }).getAttribute("href")).toBe(
      "/schedule"
    );
    const crossLinks = screen.getAllByRole("link", { name: /计分方式/ });
    expect(crossLinks.length).toBeGreaterThan(0);
    for (const link of crossLinks) {
      expect(link.getAttribute("href")).toBe("/methodology/glicko2");
    }
  });

  it("配对列表页头部有说明书入口链接", async () => {
    render(await SchedulePage());

    const link = screen.getByRole("link", { name: "配对方式" });
    expect(link.getAttribute("href")).toBe("/methodology/pairing");
    expect(link.getAttribute("title")).toBe("配对方式说明");
  });

  it("新建配对页头部有说明书入口链接", () => {
    render(<NewScheduleProjectPage />);

    const link = screen.getByRole("link", { name: "配对方式" });
    expect(link.getAttribute("href")).toBe("/methodology/pairing");
    expect(link.getAttribute("title")).toBe("配对方式说明");
  });
});
