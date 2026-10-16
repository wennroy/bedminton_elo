// @vitest-environment jsdom
/**
 * /schedule/[projectId] 详情：
 * 三态分组渲染与 superseded 只读；改名（含 409 先刷新）；
 * 续排旁注「保留 X 场已打比赛，替换 Y 场待打比赛」现算；名单 <4 按钮文案切换；
 * 标记已打后卡片移入待录分组且不重复计数；checkbox 关闭不隐藏已打分组；
 * 轮询（假定时器）不覆盖输入中的改名草稿、页面隐藏时暂停。
 */
import * as React from "react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";

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

const refreshMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), refresh: refreshMock }),
}));

import { ProjectDetailClient } from "./project-detail";
import type { Fixture, ProjectDetail } from "../types";

let fixtureSeq = 0;
function fixture(overrides: Partial<Fixture> = {}): Fixture {
  fixtureSeq += 1;
  return {
    id: fixtureSeq,
    projectId: 1,
    generationId: 1,
    displayNo: fixtureSeq,
    pa1: "1",
    pa2: "2",
    pb1: "3",
    pb2: "4",
    generatedWinRate: 0.55,
    status: "pending",
    revision: 0,
    playedAt: null,
    markedAt: null,
    markedBy: null,
    matchId: null,
    scoreA: null,
    scoreB: null,
    ...overrides,
  };
}

function makeProject(overrides: Partial<ProjectDetail> = {}): ProjectDetail {
  return {
    id: 1,
    name: "周五晚场",
    model: "legacy",
    revision: 0,
    nextFixtureNo: 6,
    createdBy: "1",
    createdAt: "2026-10-01 12:00:00",
    updatedAt: "2026-10-10 13:30:00",
    members: [
      { playerId: "1", fairnessBase: 0, playedAtJoin: 0, joinedAt: null, leftAt: null, actualPlayed: 1, pendingCount: 3 },
      { playerId: "2", fairnessBase: 0, playedAtJoin: 0, joinedAt: null, leftAt: null, actualPlayed: 1, pendingCount: 3 },
      { playerId: "3", fairnessBase: 0, playedAtJoin: 0, joinedAt: null, leftAt: null, actualPlayed: 0, pendingCount: 2 },
      { playerId: "4", fairnessBase: 0, playedAtJoin: 0, joinedAt: null, leftAt: null, actualPlayed: 0, pendingCount: 2 },
    ],
    fixtures: [
      fixture({ displayNo: 1, id: 101 }),
      fixture({ displayNo: 2, id: 102 }),
      fixture({ displayNo: 3, id: 103 }),
      fixture({
        displayNo: 4,
        id: 104,
        status: "played_unscored",
        playedAt: "2026-10-09",
      }),
      fixture({
        displayNo: 5,
        id: 105,
        status: "recorded",
        playedAt: "2026-10-01",
        markedBy: "2",
        matchId: 9,
        scoreA: 21,
        scoreB: 9,
      }),
    ],
    superseded: [fixture({ displayNo: 6, id: 106, status: "superseded" })],
    latestGeneration: {
      id: 1,
      requestId: "r1",
      sourceRevision: 0,
      input: { matches: 3, avoidPlayedRematches: true, playerIds: ["1", "2", "3", "4"] },
      metrics: null,
      createdAt: "2026-10-01 12:00:00",
    },
    ...overrides,
  };
}

const NAMES: Record<string, string> = {
  "1": "甲",
  "2": "乙",
  "3": "丙",
  "4": "丁",
};

const DIRECTORY = [
  { id: 1, name: "甲" },
  { id: 2, name: "乙" },
  { id: 3, name: "丙" },
  { id: 4, name: "丁" },
  { id: 5, name: "戊" },
];

function jsonResponse(data: unknown, status = 200, ok = true) {
  return { ok, status, json: () => Promise.resolve(data) };
}

function section(title: string): HTMLElement {
  const heading = screen.getByText(title);
  const sec = heading.closest("section");
  if (!sec) throw new Error(`section not found for ${title}`);
  return sec as HTMLElement;
}

describe("ProjectDetailClient（/schedule/[projectId] 详情）", () => {
  beforeEach(() => {
    window.localStorage.setItem("badminton:myPlayerId", "1");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    vi.useRealTimers();
    window.localStorage.clear();
  });

  it("渲染三态分组、生成时胜率、录入信息与 superseded 只读态", () => {
    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );

    const pendingSection = section("待打 · 3");
    expect(within(pendingSection).getByText("第 1 场")).toBeTruthy();
    // 3 张待打卡片各带生成时胜率（原快照口径）
    expect(
      within(pendingSection).getAllByText("生成时预测 A 队胜率 55%")
    ).toHaveLength(3);
    const predictLinks = within(pendingSection).getAllByRole("link", {
      name: /当前预测/,
    });
    expect(predictLinks).toHaveLength(3);
    expect(predictLinks[0].getAttribute("href")).toBe(
      "/predict?pa1=1&pa2=2&pb1=3&pb2=4"
    );
    const scoreLinks = within(pendingSection).getAllByRole("link", {
      name: "记分",
    });
    expect(scoreLinks).toHaveLength(3);
    expect(scoreLinks[0].getAttribute("href")).toBe(
      "/schedule/1/matches/101/record"
    );
    expect(
      within(pendingSection).getAllByRole("button", { name: "标记已打" })
    ).toHaveLength(3);

    const unscoredSection = section("已打待录分 · 1");
    expect(within(unscoredSection).getByText("已于 2026-10-09 标记已打")).toBeTruthy();
    expect(within(unscoredSection).getByRole("link", { name: "补录比分" })).toBeTruthy();

    const recordedSection = section("已登记 · 1");
    expect(within(recordedSection).getByText("21 : 9")).toBeTruthy();
    expect(within(recordedSection).getByText(/乙 录入/)).toBeTruthy();

    const supersededSection = section("已被替换 · 1");
    expect(within(supersededSection).getByText("已被替换")).toBeTruthy();
    // 只读：无记分/标记/补录操作
    expect(within(supersededSection).queryByRole("button")).toBeNull();
    expect(within(supersededSection).queryByRole("link", { name: "记分" })).toBeNull();
  });

  it("成员名单展示每人真实已打/待打", () => {
    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    expect(screen.getByText("当前名单 · 4 人")).toBeTruthy();
    // 成员 1/2 统计相同，各出现一次
    expect(screen.getAllByText("已打 1 · 待打 3")).toHaveLength(2);
    expect(screen.getAllByText("已打 0 · 待打 2")).toHaveLength(2);
  });

  it("页头有配对方式说明书入口链接", () => {
    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    const link = screen.getByRole("link", { name: "配对方式" });
    expect(link.getAttribute("href")).toBe("/methodology/pairing");
    expect(link.getAttribute("title")).toBe("配对方式说明");
  });

  it("续排旁注按详情数据现算 X/Y，checkbox 回显上代选择", () => {
    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /调整人员并续排/ }));
    // X = 已打待录 1 + 已登记 1 = 2；Y = 待打 3
    expect(
      screen.getByText("保留 2 场已打比赛，替换 3 场待打比赛")
    ).toBeTruthy();
    const checkbox = screen.getByRole("checkbox", {
      name: /减少已打对阵重复/,
    }) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    // 未来场数默认当前 pending 数
    expect(screen.getByLabelText("续排场次数")).toHaveProperty("value", "3");
  });

  it("名单不足 4 人时提交按钮切换为「保存名单，暂不安排比赛」", () => {
    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /调整人员并续排/ }));
    expect(screen.getByRole("button", { name: "重新生成待打比赛" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "移除 丁" }));
    expect(
      screen.getByRole("button", { name: "保存名单，暂不安排比赛" })
    ).toBeTruthy();
    // 文案说明不影响已打数据
    expect(screen.getByText(/不会删除球员，也不会删除任何已登记的历史比赛/)).toBeTruthy();
  });

  it("标记已打成功后卡片进入待录分组且不重复计数", async () => {
    const moved = makeProject();
    const afterMark = makeProject();
    afterMark.revision = 1;
    afterMark.fixtures = afterMark.fixtures.map((f) =>
      f.id === 101
        ? { ...f, status: "played_unscored" as const, playedAt: "2026-10-11", revision: 1 }
        : f
    );
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ fixture: { id: 101 } }, 200))
      .mockResolvedValueOnce(jsonResponse({ project: afterMark, names: NAMES }));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ProjectDetailClient
        initialProject={moved}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    const pendingSection = section("待打 · 3");
    // 3 张待打卡片都有「标记已打」，点第一张（第 1 场）
    fireEvent.click(
      within(pendingSection).getAllByRole("button", { name: "标记已打" })[0]
    );

    await screen.findByText("待打 · 2");
    const unscoredSection = section("已打待录分 · 2");
    // 第 1 场只出现一次，且落在待录分组
    expect(within(unscoredSection).getByText("第 1 场")).toBeTruthy();
    expect(screen.getAllByText("第 1 场")).toHaveLength(1);
    expect(fetchMock.mock.calls[0][0]).toBe(
      "/api/schedule/projects/1/matches/101/played"
    );
  });

  it("checkbox 关闭不隐藏已打分组", () => {
    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /调整人员并续排/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /减少已打对阵重复/ }));
    expect(screen.getByText("已登记 · 1")).toBeTruthy();
    expect(screen.getByText("已打待录分 · 1")).toBeTruthy();
    // 已登记比分仍可见
    expect(screen.getByText("21 : 9")).toBeTruthy();
  });

  it("改名 409 时先刷新显示最新名称再提示", async () => {
    const latest = makeProject();
    latest.name = "别人改的名字";
    latest.revision = 4;
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse(
          { error: "revision changed", code: "revision_conflict", projectRevision: 4 },
          409,
          false
        )
      )
      .mockResolvedValueOnce(jsonResponse({ project: latest, names: NAMES }));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: "改名" }));
    fireEvent.change(screen.getByLabelText("项目名称"), {
      target: { value: "我想改的名" },
    });
    fireEvent.click(screen.getByRole("button", { name: "保存" }));

    await screen.findByText("别人改的名字");
    expect(screen.getByText(/已刷新为最新/)).toBeTruthy();
  });

  it("轮询（假定时器）不覆盖输入中的改名草稿，且页面隐藏时暂停", async () => {
    vi.useFakeTimers();
    const latest = makeProject();
    latest.revision = 9;
    latest.members = latest.members.map((m) => ({ ...m, pendingCount: 0 }));
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(jsonResponse({ project: latest, names: NAMES }))
    );
    vi.stubGlobal("fetch", fetchMock);

    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    // 打开改名输入并输入内容
    fireEvent.click(screen.getByRole("button", { name: "改名" }));
    const input = screen.getByLabelText("项目名称") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "输入到一半的草稿" } });

    // 隐藏时暂停：不发起轮询
    const hiddenSpy = vi
      .spyOn(document, "hidden", "get")
      .mockReturnValue(true);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(fetchMock).not.toHaveBeenCalled();
    hiddenSpy.mockReturnValue(false);

    // 前台轮询到达：应用新数据（成员待打归零、revision 更新），但不覆盖输入草稿
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(screen.getAllByText("已打 1 · 待打 0")).toHaveLength(2);
    expect(input.value).toBe("输入到一半的草稿");
    expect(screen.queryByText("输入到一半的草稿")).toBeNull(); // 未保存，标题不变
    hiddenSpy.mockRestore();
  });

  it("未选身份时改名与标记已打禁用并提示", () => {
    window.localStorage.clear();
    render(
      <ProjectDetailClient
        initialProject={makeProject()}
        initialNames={NAMES}
        directory={DIRECTORY}
      />
    );
    expect(screen.getByRole("button", { name: "改名" })).toHaveProperty(
      "disabled",
      true
    );
    const pendingSection = section("待打 · 3");
    const markButtons = within(pendingSection).getAllByRole("button", {
      name: "标记已打",
    });
    expect(markButtons).toHaveLength(3);
    for (const button of markButtons) {
      expect(button).toHaveProperty("disabled", true);
    }
    expect(screen.getByText(/请先在右上角选择球员身份/)).toBeTruthy();
  });
});
