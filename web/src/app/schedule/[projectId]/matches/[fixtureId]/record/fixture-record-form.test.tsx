// @vitest-environment jsdom
/**
 * /schedule/[projectId]/matches/[fixtureId]/record 项目记分：
 * 固定阵容（无换人/交换双方）；比分输入与提交（expectedFixtureRevision 随行）；
 * 201 → 已登记视图（含 MatchRatingField 结果 + 返回项目入口，无「同阵容再记一场」）；
 * 已登记/已被替换打开即为只读；提交 409 already_recorded → 刷新为只读并显示现状。
 */
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

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

import { FixtureRecordForm } from "./fixture-record-form";
import type { Fixture } from "../../../../types";

function fixture(overrides: Partial<Fixture> = {}): Fixture {
  return {
    id: 101,
    projectId: 1,
    generationId: 1,
    displayNo: 3,
    pa1: "1",
    pa2: "2",
    pb1: "3",
    pb2: "4",
    generatedWinRate: 0.6,
    status: "pending",
    revision: 2,
    playedAt: null,
    markedAt: null,
    markedBy: null,
    matchId: null,
    scoreA: null,
    scoreB: null,
    ...overrides,
  };
}

const NAMES: Record<string, string> = {
  "1": "甲",
  "2": "乙",
  "3": "丙",
  "4": "丁",
};

const SCORE_RESPONSE = {
  id: 55,
  fixture: fixture({ status: "recorded", scoreA: 21, scoreB: 9, revision: 3 }),
  before: [
    { id: 1, name: "甲", elo: 1000 },
    { id: 2, name: "乙", elo: 1000 },
    { id: 3, name: "丙", elo: 1000 },
    { id: 4, name: "丁", elo: 1000 },
  ],
  after: [
    { id: 1, name: "甲", elo: 1008 },
    { id: 2, name: "乙", elo: 1008 },
    { id: 3, name: "丙", elo: 992 },
    { id: 4, name: "丁", elo: 992 },
  ],
  legacyRatingsAvailable: true,
  rating: {
    state: "not_effective",
    model: "glicko2",
    reason: "rating config not initialized",
  },
};

describe("FixtureRecordForm（项目记分）", () => {
  beforeEach(() => {
    window.localStorage.setItem("badminton:myPlayerId", "1");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    window.localStorage.clear();
  });

  it("固定阵容 + 比分输入 + 提交 payload；成功后展示已登记与评分结果，无「同阵容再记一场」", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: () => Promise.resolve(SCORE_RESPONSE),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <FixtureRecordForm
        projectId={1}
        projectName="周五晚场"
        fixture={fixture()}
        names={NAMES}
      />
    );

    // 固定阵容：展示双方四人，无换人/交换/清空操作
    expect(screen.getByText("第 3 场 · 阵容已固定")).toBeTruthy();
    expect(screen.getByText("甲 / 乙")).toBeTruthy();
    expect(screen.getByText("丙 / 丁")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /交换两边/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /清空阵容/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /选择A队1号球员/ })).toBeNull();

    // 修改比分并提交
    fireEvent.change(screen.getByLabelText("B队比分"), { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: /确认登记比分/ }));

    await screen.findByText("比赛已登记");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/schedule/projects/1/matches/101/score");
    const body = JSON.parse(String(init.body));
    expect(body).toMatchObject({
      scoreA: 21,
      scoreB: 9,
      expectedFixtureRevision: 2,
      actorId: "1",
    });
    expect("playedAt" in body).toBe(false); // 日期口径由服务端按状态决定

    // 成功视图：比分 + 新版评分说明 + legacy 参考 + 返回项目入口
    expect(screen.getByText("21 : 9")).toBeTruthy();
    expect(screen.getByText("新版评分未启用，本次记录未产生新版计分。")).toBeTruthy();
    const backLink = screen.getByRole("link", { name: /返回项目/ });
    expect(backLink.getAttribute("href")).toBe("/schedule/1");
    expect(screen.queryByRole("button", { name: /再记一场|同阵容/ })).toBeNull();
    expect(refreshMock).toHaveBeenCalled();
  });

  it("played_unscored 补录：提示保留原比赛日期", () => {
    render(
      <FixtureRecordForm
        projectId={1}
        projectName="周五晚场"
        fixture={fixture({
          status: "played_unscored",
          playedAt: "2026-10-05",
          revision: 4,
        })}
        names={NAMES}
      />
    );
    const note = screen.getByText(/补录比分将保留原比赛日期/);
    expect(note.textContent).toContain("2026-10-05");
    expect(note.textContent).toContain("标记为已打");
  });

  it("已登记场次打开即为只读，展示比分与录入信息", () => {
    render(
      <FixtureRecordForm
        projectId={1}
        projectName="周五晚场"
        fixture={fixture({
          status: "recorded",
          scoreA: 18,
          scoreB: 21,
          playedAt: "2026-10-02",
          markedBy: "3",
          matchId: 40,
          revision: 5,
        })}
        names={NAMES}
      />
    );
    expect(screen.getByText("已登记")).toBeTruthy();
    expect(screen.getByText("18 : 21")).toBeTruthy();
    expect(screen.getByText(/丙 录入/)).toBeTruthy();
    expect(screen.queryByLabelText("A队比分")).toBeNull();
  });

  it("已被替换场次打开即为只读", () => {
    render(
      <FixtureRecordForm
        projectId={1}
        projectName="周五晚场"
        fixture={fixture({ status: "superseded", revision: 1 })}
        names={NAMES}
      />
    );
    expect(screen.getByText("已被替换")).toBeTruthy();
    expect(screen.getByText(/已被新一批替换/)).toBeTruthy();
    expect(screen.queryByLabelText("A队比分")).toBeNull();
  });

  it("提交遇 409 already_recorded：刷新为只读并显示现状", async () => {
    const recorded = fixture({
      status: "recorded",
      scoreA: 21,
      scoreB: 9,
      playedAt: "2026-10-03",
      markedBy: "2",
      matchId: 41,
      revision: 6,
    });
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: () =>
          Promise.resolve({
            error: "已录入比分",
            code: "already_recorded",
            fixture: recorded,
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () =>
          Promise.resolve({
            project: { fixtures: [recorded], superseded: [] },
            names: NAMES,
          }),
      });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <FixtureRecordForm
        projectId={1}
        projectName="周五晚场"
        fixture={fixture()}
        names={NAMES}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /确认登记比分/ }));

    await screen.findByText("已登记");
    expect(screen.getByText("21 : 9")).toBeTruthy();
    expect(
      screen.getByText(/已由其他人登记比分，已刷新为最新结果/)
    ).toBeTruthy();
    // 只读：比分输入已消失
    expect(screen.queryByLabelText("A队比分")).toBeNull();
    // 先刷新再提示：详情拉取在提示前发生
    expect(fetchMock.mock.calls[1][0]).toBe("/api/schedule/projects/1");
  });

  it("比分相等时禁止提交并提示", () => {
    render(
      <FixtureRecordForm
        projectId={1}
        projectName="周五晚场"
        fixture={fixture()}
        names={NAMES}
      />
    );
    fireEvent.change(screen.getByLabelText("B队比分"), { target: { value: "21" } });
    expect(screen.getByRole("button", { name: /确认登记比分/ })).toHaveProperty(
      "disabled",
      true
    );
    expect(screen.getByText(/比赛不能以平局结束/)).toBeTruthy();
  });
});
