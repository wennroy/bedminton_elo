// @vitest-environment jsdom
/**
 * /schedule/new 新建配对项目表单：
 * <4 人禁提交；「减少已打对阵重复」默认勾选；提交带 UUID requestId；
 * 201 跳详情；409 rating_unavailable 提示；网络失败可安全重试（同 requestId）；
 * 目录外新球员添加后保持身份不变并留在表单；旧 localStorage 草稿仅作提示与勾选参考。
 */
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, refresh: vi.fn() }),
}));

import { NewProjectForm } from "./new-project-form";

const PLAYERS = [
  { id: 1, name: "甲" },
  { id: 2, name: "乙" },
  { id: 3, name: "丙" },
  { id: 4, name: "丁" },
  { id: 5, name: "戊" },
];

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

function selectPlayers(names: string[]) {
  for (const name of names) {
    fireEvent.click(screen.getByRole("button", { name }));
  }
}

function createdResponse(id: number) {
  return {
    ok: true,
    status: 201,
    json: () => Promise.resolve({ project: { id }, replayed: false }),
  };
}

describe("NewProjectForm（新建配对项目）", () => {
  beforeEach(() => {
    window.localStorage.setItem("badminton:myPlayerId", "1");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
    window.localStorage.clear();
  });

  it("少于 4 人时禁用提交，checkbox 默认勾选", () => {
    render(<NewProjectForm players={PLAYERS} />);
    expect(screen.getByText(/已选 0 人（至少 4 人）/)).toBeTruthy();
    expect(screen.getByRole("button", { name: /创建配对项目/ })).toHaveProperty(
      "disabled",
      true
    );
    const checkbox = screen.getByRole("checkbox", {
      name: /减少已打对阵重复/,
    }) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
  });

  it("提交生成 UUID requestId，201 后跳转详情，checkbox 状态随行", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(createdResponse(7));
    vi.stubGlobal("fetch", fetchMock);
    render(<NewProjectForm players={PLAYERS} />);

    selectPlayers(["甲", "乙", "丙", "丁"]);
    fireEvent.click(screen.getByRole("checkbox", { name: /减少已打对阵重复/ }));
    fireEvent.click(screen.getByRole("button", { name: /创建配对项目/ }));

    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/schedule/7"));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/schedule/projects");
    const body = JSON.parse(String(init.body));
    expect(body.playerIds).toEqual(["1", "2", "3", "4"]);
    expect(body.matches).toBe(4);
    expect(body.avoidPlayedRematches).toBe(false);
    expect(body.actorId).toBe("1");
    expect(body.requestId).toMatch(UUID_RE);
    expect("seed" in body).toBe(false);
    expect("name" in body).toBe(false);
  });

  it("409 rating_unavailable 显示评分不可用提示", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: () =>
        Promise.resolve({
          error: "rating service unavailable: x",
          code: "rating_unavailable",
          state: "unavailable",
          reason: "boom",
        }),
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<NewProjectForm players={PLAYERS} />);
    selectPlayers(["甲", "乙", "丙", "丁"]);
    fireEvent.click(screen.getByRole("button", { name: /创建配对项目/ }));

    await screen.findByText(/评分服务暂不可用/);
    expect(pushMock).not.toHaveBeenCalled();
  });

  it("网络失败提示可安全重试，重试沿用同一 requestId", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("network down"))
      .mockResolvedValueOnce(createdResponse(8));
    vi.stubGlobal("fetch", fetchMock);
    render(<NewProjectForm players={PLAYERS} />);
    selectPlayers(["甲", "乙", "丙", "丁"]);

    fireEvent.click(screen.getByRole("button", { name: /创建配对项目/ }));
    await screen.findByText(/不会重复建项目/);

    fireEvent.click(screen.getByRole("button", { name: /创建配对项目/ }));
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/schedule/8"));

    const first = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    const second = JSON.parse(String(fetchMock.mock.calls[1][1].body));
    expect(second.requestId).toBe(first.requestId);
  });

  it("添加目录外新球员：保持操作者身份不变、留在表单并自动勾选", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: () => Promise.resolve({ id: 6, name: "新来的" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<NewProjectForm players={PLAYERS} />);

    fireEvent.click(screen.getByRole("button", { name: "新球员" }));
    fireEvent.change(screen.getByLabelText("新球员名字"), {
      target: { value: "新来的" },
    });
    fireEvent.click(screen.getByRole("button", { name: "确认添加" }));

    await screen.findByRole("button", { name: "新来的" });
    // 身份不被覆盖，仍停留在表单（没有跳转）
    expect(window.localStorage.getItem("badminton:myPlayerId")).toBe("1");
    expect(pushMock).not.toHaveBeenCalled();
    // 新球员已勾选：与另外三人一起即可提交
    selectPlayers(["甲", "乙", "丙"]);
    expect(screen.getByText(/已选 4 人/)).toBeTruthy();
  });

  it("旧 localStorage 草稿：按当时名单勾选并提示不上传", () => {
    window.localStorage.setItem(
      "badminton:schedule:1",
      JSON.stringify({
        playerIds: [1, 2, 3, 4],
        matches: 6,
        seed: 42,
        lambda: 0.5,
        result: {
          schedule: [
            { a1: "1", a2: "2", b1: "3", b2: "4", winRate: 0.5 },
          ],
          metrics: {
            alphaVar: 0.5,
            bestLoss: 0.1,
            meanCloseness: 0.9,
            maxCloseness: 0.95,
            entropy: 2.1,
          },
          names: { "1": "甲", "2": "乙", "3": "丙", "4": "丁" },
        },
        savedAt: "2026-09-01T13:00:00.000Z",
      })
    );
    render(<NewProjectForm players={PLAYERS} />);
    expect(screen.getByText(/已按当时的名单勾选/)).toBeTruthy();
    expect(screen.getByText(/已选 4 人/)).toBeTruthy();
    const matchesInput = screen.getByLabelText("场次数") as HTMLInputElement;
    expect(matchesInput.value).toBe("6");
  });

  it("郭老师模式只允许恰好 8 人，轮次取代自由场次数", () => {
    const players = Array.from({ length: 9 }, (_, i) => ({ id: i + 1, name: `球员${i + 1}` }));
    render(<NewProjectForm players={players} />);
    fireEvent.click(screen.getByRole("radio", { name: /郭老师分配方法/ }));
    selectPlayers(players.slice(0, 7).map((p) => p.name));
    const submit = screen.getByRole("button", { name: /创建配对项目/ });
    expect(submit).toHaveProperty("disabled", true);
    expect(screen.queryByLabelText("场次数")).toBeNull();
    expect(screen.getByText(/每轮 6 场/)).toBeTruthy();
    selectPlayers([players[7].name]);
    expect(submit).toHaveProperty("disabled", false);
    selectPlayers([players[8].name]);
    expect(submit).toHaveProperty("disabled", true);
  });

  it("郭老师模式提交模式和 8 人名单，切回普通配对保留原场次", async () => {
    const players = Array.from({ length: 8 }, (_, i) => ({ id: i + 1, name: `球员${i + 1}` }));
    const fetchMock = vi.fn().mockResolvedValue(createdResponse(9));
    vi.stubGlobal("fetch", fetchMock);
    render(<NewProjectForm players={players} />);
    fireEvent.change(screen.getByLabelText("场次数"), { target: { value: "9" } });
    fireEvent.click(screen.getByRole("radio", { name: /郭老师分配方法/ }));
    selectPlayers(players.map((p) => p.name));
    fireEvent.click(screen.getByRole("radio", { name: /普通配对/ }));
    expect(screen.getByLabelText("场次数")).toHaveProperty("value", "9");
    fireEvent.click(screen.getByRole("radio", { name: /郭老师分配方法/ }));
    fireEvent.click(screen.getByRole("button", { name: /创建配对项目/ }));
    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/schedule/9"));
    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body.mode).toBe("guo");
    expect(body.playerIds).toHaveLength(8);
    expect(body.matches).toBe(6);
  });
});
