// @vitest-environment jsdom
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { ProjectDetailClient } from "./project-detail";
import type { ProjectDetail, Fixture } from "../types";

vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: vi.fn(), push: vi.fn() }) }));

const ids = Array.from({ length: 8 }, (_, i) => String(i + 1));
const names = Object.fromEntries(ids.map((id) => [id, `球员${id}`]));
function project(complete = false, roundNo = 1): ProjectDetail {
  const fixtures: Fixture[] = Array.from({ length: 6 }, (_, i) => ({
    id: roundNo * 100 + i, projectId: 1, generationId: roundNo, displayNo: (roundNo - 1) * 6 + i + 1,
    pa1: i < 3 ? "1" : "5", pa2: i < 3 ? "2" : "6", pb1: i < 3 ? "3" : "7", pb2: i < 3 ? "4" : "8",
    generatedWinRate: 0.5, status: complete ? "recorded" : "pending", revision: 0,
    playedAt: complete ? "2026-10-19" : null, markedAt: null, markedBy: null,
    matchId: complete ? 1000 + i : null, scoreA: complete ? 21 : null, scoreB: complete ? 15 : null,
  }));
  const standing = (playerId: string, index: number) => ({ playerId, played: complete ? 3 : 0, wins: complete ? 2 : 0, pointsFor: complete ? 60 : 0, pointsAgainst: complete ? 50 : 0, netPoints: complete ? 10 : 0, expectedWins: complete ? 1.5 : 0, overperformance: complete ? 0.5 : 0, drawOrder: index });
  return {
    id: 1, name: "郭老师周三局", model: "legacy", revision: 6, nextFixtureNo: 7,
    createdBy: "1", createdAt: "2026-10-19 12:00:00", updatedAt: "2026-10-19 12:00:00",
    members: ids.map((playerId) => ({ playerId, fairnessBase: 0, playedAtJoin: 0, joinedAt: null, leftAt: null, actualPlayed: complete ? 3 : 0, pendingCount: complete ? 0 : 3 })),
    fixtures, superseded: [], latestGeneration: null,
    guo: { maxRounds: 4, canAdvance: complete && roundNo < 4, nextRoundKind: roundNo === 4 ? null : roundNo === 2 ? "cross" : "promotion", rounds: [{ roundNo, generationId: roundNo, kind: roundNo === 1 ? "initial" : roundNo === 3 ? "cross" : "promotion", groupA: ids.slice(0, 4), groupB: ids.slice(4), standingsA: ids.slice(0, 4).map(standing), standingsB: ids.slice(4).map(standing), complete }] },
  };
}

beforeEach(() => window.localStorage.setItem("badminton:myPlayerId", "1"));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.clearAllMocks(); window.localStorage.clear(); });

describe("郭老师项目详情", () => {
  it("按组显示本轮六场和个人战绩，未录齐比分不能开下一轮", () => {
    render(<ProjectDetailClient initialProject={project()} initialNames={names} directory={[]} />);
    expect(screen.getByText("郭老师分配方法")).toBeTruthy();
    const groupA = screen.getByRole("region", { name: "A 组 · 场地 1" });
    const groupB = screen.getByRole("region", { name: "B 组 · 场地 2" });
    expect(within(groupA).getAllByRole("link", { name: /记分/ })).toHaveLength(3);
    expect(within(groupB).getAllByRole("link", { name: /记分/ })).toHaveLength(3);
    expect(within(groupA).getByRole("columnheader", { name: "净胜分" })).toBeTruthy();
    expect(screen.getByRole("button", { name: /开启第 2 轮/ })).toHaveProperty("disabled", true);
    expect(screen.getByText(/还差 6 场比分/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /续排/ })).toBeNull();
  });

  it("录齐后生成交叉第三轮，网络重试沿用幂等键", async () => {
    const first = project(true, 2);
    const next = project(false, 3);
    const fetchMock = vi.fn().mockRejectedValueOnce(new TypeError("network"))
      .mockResolvedValueOnce({ ok: true, status: 201, json: async () => ({ project: next, names }) });
    vi.stubGlobal("fetch", fetchMock);
    render(<ProjectDetailClient initialProject={first} initialNames={names} directory={[]} />);
    fireEvent.click(screen.getByRole("button", { name: /开启第 3 轮/ }));
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: /开启第 3 轮/ }));
    await screen.findByRole("heading", { name: /第 3 轮/ });
    const bodies = fetchMock.mock.calls.map((call) => JSON.parse(String(call[1].body)));
    expect(fetchMock.mock.calls[0][0]).toBe("/api/schedule/projects/1/generate");
    expect(bodies[0]).toEqual({ actorId: "1", expectedRevision: 6, requestId: expect.any(String) });
    expect(bodies[1].requestId).toBe(bodies[0].requestId);
    expect(screen.getByText(/交叉交流/)).toBeTruthy();
  });

  it("并发冲突先刷新项目，下一轮按刷新后的状态显示", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce({ ok: false, status: 409, json: async () => ({ code: "revision_conflict", error: "项目已更新" }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ project: project(false, 2), names }) });
    vi.stubGlobal("fetch", fetchMock);
    render(<ProjectDetailClient initialProject={project(true)} initialNames={names} directory={[]} />);
    fireEvent.click(screen.getByRole("button", { name: /开启第 2 轮/ }));
    await screen.findByRole("heading", { name: /第 2 轮/ });
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("alert").textContent).toContain("刷新");
  });

  it("三轮结束可自选第四轮，第四轮结束显示完成", () => {
    const view = render(<ProjectDetailClient initialProject={project(true, 3)} initialNames={names} directory={[]} />);
    expect(screen.getByRole("button", { name: /再打第 4 轮/ })).toHaveProperty("disabled", false);
    view.unmount();
    render(<ProjectDetailClient initialProject={project(true, 4)} initialNames={names} directory={[]} />);
    expect(screen.getByText("四轮已完成，今天辛苦了！")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /第 5 轮/ })).toBeNull();
  });
});
