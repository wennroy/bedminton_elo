// @vitest-environment jsdom
/**
 * record-form 手动记分回归：「比分输入 / 评分结果显示」提取为共用组件后，
 * 手动记分（/record）的行为必须完全不变——
 * 选满 4 人才可提交、比分输入与加减分、确认弹层、提交 payload、
 * 成功弹层（新版评分三态 + legacy ELO 参考）、再记一场保留阵容重置比分。
 */
import * as React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";

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

const pushMock = vi.fn();
const refreshMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, refresh: refreshMock }),
}));

// Radix Dialog 在 jsdom 下依赖 pointer capture 等浏览器 API；
// 用受控的轻量替身聚焦表单逻辑本身（打开即渲染内容）。
vi.mock("@/components/ui/dialog", async () => {
  const React = await import("react");
  const passthrough = (tag: string) =>
    function DialogStub({
      children,
      onOpenAutoFocus: _ignored1,
      showCloseButton: _ignored2,
      ...rest
    }: {
      children?: React.ReactNode;
      onOpenAutoFocus?: unknown;
      showCloseButton?: unknown;
    }) {
      return React.createElement(tag, rest, children);
    };
  return {
    Dialog: ({
      open,
      children,
    }: {
      open: boolean;
      children?: React.ReactNode;
    }) => (open ? React.createElement("div", { "data-testid": "dialog" }, children) : null),
    DialogTrigger: ({ children }: { children?: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    DialogContent: passthrough("div"),
    DialogHeader: passthrough("div"),
    DialogTitle: passthrough("div"),
    DialogDescription: passthrough("p"),
  };
});

import { RecordForm } from "./record-form";

const PLAYERS = [
  { id: 1, name: "甲" },
  { id: 2, name: "乙" },
  { id: 3, name: "丙" },
  { id: 4, name: "丁" },
  { id: 5, name: "戊" },
];

const RECORD_RESPONSE = {
  id: 9,
  before: [
    { id: 1, name: "甲", elo: 1000 },
    { id: 2, name: "乙", elo: 1000 },
    { id: 3, name: "丙", elo: 1000 },
    { id: 4, name: "丁", elo: 1000 },
  ],
  after: [
    { id: 1, name: "甲", elo: 1016 },
    { id: 2, name: "乙", elo: 1016 },
    { id: 3, name: "丙", elo: 984 },
    { id: 4, name: "丁", elo: 984 },
  ],
  legacyRatingsAvailable: true,
  rating: {
    state: "not_effective",
    model: "glicko2",
    reason: "rating config not initialized",
  },
};

function mockFetchOk() {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 201,
    json: () => Promise.resolve(RECORD_RESPONSE),
  });
}

async function fillSlot(slotLabel: string, playerName: string) {
  fireEvent.click(screen.getByRole("button", { name: new RegExp(slotLabel) }));
  const dialog = screen.getByTestId("dialog");
  // 头像与姓名都含首字文本，改用 aria-label 精确定位候选按钮。
  fireEvent.click(
    within(dialog).getByRole("button", { name: new RegExp(`^${playerName}`) })
  );
}

describe("RecordForm（手动记分回归）", () => {
  beforeEach(() => {
    window.localStorage.setItem("badminton:myPlayerId", "1");
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    pushMock.mockReset();
    refreshMock.mockReset();
  });

  it("未选满 4 人时禁用提交并给出提示", () => {
    render(<RecordForm players={PLAYERS} matches={[]} />);
    expect(screen.getByText(/已选 0 \/ 4 人/)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: /确认比分/ })
    ).toHaveProperty("disabled", true);
  });

  it("加减分与直接输入都作用于比分", () => {
    render(<RecordForm players={PLAYERS} matches={[]} />);
    const scoreA = screen.getByLabelText("A队比分") as HTMLInputElement;
    expect(scoreA.value).toBe("21");

    fireEvent.click(screen.getByRole("button", { name: "A队加一分" }));
    expect(scoreA.value).toBe("22");
    fireEvent.click(screen.getByRole("button", { name: "A队减一分" }));
    expect(scoreA.value).toBe("21");

    const scoreB = screen.getByLabelText("B队比分") as HTMLInputElement;
    fireEvent.change(scoreB, { target: { value: "18" } });
    expect(scoreB.value).toBe("18");
  });

  it("完整流程：选阵容 → 确认 → 提交 payload 正确 → 成功弹层展示三态结果与 legacy 参考", async () => {
    const fetchMock = mockFetchOk();
    vi.stubGlobal("fetch", fetchMock);
    render(<RecordForm players={PLAYERS} matches={[]} />);

    await fillSlot("选择A队1号球员", "甲");
    await fillSlot("选择A队2号球员", "乙");
    await fillSlot("选择B队1号球员", "丙");
    await fillSlot("选择B队2号球员", "丁");

    expect(screen.getByText("确认后展示四位球员的积分变化")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /确认比分/ }));

    // 确认弹层：阵容与比分复核
    const confirmDialog = screen.getByTestId("dialog");
    expect(within(confirmDialog).getByText("确认这场比赛")).toBeTruthy();
    fireEvent.click(within(confirmDialog).getByRole("button", { name: "确认记分" }));

    await screen.findByText("比赛已记录");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/matches");
    const body = JSON.parse(String(init.body));
    expect(body).toMatchObject({
      pa1: 1,
      pa2: 2,
      pb1: 3,
      pb2: 4,
      scoreA: 21,
      scoreB: 0,
      enteredBy: 1,
    });
    expect(body.playedAt).toMatch(/^\d{4}-\d{2}-\d{2}$/);

    // 成功弹层：not_effective 说明 + legacy ELO 参考卡
    expect(screen.getByText("新版评分未启用，本次记录未产生新版计分。")).toBeTruthy();
    expect(screen.getByText("ELO")).toBeTruthy();
    // 甲/乙 +16，丙/丁 -16（round 后整差）
    expect(screen.getAllByText("+16")).toHaveLength(2);
    expect(screen.getAllByText("-16")).toHaveLength(2);
    expect(refreshMock).toHaveBeenCalled();
  });

  it("再记一场：保留阵容、比分重置", async () => {
    const fetchMock = mockFetchOk();
    vi.stubGlobal("fetch", fetchMock);
    render(<RecordForm players={PLAYERS} matches={[]} />);
    await fillSlot("选择A队1号球员", "甲");
    await fillSlot("选择A队2号球员", "乙");
    await fillSlot("选择B队1号球员", "丙");
    await fillSlot("选择B队2号球员", "丁");
    fireEvent.click(screen.getByRole("button", { name: /确认比分/ }));
    fireEvent.click(
      within(screen.getByTestId("dialog")).getByRole("button", { name: "确认记分" })
    );
    await screen.findByText("比赛已记录");

    fireEvent.click(screen.getByRole("button", { name: /再记一场/ }));
    expect(screen.getByLabelText("A队比分")).toHaveProperty("value", "21");
    expect(screen.getByLabelText("B队比分")).toHaveProperty("value", "0");
    // 阵容保留：再次进入确认弹层时仍是原班人马
    fireEvent.click(screen.getByRole("button", { name: /确认比分/ }));
    expect(screen.getByText("确认这场比赛")).toBeTruthy();
  });
});
