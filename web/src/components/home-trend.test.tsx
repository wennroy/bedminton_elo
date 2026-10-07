// @vitest-environment jsdom
/**
 * 「趋势图删除日期按钮行与查看提示」的组件级回归(新版 glicko2 + Legacy 两条渲染路径)。
 *
 * 验收口径来自需求本身,而不是照着实现写断言:
 *  - 两条路径都不再渲染 role="group" aria-label="选择查看日期" 的日期按钮行;
 *  - Legacy 不再出现「点击日期查看当日排名与积分」「颜色与球员固定对应」;
 *    新版不再出现「点击日期查看该时点的排名与评分」;
 *  - 保留:读数栏 aside(默认最新时点)、成员筛选、周期/季度切换、
 *    新版「正式结算 / 本周预估」图例(右端对齐)、compact 变体「展开大图」链接;
 *  - 边界:清空成员空态无按钮行残留;无数据空态无按钮行残留。
 *
 * 环境说明:recharts 在 jsdom 以 0 尺寸挂载(线不渲染,组件正常挂载);
 * 桌面 hover / 手机点按的「读数栏时点切换」由浏览器旅程验收,不在此覆盖。
 * next/link 以普通 <a> 替身渲染(无 App Router 上下文);真实导航由浏览器旅程覆盖。
 */
import * as React from "react";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { HomeTrend, TrendXAxisTick } from "./home-trend";
import { createRatingConfig } from "@/lib/ratings/config";
import { replayRatings } from "@/lib/ratings/replay";
import { projectRatingView } from "@/lib/ratings/projections";
import { replayMatch } from "../../test/fixtures/ratings-scenarios";
import type { EloHistoryPoint } from "@/lib/stats";

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

beforeAll(() => {
  // recharts ResponsiveContainer 需要 ResizeObserver,jsdom 未实现。
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  if (!("ResizeObserver" in globalThis)) {
    (globalThis as Record<string, unknown>).ResizeObserver = ResizeObserverStub;
  }
});

afterEach(() => cleanup());

const DIRECTORY = [
  { id: 1, name: "球员一" },
  { id: 2, name: "球员二" },
  { id: 3, name: "球员三" },
  { id: 4, name: "球员四" },
];

/**
 * 跨两个带标签季度的场景:firstSeasonStart = 2026-07-01(Q3 季首),
 * 09-28 比赛落在 Q3、10-02/10-05 落在 Q4,asOf 在 Q4 ——
 * 季度选项为 当前季度 / 2026年Q3 / 2026年Q4 / 全部历史。
 */
const twoSeasonConfig = createRatingConfig({ firstSeasonStart: "2026-07-01" });

function glicko2Props() {
  const replay = replayRatings(
    [
      replayMatch(1, "2026-09-28"),
      replayMatch(2, "2026-10-02", 17, 21),
      replayMatch(3, "2026-10-05"),
    ],
    [1, 2, 3, 4],
    { config: twoSeasonConfig, asOf: "2026-10-06T10:00:00+08:00" }
  );
  return {
    model: "glicko2" as const,
    view: projectRatingView(replay, DIRECTORY),
    currentSegmentId: replay.currentSegment.id,
    currentSeason: replay.currentSegment.seasonId,
    now: replay.asOf,
    variant: "full" as const,
    ratingQuery: "?rating=glicko2",
  };
}

/** 无任何有效比赛的视图:当前季度无数据,走 rows.length === 0 空态分支。 */
function glicko2EmptyProps() {
  const replay = replayRatings([], [1, 2, 3, 4], {
    config: twoSeasonConfig,
    asOf: "2026-10-06T10:00:00+08:00",
  });
  return {
    model: "glicko2" as const,
    view: projectRatingView(replay, DIRECTORY),
    currentSegmentId: replay.currentSegment.id,
    currentSeason: replay.currentSegment.seasonId,
    now: replay.asOf,
    variant: "full" as const,
    ratingQuery: "?rating=glicko2",
  };
}

function legacyHistory(): EloHistoryPoint[] {
  const dates = ["2026-09-28", "2026-10-02", "2026-10-05"];
  const elos: Record<number, number[]> = {
    1: [1000, 1012, 1024],
    2: [1000, 990, 1001],
    3: [1000, 998, 975],
    4: [1000, 1004, 1010],
  };
  const history: EloHistoryPoint[] = [];
  for (const p of DIRECTORY) {
    dates.forEach((date, i) => {
      history.push({
        date,
        playerId: String(p.id),
        playerName: p.name,
        elo: elos[p.id][i],
      });
    });
  }
  return history;
}

function legacyProps() {
  return {
    model: "legacy" as const,
    history: legacyHistory(),
    players: DIRECTORY,
    variant: "full" as const,
  };
}

/** 被删除的日期按钮行:role=group「选择查看日期」。其余 Segmented group 必须仍在。 */
function expectDateStripAbsent() {
  expect(
    screen.queryByRole("group", { name: "选择查看日期" })
  ).toBeNull();
}

function readoutLinks(): HTMLElement[] {
  const aside = document.querySelector("aside");
  expect(aside).not.toBeNull();
  return within(aside as HTMLElement).getAllByRole("link");
}

describe("Legacy 趋势(全量页)", () => {
  // Legacy 窗口以真实挂钟为锚(cutoffDate 取 Date.now()):钉住「今天」为
  // 2026-10-07(上海正午),否则 cutoff 随日子推移越过夹具日期(09-28/10-02/10-05),
  // 「近 4 周/近 12 周」断言会无需任何代码改动而误红。
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime("2026-10-07T12:00:00+08:00");
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("删除日期按钮行与底部整行提示,保留读数栏与周期/模式切换", () => {
    render(<HomeTrend {...legacyProps()} />);

    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看当日排名与积分")).toBeNull();
    expect(screen.queryByText("颜色与球员固定对应")).toBeNull();

    // 保留:周期/模式分段控件与读数栏(默认最新时点 = 最后比赛日)。
    expect(
      screen.getByRole("group", { name: "全员趋势周期" })
    ).toBeTruthy();
    expect(
      screen.getByRole("group", { name: "全员趋势类型" })
    ).toBeTruthy();
    const aside = document.querySelector("aside") as HTMLElement;
    expect(within(aside).getByText("2026-10-05")).toBeTruthy();
    expect(within(aside).getByText("ELO")).toBeTruthy();
    expect(readoutLinks()).toHaveLength(4);
    // 最新时点读数:球员一 1024 居首。
    expect(within(aside).getByText("1024")).toBeTruthy();

    // 周期切换(近 4 周 / 近 12 周 / 全部)与排名模式仍可用。
    fireEvent.click(screen.getByRole("button", { name: "全部" }));
    expect(within(aside).getByText("2026-10-05")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "近 4 周" }));
    expect(within(aside).getByText("2026-10-05")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "排名" }));
    expect(within(aside).getByText("名次 / ELO")).toBeTruthy();
    expectDateStripAbsent();
  });

  it("清空成员显示空态且无按钮行残留,全选后恢复读数", () => {
    render(<HomeTrend {...legacyProps()} />);

    fireEvent.click(screen.getByRole("button", { name: "清空" }));
    expect(screen.getByText("选择下方成员，查看 ELO 趋势。")).toBeTruthy();
    expect(screen.getByText("尚未选择成员")).toBeTruthy();
    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看当日排名与积分")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    expect(readoutLinks()).toHaveLength(4);
  });

  it("无比赛数据空态:无按钮行与提示残留", () => {
    render(<HomeTrend {...legacyProps()} history={[]} />);
    expect(
      screen.getByText("还没有比赛数据，记一场后这里会出现趋势。")
    ).toBeTruthy();
    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看当日排名与积分")).toBeNull();
    expect(screen.queryByText("颜色与球员固定对应")).toBeNull();
  });

  it("compact 变体:无按钮行/提示,保留「展开大图」链接", () => {
    render(<HomeTrend {...legacyProps()} variant="compact" />);
    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看当日排名与积分")).toBeNull();
    expect(screen.queryByText("颜色与球员固定对应")).toBeNull();
    const link = screen.getByRole("link", { name: /展开大图/ });
    expect(link.getAttribute("href")).toBe("/trends");
  });
});

describe("新版 glicko2 趋势(全量页)", () => {
  it("删除事件时点按钮行与提示,图例保留且整行右对齐", () => {
    render(<HomeTrend {...glicko2Props()} />);

    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看该时点的排名与评分")).toBeNull();

    // 图例保留:正式结算(实线)/ 本周预估(虚线),且容器右对齐(justify-end)。
    const legendSolid = screen.getByText("正式结算");
    const legendDashed = screen.getByText("本周预估");
    const legendRow = legendDashed.closest("div") as HTMLElement;
    expect(legendRow.className).toContain("justify-end");
    expect(legendRow.contains(legendSolid)).toBe(true);

    // 保留:类型/季度分段控件与读数栏(默认最新时点 = 合成「现在」行 10.06)。
    expect(
      screen.getByRole("group", { name: "全员趋势季度" })
    ).toBeTruthy();
    expect(
      screen.getByRole("group", { name: "全员趋势类型" })
    ).toBeTruthy();
    const aside = document.querySelector("aside") as HTMLElement;
    expect(within(aside).getByText("10.06")).toBeTruthy();
    expect(within(aside).getByText(/评分/)).toBeTruthy();
    expect(readoutLinks()).toHaveLength(4);
  });

  it("季度切换:当前季度 / 历史季度 / 全部历史均正常渲染", () => {
    render(<HomeTrend {...glicko2Props()} />);
    const aside = document.querySelector("aside") as HTMLElement;

    // 全部历史
    fireEvent.click(screen.getByRole("button", { name: "全部历史" }));
    expect(readoutLinks().length).toBeGreaterThan(0);
    expect(screen.getByText("历史回放")).toBeTruthy();

    // 历史具体季度(夹具含 2026-07-01 赛季 = 2026年Q3):不追加「现在」行
    fireEvent.click(screen.getByRole("button", { name: "2026年Q3" }));
    expect(readoutLinks().length).toBeGreaterThan(0);
    expect(within(aside).queryByText("10.06")).toBeNull();

    // 回到当前季度:「现在」行恢复
    fireEvent.click(screen.getByRole("button", { name: "当前季度" }));
    expect(within(aside).getByText("10.06")).toBeTruthy();
    expectDateStripAbsent();
  });

  it("排名模式切换正常", () => {
    render(<HomeTrend {...glicko2Props()} />);
    const aside = document.querySelector("aside") as HTMLElement;
    fireEvent.click(screen.getByRole("button", { name: "排名" }));
    expect(within(aside).getByText(/名次 \/ 评分/)).toBeTruthy();
    expect(readoutLinks()).toHaveLength(4);
    expectDateStripAbsent();
  });

  it("清空成员显示空态且无按钮行残留,全选后恢复读数", () => {
    render(<HomeTrend {...glicko2Props()} />);

    fireEvent.click(screen.getByRole("button", { name: "清空" }));
    expect(screen.getByText("选择下方成员，查看评分趋势。")).toBeTruthy();
    expect(screen.getByText("尚未选择成员")).toBeTruthy();
    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看该时点的排名与评分")).toBeNull();
    // 清空时图例行仍保留(与交互前一致)
    expect(screen.getByText("正式结算")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "全选" }));
    expect(readoutLinks()).toHaveLength(4);
  });

  it("当前季度无数据空态:无按钮行与提示残留,图例行仍在", () => {
    render(<HomeTrend {...glicko2EmptyProps()} />);
    expect(
      screen.getByText("该季度还没有比赛数据，记一场后这里会出现趋势。")
    ).toBeTruthy();
    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看该时点的排名与评分")).toBeNull();
    expect(screen.getByText("正式结算")).toBeTruthy();
  });

  it("compact 变体:无按钮行/提示,保留「展开大图」链接与评分模式", () => {
    render(<HomeTrend {...glicko2Props()} variant="compact" />);
    expectDateStripAbsent();
    expect(screen.queryByText("点击日期查看该时点的排名与评分")).toBeNull();
    const link = screen.getByRole("link", { name: /展开大图/ });
    expect(link.getAttribute("href")).toBe("/trends?rating=glicko2");
  });
});

describe("X 轴刻度(跨年第二行年份)", () => {
  const renderTick = (
    value: string,
    boundaries: ReadonlySet<string>,
    dateOf: (v: string) => string | null = (v) => v
  ) =>
    render(
      <svg>
        <TrendXAxisTick
          x={50}
          y={270}
          payload={{ value }}
          boundaries={boundaries}
          dateOf={dateOf}
        />
      </svg>
    );

  it("跨年边界刻度渲染两行:MM.DD + 年份", () => {
    const { container } = renderTick("2026-01-04", new Set(["2026-01-04"]));
    const tspans = container.querySelectorAll("tspan");
    expect(tspans).toHaveLength(2);
    expect(tspans[0].textContent).toBe("01.04");
    expect(tspans[1].textContent).toBe("2026");
  });

  it("非边界刻度只渲染 MM.DD 一行", () => {
    const { container } = renderTick("2026-01-11", new Set(["2026-01-04"]));
    const tspans = container.querySelectorAll("tspan");
    expect(tspans).toHaveLength(1);
    expect(tspans[0].textContent).toBe("01.11");
  });

  it("刻度值无法换算日期时渲染空(glicko2 未知 key)", () => {
    const { container } = renderTick("missing-key", new Set(), () => null);
    expect(container.querySelector("text")).toBeNull();
  });
});
