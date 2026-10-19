import { describe, expect, it } from "vitest";
import { createGuoDraw, createGuoGroups, createGuoSchedule, rankGuoGroup, nextGuoGroups } from "./guo";

const ids = ["1", "2", "3", "4", "5", "6", "7", "8"];
const draw = Object.fromEntries(ids.map((id, i) => [id, i]));
const blank = (playerId: string, wins = 0, netPoints = 0, overperformance = 0) => ({ playerId, played: 3, wins, netPoints, pointsFor: 63, pointsAgainst: 63 - netPoints, expectedWins: wins - overperformance, overperformance, drawOrder: draw[playerId] });

describe("郭老师分配纯引擎", () => {
  it("初始按所选实力分排序，同分使用可复现的持久抽签", () => {
    const d = createGuoDraw(ids, 42);
    expect(createGuoDraw([...ids].reverse(), 42)).toEqual(d);
    expect(new Set(Object.values(d)).size).toBe(8);
    const groups = createGuoGroups(ids, Object.fromEntries(ids.map((id) => [id, Number(id)])), d);
    expect(groups.groupA).toEqual(["8", "7", "6", "5"]);
    const tied = createGuoGroups(ids, Object.fromEntries(ids.map((id) => [id, 1000])), d);
    expect([...tied.groupA, ...tied.groupB]).toEqual([...ids].sort((a, b) => d[a] - d[b]));
  });

  it("仅支持八名唯一球员，每组四人三种搭档恰好各打一场", () => {
    expect(() => createGuoGroups(ids.slice(1), {}, draw)).toThrow();
    expect(() => createGuoSchedule(ids.slice(0, 4), ["5", "6", "7", "7"])).toThrow();
    const schedule = createGuoSchedule(ids.slice(0, 4), ids.slice(4));
    expect(schedule).toHaveLength(6);
    for (const group of [ids.slice(0, 4), ids.slice(4)]) {
      const pairs = schedule.filter((m) => group.includes(m.a1)).flatMap((m) => [[m.a1, m.a2].sort().join(), [m.b1, m.b2].sort().join()]);
      expect(new Set(pairs).size).toBe(6);
      for (const id of group) expect(schedule.filter((m) => Object.values(m).includes(id))).toHaveLength(3);
    }
  });

  it("完整并列依次比较胜场、净胜分、超预期分与初始签位", () => {
    const ordered = rankGuoGroup([blank("1", 1, 100), blank("2", 2, -5), blank("3", 1, 100, 0.3), blank("4", 1, 100, 0.3)]);
    expect(ordered.map((r) => r.playerId)).toEqual(["2", "3", "4", "1"]);
  });

  it("第二/四轮正常升降，第三轮 A 前二与 B 后二同组", () => {
    const a = ids.slice(0, 4).map((id) => blank(id));
    const b = ids.slice(4).map((id) => blank(id));
    expect(nextGuoGroups(a, b, "promotion")).toEqual({ groupA: ["1", "2", "5", "6"], groupB: ["3", "4", "7", "8"] });
    expect(nextGuoGroups(a, b, "cross")).toEqual({ groupA: ["1", "2", "7", "8"], groupB: ["3", "4", "5", "6"] });
  });
});
