import { describe, expect, it } from "vitest";
import { optimizeSchedule } from "./scheduler";
import { predictTeamOutcomeWin } from "./trueskill";
import golden from "../../test/golden/scheduler.json";

describe("scheduler", () => {
  const input = golden.input as {
    playerIds: string[];
    matches: number;
    players: Array<{ mu: number; sigma: number }>;
    seed: number;
    lambda: number;
  };

  it("produces the expected number of matches", () => {
    const result = optimizeSchedule(input);
    expect(result.schedule).toHaveLength(input.matches);
  });

  it("has four distinct players in every match", () => {
    const result = optimizeSchedule(input);
    for (const match of result.schedule) {
      const ids = [match.a1, match.a2, match.b1, match.b2];
      expect(new Set(ids).size).toBe(4);
    }
  });

  it("gives every selected player at least one appearance when feasible", () => {
    const result = optimizeSchedule(input);
    const seen = new Set<string>();
    for (const match of result.schedule) {
      seen.add(match.a1);
      seen.add(match.a2);
      seen.add(match.b1);
      seen.add(match.b2);
    }
    for (const id of input.playerIds) {
      expect(seen.has(id)).toBe(true);
    }
  });

  it("only uses players from the input list", () => {
    const result = optimizeSchedule(input);
    for (const match of result.schedule) {
      expect(input.playerIds).toContain(match.a1);
      expect(input.playerIds).toContain(match.a2);
      expect(input.playerIds).toContain(match.b1);
      expect(input.playerIds).toContain(match.b2);
    }
  });

  it("is deterministic for the same seed", () => {
    const r1 = optimizeSchedule(input);
    const r2 = optimizeSchedule(input);
    expect(r1.schedule).toEqual(r2.schedule);
    expect(r1.metrics).toEqual(r2.metrics);
  });

  it("exposes optimization metrics", () => {
    const result = optimizeSchedule(input);
    expect(result.metrics.alphaVar).toBeGreaterThanOrEqual(0);
    expect(result.metrics.bestLoss).toBeGreaterThanOrEqual(0);
    expect(result.metrics.meanCloseness).toBeGreaterThanOrEqual(0);
    expect(result.metrics.maxCloseness).toBeGreaterThanOrEqual(0);
    expect(result.metrics.entropy).toBeGreaterThanOrEqual(0);
  });

  it("can run with different seeds", () => {
    const r1 = optimizeSchedule({ ...input, seed: 1 });
    const r2 = optimizeSchedule({ ...input, seed: 2 });
    expect(r1.schedule).toHaveLength(input.matches);
    expect(r2.schedule).toHaveLength(input.matches);
  });

  it("uses the injected winProbability callback for closeness", () => {
    const seen: string[][] = [];
    const result = optimizeSchedule({
      ...input,
      winProbability: (match) => {
        seen.push([match.a1, match.a2, match.b1, match.b2]);
        return 0.5;
      },
    });
    // 常数 0.5 回调：所有 closeness 必须为 0；回调在每次 loss 评估都被调用
    // （含模拟退火邻域），每次都收到一场四方阵容。
    expect(result.metrics.meanCloseness).toBe(0);
    expect(result.metrics.maxCloseness).toBe(0);
    expect(seen.length).toBeGreaterThanOrEqual(input.matches);
    for (const ids of seen) {
      expect(new Set(ids).size).toBe(4);
    }
  });

  it("explicit TrueSkill-equivalent callback reproduces the default bit-for-bit", () => {
    const explicit = optimizeSchedule({
      ...input,
      winProbability: (match) =>
        predictTeamOutcomeWin(
          [match.a1, match.a2].map(
            (id) => input.players[input.playerIds.indexOf(id)]
          ),
          [match.b1, match.b2].map(
            (id) => input.players[input.playerIds.indexOf(id)]
          )
        ),
    });
    const def = optimizeSchedule(input);
    expect(explicit.schedule).toEqual(def.schedule);
    expect(explicit.metrics).toEqual(def.metrics);
  });

  it("is deterministic for the same seed with an injected callback", () => {
    const callback = (match: { a1: string; a2: string; b1: string; b2: string }) =>
      (Number(match.a1) + Number(match.a2)) / 100;
    const r1 = optimizeSchedule({ ...input, winProbability: callback });
    const r2 = optimizeSchedule({ ...input, winProbability: callback });
    expect(r1.schedule).toEqual(r2.schedule);
    expect(r1.metrics).toEqual(r2.metrics);
  });
});
