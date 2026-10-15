import { describe, expect, it } from "vitest";
import {
  canonicalMatchKey,
  validateInput,
  ALGORITHM_VERSION,
  OBJECTIVE_VERSION,
  type SchedulingInput,
} from "./types";

describe("canonicalMatchKey", () => {
  it("'AB vs CD'、'BA vs DC'、'CD vs AB' 为同一键", () => {
    const k1 = canonicalMatchKey("A", "B", "C", "D");
    const k2 = canonicalMatchKey("B", "A", "D", "C");
    const k3 = canonicalMatchKey("C", "D", "A", "B");
    expect(k2).toBe(k1);
    expect(k3).toBe(k1);
  });

  it("'AC vs BD' 是另一键", () => {
    const k = canonicalMatchKey("A", "B", "C", "D");
    expect(canonicalMatchKey("A", "C", "B", "D")).not.toBe(k);
  });

  it("队内/队间顺序全部 8 种排列收敛到两个键", () => {
    const perms: Array<[string, string, string, string]> = [
      ["A", "B", "C", "D"],
      ["B", "A", "C", "D"],
      ["A", "B", "D", "C"],
      ["B", "A", "D", "C"],
      ["C", "D", "A", "B"],
      ["D", "C", "A", "B"],
      ["C", "D", "B", "A"],
      ["D", "C", "B", "A"],
    ];
    const keys = new Set(perms.map((p) => canonicalMatchKey(...p)));
    expect(keys.size).toBe(1);
  });

  it("键形状为 JSON 字符串化的两队数组", () => {
    expect(canonicalMatchKey("A", "B", "C", "D")).toBe(
      JSON.stringify([
        ["A", "B"],
        ["C", "D"],
      ])
    );
  });
});

function makeInput(overrides: Partial<SchedulingInput> = {}): SchedulingInput {
  const playerIds = ["a", "b", "c", "d"];
  return {
    playerIds,
    matches: 2,
    effectiveAppearances: { a: 0, b: 0, c: 0, d: 0 },
    playedMatchCounts: {},
    avoidPlayedRematches: true,
    seed: 0,
    algorithmVersion: ALGORITHM_VERSION,
    objectiveVersion: OBJECTIVE_VERSION,
    winProbability: () => 0.5,
    ...overrides,
  };
}

describe("validateInput", () => {
  it("合法输入通过（边界值也接受）", () => {
    expect(() => validateInput(makeInput())).not.toThrow();
    expect(() =>
      validateInput(makeInput({ matches: 1, seed: 0 }))
    ).not.toThrow();
    expect(() =>
      validateInput(makeInput({ matches: 99, seed: 0xffffffff }))
    ).not.toThrow();
    const many = Array.from({ length: 64 }, (_, i) => `p${i}`);
    const appearances: Record<string, number> = {};
    for (const id of many) appearances[id] = 0;
    expect(() =>
      validateInput(makeInput({ playerIds: many, effectiveAppearances: appearances }))
    ).not.toThrow();
  });

  it("人数越界抛错", () => {
    expect(() =>
      validateInput(makeInput({ playerIds: ["a", "b", "c"] }))
    ).toThrow(/4.*64/);
    const tooMany = Array.from({ length: 65 }, (_, i) => `p${i}`);
    expect(() => validateInput(makeInput({ playerIds: tooMany }))).toThrow();
  });

  it("playerIds 重复抛错（唯一性校验，不负责排序）", () => {
    expect(() =>
      validateInput(makeInput({ playerIds: ["a", "b", "c", "a"] }))
    ).toThrow(/唯一/);
    // 未排序但唯一：通过（排序由调用方保证的约定）
    expect(() =>
      validateInput(
        makeInput({
          playerIds: ["d", "c", "b", "a"],
          effectiveAppearances: { a: 0, b: 0, c: 0, d: 0 },
        })
      )
    ).not.toThrow();
  });

  it("matches 非安全整数或越界抛错", () => {
    expect(() => validateInput(makeInput({ matches: 0 }))).toThrow();
    expect(() => validateInput(makeInput({ matches: 100 }))).toThrow();
    expect(() => validateInput(makeInput({ matches: 1.5 }))).toThrow();
    expect(() =>
      validateInput(makeInput({ matches: Number.MAX_SAFE_INTEGER + 1 }))
    ).toThrow();
  });

  it("seed 越界或非整数抛错", () => {
    expect(() => validateInput(makeInput({ seed: -1 }))).toThrow();
    expect(() => validateInput(makeInput({ seed: 0x100000000 }))).toThrow();
    expect(() => validateInput(makeInput({ seed: 1.5 }))).toThrow();
  });

  it("effectiveAppearances 缺键/负数/非整数/名单外键抛错", () => {
    expect(() =>
      validateInput(
        makeInput({ effectiveAppearances: { a: 0, b: 0, c: 0 } as never })
      )
    ).toThrow(/缺少/);
    expect(() =>
      validateInput(
        makeInput({
          effectiveAppearances: { a: 0, b: 0, c: 0, d: -1 },
        })
      )
    ).toThrow(/非负/);
    expect(() =>
      validateInput(
        makeInput({ effectiveAppearances: { a: 0, b: 0, c: 0, d: 1.5 } })
      )
    ).toThrow(/非负/);
    expect(() =>
      validateInput(
        makeInput({
          effectiveAppearances: { a: 0, b: 0, c: 0, d: 0, e: 0 },
        })
      )
    ).toThrow(/名单外/);
  });

  it("playedMatchCounts 值非负整数校验", () => {
    expect(() =>
      validateInput(makeInput({ playedMatchCounts: { "[[\"A\",\"B\"],[\"C\",\"D\"]]": 2 } }))
    ).not.toThrow();
    expect(() =>
      validateInput(makeInput({ playedMatchCounts: { k: -1 } }))
    ).toThrow();
    expect(() =>
      validateInput(makeInput({ playedMatchCounts: { k: 1.5 } }))
    ).toThrow();
  });

  it("版本/回调/avoidPlayedRematches 类型校验", () => {
    expect(() =>
      validateInput(makeInput({ algorithmVersion: "other" as never }))
    ).toThrow(/algorithmVersion/);
    expect(() =>
      validateInput(makeInput({ objectiveVersion: "other" as never }))
    ).toThrow(/objectiveVersion/);
    expect(() =>
      validateInput(makeInput({ winProbability: undefined as never }))
    ).toThrow(/winProbability/);
    expect(() =>
      validateInput(makeInput({ avoidPlayedRematches: "yes" as never }))
    ).toThrow(/avoidPlayedRematches/);
  });
});
