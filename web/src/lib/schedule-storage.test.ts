import { describe, it, expect } from "vitest";
import {
  parseStoredSchedule,
  type StoredSchedule,
} from "./schedule-storage";

const VALID: StoredSchedule = {
  playerIds: [1, 2, 3, 4],
  matches: 4,
  seed: 42,
  lambda: 0.5,
  result: {
    schedule: [
      { a1: "1", a2: "2", b1: "3", b2: "4", winRate: 0.52 },
    ],
    metrics: {
      alphaVar: 0.5,
      bestLoss: 0.1,
      meanCloseness: 0.9,
      maxCloseness: 0.95,
      entropy: 2.1,
    },
    names: { "1": "Alice", "2": "Bob", "3": "Carol", "4": "Dave" },
  },
  savedAt: "2026-09-01T13:00:00.000Z",
};

describe("parseStoredSchedule", () => {
  it("round-trips a valid stored schedule", () => {
    // 旧格式读出时显式补 model: null 标识旧快照（不补造其他元信息）。
    expect(parseStoredSchedule(JSON.stringify(VALID))).toEqual({
      ...VALID,
      model: null,
      result: { ...VALID.result, model: null },
    });
  });

  it("returns null for invalid JSON", () => {
    expect(parseStoredSchedule("not json{")).toBeNull();
    expect(parseStoredSchedule("")).toBeNull();
  });

  it("returns null for wrong shapes", () => {
    expect(parseStoredSchedule("null")).toBeNull();
    expect(parseStoredSchedule("[]")).toBeNull();
    expect(parseStoredSchedule(JSON.stringify({ ...VALID, playerIds: ["1"] }))).toBeNull();
    expect(parseStoredSchedule(JSON.stringify({ ...VALID, seed: "42" }))).toBeNull();
    expect(parseStoredSchedule(JSON.stringify({ ...VALID, result: null }))).toBeNull();
  });

  it("rejects schedule entries with missing fields", () => {
    const bad = structuredClone(VALID);
    // @ts-expect-error 故意造坏数据
    delete bad.result.schedule[0].winRate;
    expect(parseStoredSchedule(JSON.stringify(bad))).toBeNull();

    const bad2 = structuredClone(VALID);
    // @ts-expect-error 故意造坏数据
    bad2.result.schedule[0].a1 = 1;
    expect(parseStoredSchedule(JSON.stringify(bad2))).toBeNull();
  });

  it("marks old-format records as legacy snapshots without fabricating metadata", () => {
    const raw = JSON.stringify(VALID);
    const parsed = parseStoredSchedule(raw);
    expect(parsed).not.toBeNull();
    // 旧格式无模型字段：显式补 model: null，其余元信息保持缺省（不补造）。
    expect(parsed!.model).toBeNull();
    expect(parsed!.configVersion).toBeUndefined();
    expect(parsed!.asOf).toBeUndefined();
    expect(parsed!.inputHash).toBeUndefined();
    expect(parsed!.result.model).toBeNull();
    expect(parsed!.result.configVersion).toBeUndefined();
    // 阵容与结果原样保留。
    expect(parsed!.result.schedule).toEqual(VALID.result.schedule);
    expect(parsed!.playerIds).toEqual(VALID.playerIds);
  });

  it("round-trips a glicko2 record with model metadata", () => {
    const glicko2: StoredSchedule = {
      ...VALID,
      model: "glicko2",
      configVersion: "p1:2026-07-01",
      asOf: "2026-09-22T12:00:00.000Z",
      inputHash: "abc123",
      result: {
        ...VALID.result,
        model: "glicko2",
        configVersion: "p1:2026-07-01",
        asOf: "2026-09-22T12:00:00.000Z",
        inputHash: "abc123",
      },
    };
    expect(parseStoredSchedule(JSON.stringify(glicko2))).toEqual(glicko2);
  });

  it("rejects invalid model metadata", () => {
    const badModel = structuredClone(VALID);
    // @ts-expect-error 故意造坏数据
    badModel.model = "trueskill";
    expect(parseStoredSchedule(JSON.stringify(badModel))).toBeNull();

    const badInner = structuredClone(VALID);
    // @ts-expect-error 故意造坏数据
    badInner.result.model = "elo";
    expect(parseStoredSchedule(JSON.stringify(badInner))).toBeNull();

    const badHash = structuredClone(VALID);
    // @ts-expect-error 故意造坏数据
    badHash.result.inputHash = 123;
    expect(parseStoredSchedule(JSON.stringify(badHash))).toBeNull();
  });
});
