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
    expect(parseStoredSchedule(JSON.stringify(VALID))).toEqual(VALID);
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
});
