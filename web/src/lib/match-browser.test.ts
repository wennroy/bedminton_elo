import { describe, expect, it } from "vitest";
import type { MatchWithNames } from "@/lib/repo";
import { filterMatches, type MatchFilters } from "./match-browser";

const noFilters: MatchFilters = {
  query: "",
  playerId: null,
  from: "",
  to: "",
};

function match(overrides: Partial<MatchWithNames> = {}): MatchWithNames {
  return {
    id: 1,
    pa1: 11,
    pa2: 12,
    pb1: 13,
    pb2: 14,
    scoreA: 21,
    scoreB: 19,
    playedAt: "2025-01-15",
    enteredBy: null,
    createdAt: "2025-01-15 10:00:00",
    pa1Name: "张伟",
    pa2Name: "李娜",
    pb1Name: "王强",
    pb2Name: "陈晨",
    ...overrides,
  };
}

describe("filterMatches", () => {
  it("matches a Chinese substring in every player-name position", () => {
    const matches = [
      match({ id: 1, pa1Name: "小明" }),
      match({ id: 2, pa2Name: "阿明" }),
      match({ id: 3, pb1Name: "明月" }),
      match({ id: 4, pb2Name: "明华" }),
    ];

    expect(filterMatches(matches, { ...noFilters, query: "明" }).map((m) => m.id)).toEqual([
      4, 3, 2, 1,
    ]);
  });

  it("matches English names case-insensitively after trimming the query", () => {
    const matches = [
      match({ id: 1, pa1Name: "ALIce" }),
      match({ id: 2, pa1Name: "Bob" }),
    ];

    expect(
      filterMatches(matches, { ...noFilters, query: "  aLi  " }).map((m) => m.id)
    ).toEqual([1]);
  });

  it("does not limit results for an empty or whitespace-only query", () => {
    const matches = [
      match({ id: 1, playedAt: "2025-01-01" }),
      match({ id: 2, playedAt: "2025-01-02" }),
    ];

    expect(filterMatches(matches, noFilters).map((m) => m.id)).toEqual([2, 1]);
    expect(filterMatches(matches, { ...noFilters, query: "   " }).map((m) => m.id)).toEqual([
      2, 1,
    ]);
  });

  it.each([
    ["pa1", 11],
    ["pa2", 12],
    ["pb1", 13],
    ["pb2", 14],
  ] as const)("matches a selected player in %s", (_position, playerId) => {
    const matches = [
      match({ id: 1 }),
      match({ id: 2, pa1: 91, pa2: 92, pb1: 93, pb2: 94 }),
    ];

    expect(filterMatches(matches, { ...noFilters, playerId }).map((m) => m.id)).toEqual([1]);
  });

  it("selects by player ID when different players have the same name", () => {
    const matches = [
      match({ id: 1, pa1: 42, pa1Name: "Alex" }),
      match({ id: 2, pa1: 43, pa1Name: "Alex" }),
    ];

    expect(filterMatches(matches, { ...noFilters, playerId: 42 }).map((m) => m.id)).toEqual([1]);
  });

  it("intersects query, selected player, and date constraints", () => {
    const matches = [
      match({ id: 1, pa1: 42, pa1Name: "王小明", playedAt: "2025-03-10" }),
      match({ id: 2, pa1: 42, pa1Name: "王小明", playedAt: "2025-03-20" }),
      match({ id: 3, pa1: 43, pa1Name: "王小明", playedAt: "2025-03-10" }),
      match({
        id: 4,
        pa1: 42,
        pa1Name: "李娜",
        pa2Name: "赵敏",
        pb1Name: "孙浩",
        pb2Name: "周洁",
        playedAt: "2025-03-10",
      }),
    ];

    expect(
      filterMatches(matches, {
        query: "王",
        playerId: 42,
        from: "2025-03-01",
        to: "2025-03-15",
      }).map((m) => m.id)
    ).toEqual([1]);
  });

  it("uses inclusive start and end dates", () => {
    const matches = [
      match({ id: 1, playedAt: "2025-01-01" }),
      match({ id: 2, playedAt: "2025-01-15" }),
      match({ id: 3, playedAt: "2025-01-31" }),
      match({ id: 4, playedAt: "2025-02-01" }),
    ];

    expect(
      filterMatches(matches, { ...noFilters, from: "2025-01-01", to: "2025-01-31" }).map(
        (m) => m.id
      )
    ).toEqual([3, 2, 1]);
  });

  it("supports single-sided and cross-year date bounds", () => {
    const matches = [
      match({ id: 1, playedAt: "2024-12-30" }),
      match({ id: 2, playedAt: "2024-12-31" }),
      match({ id: 3, playedAt: "2025-01-01" }),
      match({ id: 4, playedAt: "2025-01-02" }),
    ];

    expect(filterMatches(matches, { ...noFilters, from: "2025-01-01" }).map((m) => m.id)).toEqual([
      4, 3,
    ]);
    expect(filterMatches(matches, { ...noFilters, to: "2024-12-31" }).map((m) => m.id)).toEqual([
      2, 1,
    ]);
    expect(
      filterMatches(matches, { ...noFilters, from: "2024-12-31", to: "2025-01-01" }).map(
        (m) => m.id
      )
    ).toEqual([3, 2]);
  });

  it.each(["2025-02-29", "2025-02-3", "2025-04-31"])(
    "returns no matches for invalid date bound %s",
    (invalidDate) => {
      const matches = [match()];

      expect(filterMatches(matches, { ...noFilters, from: invalidDate })).toEqual([]);
      expect(filterMatches(matches, { ...noFilters, to: invalidDate })).toEqual([]);
    }
  );

  it("returns no matches when the start date is after the end date", () => {
    expect(
      filterMatches([match()], { ...noFilters, from: "2025-01-16", to: "2025-01-15" })
    ).toEqual([]);
  });

  it("sorts by played date, creation time, then descending ID", () => {
    const matches = [
      match({ id: 1, playedAt: "2025-05-01", createdAt: "2025-05-01 10:00:00" }),
      match({ id: 2, playedAt: "2025-05-02", createdAt: "2025-05-02 09:00:00" }),
      match({ id: 3, playedAt: "2025-05-02", createdAt: "2025-05-02T09:00:01Z" }),
      match({ id: 4, playedAt: "2025-05-02", createdAt: "2025-05-02T09:00:01Z" }),
    ];

    expect(filterMatches(matches, noFilters).map((m) => m.id)).toEqual([4, 3, 2, 1]);
  });

  it("uses IDs deterministically when creation times cannot be parsed", () => {
    const matches = [
      match({ id: 10, createdAt: "not-a-date" }),
      match({ id: 11, createdAt: "also-not-a-date" }),
    ];

    expect(filterMatches(matches, noFilters).map((m) => m.id)).toEqual([11, 10]);
  });

  it("orders valid creation times before invalid ones regardless of input order", () => {
    const newest = match({ id: 1, createdAt: "2025-01-15T11:00:00Z" });
    const older = match({ id: 3, createdAt: "2025-01-15 10:00:00" });
    const invalid = match({ id: 2, createdAt: "not-a-date" });

    expect(filterMatches([newest, older, invalid], noFilters).map((m) => m.id)).toEqual([
      1, 3, 2,
    ]);
    expect(filterMatches([older, invalid, newest], noFilters).map((m) => m.id)).toEqual([
      1, 3, 2,
    ]);
  });

  it("ranks an invalid ISO calendar date after valid UTC and offset timestamps", () => {
    const valid = match({ id: 1, createdAt: "2025-02-28T11:00:00Z" });
    const invalid = match({ id: 2, createdAt: "2025-02-30T10:00:00Z" });
    const offset = match({ id: 4, createdAt: "2025-02-28T20:00:00+08:00" });

    expect(filterMatches([valid, invalid, offset], noFilters).map((m) => m.id)).toEqual([4, 1, 2]);
    expect(filterMatches([invalid, offset, valid], noFilters).map((m) => m.id)).toEqual([4, 1, 2]);
  });

  it("returns no results when no record matches", () => {
    expect(filterMatches([match()], { ...noFilters, query: "不存在" })).toEqual([]);
  });

  it("does not mutate the input array or its records", () => {
    const matches = Object.freeze([
      Object.freeze(match({ id: 1, playedAt: "2025-01-01" })),
      Object.freeze(match({ id: 2, playedAt: "2025-01-02" })),
    ]);
    const before = matches.map((m) => ({ ...m }));

    const result = filterMatches(matches, noFilters);

    expect(result.map((m) => m.id)).toEqual([2, 1]);
    expect(matches).toEqual(before);
  });

  it("returns a matching record beyond the first 20 input records", () => {
    const matches = Array.from({ length: 21 }, (_, index) =>
      match({ id: index + 1, pa1Name: `玩家${index}`, playedAt: "2025-01-01" })
    );
    matches.push(match({ id: 99, pa1Name: "Needle Player", playedAt: "2025-12-31" }));

    expect(filterMatches(matches, { ...noFilters, query: "needle" }).map((m) => m.id)).toEqual([
      99,
    ]);
  });
});
