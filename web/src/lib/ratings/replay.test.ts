import { describe, expect, it } from "vitest";
import { createRatingConfig } from "./config";
import { replayRatings } from "./replay";
import { replayConfig, replayMatch } from "../../../test/fixtures/ratings-scenarios";

const playerIds = [1, 2, 3, 4, 5, 6] as const;

function matchEstimateIds(result: ReturnType<typeof replayRatings>): number[] {
  return result.events
    .filter((event) => event.kind === "match_estimated")
    .map((event) => event.matchId);
}

describe("Glicko-2 doubles replay", () => {
  it("orders Estimated updates by playedAt, createdAt, then id and finalizes a prior week", () => {
    const matches = [
      replayMatch(3, "2026-10-06", 17, 21),
      replayMatch(2, "2026-09-29", 21, 17, [1, 3], [2, 4], "2026-09-29T13:00:00+08:00"),
      replayMatch(1, "2026-09-29", 21, 17, [1, 2], [3, 4], "2026-09-29T11:00:00+08:00"),
    ];

    const replay = replayRatings(matches, playerIds, {
      config: createRatingConfig({ firstSeasonStart: "2027-01-01" }),
      asOf: "2026-10-10T12:00:00+08:00",
    });

    expect(matchEstimateIds(replay)).toEqual([1, 2, 3]);
    expect(replay.events.filter((event) => event.kind === "weekly_final")).toHaveLength(2);
    expect(replay.currentSegment.id).toBe("2026-10-05:2026-10-05");
    expect("states" in (replay.matchEstimates["3"] as unknown as Record<string, unknown>)).toBe(false);
  });

  it("orders mixed createdAt storage formats by their original strings, without a host time zone", () => {
    const replay = replayRatings(
      [
        replayMatch(2, "2026-10-06", 21, 17, [1, 3], [2, 4], "2026-10-06T11:00:00+08:00"),
        replayMatch(1, "2026-10-06", 21, 17, [1, 2], [3, 4], "2026-10-06 11:30:00"),
      ],
      playerIds,
      { config: replayConfig, asOf: "2026-10-10T12:00:00+08:00" }
    );

    expect(matchEstimateIds(replay)).toEqual([1, 2]);
  });

  it("splits the September/October boundary, Finalizes before reset, then starts new estimates", () => {
    const replay = replayRatings(
      [
        replayMatch(1, "2026-09-28"),
        replayMatch(2, "2026-10-02", 17, 21),
        replayMatch(3, "2026-10-05"),
      ],
      playerIds,
      { config: replayConfig, asOf: "2026-10-06T10:00:00+08:00" }
    );

    expect(
      replay.events.map((event) =>
        event.kind === "weekly_final" ? `${event.kind}:${event.segment.h}` : event.kind
      )
    ).toEqual([
      "match_estimated",
      "weekly_final:0.42857142857142855",
      "season_reset",
      "match_estimated",
      "weekly_final:0.5714285714285714",
      "match_estimated",
    ]);
    expect(replay.events[2]).toMatchObject({
      kind: "season_reset",
      eventId: "season_reset:2026-10-01",
      seasonId: "2026-10-01",
    });
  });

  it("advances empty weeks after the first match through season boundaries while capping RD", () => {
    const config = createRatingConfig({
      firstSeasonStart: "2026-10-01",
      maxRd: 190,
      seasonRdFloor: 90,
      seasonLower: 999,
      seasonUpper: 1001,
    });
    const replay = replayRatings(
      [replayMatch(1, "2026-09-28")],
      playerIds,
      { config, asOf: "2027-03-15T09:00:00+08:00" }
    );

    expect(replay.events.filter((event) => event.kind === "weekly_final").length).toBeGreaterThan(20);
    expect(replay.events.filter((event) => event.kind === "season_reset").map((event) => event.seasonId)).toEqual([
      "2026-10-01",
      "2027-01-01",
    ]);
    for (const state of Object.values(replay.current)) expect(state.rd).toBeLessThanOrEqual(config.maxRd);
  });

  it("does not create history before the first real match or before firstSeasonStart", () => {
    const config = createRatingConfig({ firstSeasonStart: "2027-01-01" });
    const replay = replayRatings(
      [replayMatch(1, "2026-09-29"), replayMatch(2, "2026-10-06", 21, 17, [1, 5], [3, 4])],
      playerIds,
      { config, asOf: "2026-10-10T12:00:00+08:00" }
    );

    expect(replay.events.filter((event) => event.kind === "season_reset")).toHaveLength(0);
    const firstFinal = replay.events.find((event) => event.kind === "weekly_final");
    expect(firstFinal).toMatchObject({ kind: "weekly_final" });
    if (firstFinal?.kind !== "weekly_final") throw new Error("missing Final");
    expect(firstFinal.start["5"]).toBeUndefined();
    expect(replay.current["5"]).toBeDefined();
    expect(replay.currentSegment.start).toBe("2026-10-04T16:00:00Z");
  });

  it("records one deterministic issue per effective invalid record and ignores future records", () => {
    const invalidDate = replayMatch(4, "2026-02-30");
    const invalidScore = replayMatch(2, "2026-09-28", 21, 21);
    const duplicatePlayer = replayMatch(3, "2026-09-28", 21, 18, [1, 1], [3, 4]);
    const unknownPlayer = replayMatch(1, "2026-09-28", 21, 18, [1, 2], [3, 9]);
    const futureBad = replayMatch(5, "2026-11-01", -1, -2, [7, 7], [8, 8]);
    const replay = replayRatings(
      [invalidDate, invalidScore, duplicatePlayer, unknownPlayer, futureBad],
      playerIds,
      { config: replayConfig, asOf: "2026-10-10T12:00:00+08:00" }
    );

    expect(replay.issues).toEqual([
      { matchId: 1, reason: "unknown_player" },
      { matchId: 2, reason: "invalid_score" },
      { matchId: 3, reason: "duplicate_player" },
      { matchId: 4, reason: "invalid_date" },
    ]);
    expect(replay.events).toEqual([]);
    expect(replay.current).toEqual({});
  });

  it("rejects unsafe or duplicate match ids before a map key could collide", () => {
    const options = { config: replayConfig, asOf: "2026-10-10T12:00:00+08:00" };
    expect(() => replayRatings([replayMatch(1, "2026-09-28"), replayMatch(1, "2026-09-29")], playerIds, options)).toThrow(
      /duplicate match.id/
    );
    expect(() => replayRatings([replayMatch(Number.MAX_SAFE_INTEGER + 1, "2026-09-28")], playerIds, options)).toThrow(
      /match.id must be a non-negative safe integer/
    );
  });

  it("is pure and deterministic, while a changed earlier outcome replays later estimates", () => {
    const first = replayMatch(1, "2026-10-06", 21, 17, [1, 2], [3, 4], "2026-10-06T11:00:00+08:00");
    const second = replayMatch(2, "2026-10-06", 21, 17, [1, 2], [3, 4], "2026-10-06T12:00:00+08:00");
    const matches = Object.freeze([Object.freeze(first), Object.freeze(second)]);
    const options = Object.freeze({ config: Object.freeze(replayConfig), asOf: "2026-10-10T12:00:00+08:00" });

    const one = replayRatings(matches, playerIds, options);
    const two = replayRatings(matches, playerIds, options);
    const changed = replayRatings([{ ...first, scoreA: 17, scoreB: 21 }, second], playerIds, options);

    expect(two).toEqual(one);
    expect(changed.matchEstimates["2"].preWinA).not.toBeCloseTo(one.matchEstimates["2"].preWinA, 12);
    expect(matches[0]).toEqual(first);
  });

  it("at an exact Monday-and-quarter boundary closes once, resets once, and opens no Final for the new segment", () => {
    const config = createRatingConfig({ firstSeasonStart: "2029-01-01" });
    const replay = replayRatings(
      [replayMatch(1, "2028-12-31")],
      playerIds,
      { config, asOf: "2029-01-01T00:00:00+08:00" }
    );

    expect(replay.events.filter((event) => event.kind === "weekly_final")).toHaveLength(1);
    expect(replay.events.filter((event) => event.kind === "season_reset")).toHaveLength(1);
    expect(replay.currentSegment.weekStart).toBe("2029-01-01");
    expect(replay.currentSegment.start).toBe("2028-12-31T16:00:00Z");
    expect(replay.nextBoundary).toBe(replay.currentSegment.end);
  });

  it("returns no ratings when there are no effective valid matches", () => {
    const replay = replayRatings([replayMatch(1, "2026-10-20")], playerIds, {
      config: replayConfig,
      asOf: "2026-10-10T12:00:00+08:00",
    });

    expect(replay.current).toEqual({});
    expect(replay.lastFinal).toEqual({});
    expect(replay.events).toEqual([]);
    expect(replay.matchEstimates).toEqual({});
    expect(replay.nextBoundary).toBe(replay.currentSegment.end);
  });

  it("requires an asOf instant with an explicit time zone", () => {
    expect(() =>
      replayRatings([replayMatch(1, "2026-10-08")], playerIds, {
        config: replayConfig,
        asOf: "2026-10-10T12:00:00",
      })
    ).toThrow(/explicit time zone/);
  });

  it("keeps the first mid-week segment anchored at the first real match date", () => {
    const replay = replayRatings([replayMatch(1, "2026-10-08")], playerIds, {
      config: replayConfig,
      asOf: "2026-10-10T12:00:00+08:00",
    });

    expect(replay.currentSegment.id).toBe("2026-10-05:2026-10-08");
    expect(replay.currentSegment.h).toBe(4 / 7);
  });
});
