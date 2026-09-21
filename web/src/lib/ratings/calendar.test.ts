import { describe, expect, it } from "vitest";
import {
  assertLocalDate,
  isFutureShanghaiLocalDate,
  isValidLocalDate,
  iterateRatingSegments,
  nextQuarterStart,
  nextRatingBoundary,
  nextWeekStart,
  quarterStart,
  ratingSegmentAt,
  shanghaiLocalDateFromInstant,
  shanghaiMidnightIso,
  weekStart,
} from "./calendar";

describe("rating calendar", () => {
  it("strictly validates Gregorian local dates including leap days", () => {
    expect(isValidLocalDate("2024-02-29")).toBe(true);
    expect(isValidLocalDate("2023-02-29")).toBe(false);
    expect(isValidLocalDate("2026-04-31")).toBe(false);
    expect(isValidLocalDate("2026-2-01")).toBe(false);
    expect(isValidLocalDate("0000-01-01")).toBe(false);

    expect(() => assertLocalDate("2024-02-29")).not.toThrow();
    expect(() => assertLocalDate("2023-02-29")).toThrow(
      /valid YYYY-MM-DD Gregorian date/
    );
  });

  it("converts instants and local midnights in Shanghai instead of the host time zone", () => {
    expect(shanghaiLocalDateFromInstant("2026-09-30T15:59:59Z")).toBe("2026-09-30");
    expect(shanghaiLocalDateFromInstant("2026-09-30T16:00:00Z")).toBe("2026-10-01");
    expect(shanghaiMidnightIso("2026-10-01")).toBe("2026-09-30T16:00:00Z");
    expect(shanghaiMidnightIso("2024-02-29")).toBe("2024-02-28T16:00:00Z");
    expect(shanghaiMidnightIso("1986-06-02")).toBe("1986-06-01T15:00:00Z");
    expect(shanghaiMidnightIso("1919-04-13")).toBe("1919-04-12T16:00:00Z");
    expect(shanghaiLocalDateFromInstant(shanghaiMidnightIso("1919-04-13"))).toBe(
      "1919-04-13"
    );
  });

  it("normalizes early Gregorian years to strict four-digit local dates", () => {
    expect(shanghaiLocalDateFromInstant("0001-01-01T00:00:00Z")).toBe("0001-01-01");
    expect(shanghaiLocalDateFromInstant("0099-12-31T00:00:00Z")).toBe("0099-12-31");
    for (const localDate of ["0001-01-01", "0099-12-31"]) {
      expect(shanghaiLocalDateFromInstant(shanghaiMidnightIso(localDate))).toBe(localDate);
    }
  });

  it("finds Monday and natural-quarter boundaries with pure Gregorian dates", () => {
    expect(weekStart("2026-10-04")).toBe("2026-09-28");
    expect(nextWeekStart("2026-12-28")).toBe("2027-01-04");
    expect(quarterStart("2026-12-31")).toBe("2026-10-01");
    expect(nextQuarterStart("2026-12-31")).toBe("2027-01-01");
    expect(nextRatingBoundary("2026-09-28")).toBe("2026-10-01");
  });

  it("compares future match dates to an as-of instant in Shanghai", () => {
    expect(isFutureShanghaiLocalDate("2026-10-01", "2026-09-30T15:59:59Z")).toBe(true);
    expect(isFutureShanghaiLocalDate("2026-10-01", "2026-09-30T16:00:00Z")).toBe(false);
  });

  it("splits a Shanghai week at its quarterly boundary with exact partial-week h", () => {
    const segments = iterateRatingSegments({
      start: "2026-09-27T16:00:00Z",
      end: "2026-10-04T16:00:00Z",
      firstSeasonStart: "2026-10-01",
    });

    expect(segments).toEqual([
      {
        id: "2026-09-28:2026-09-28",
        weekStart: "2026-09-28",
        seasonId: null,
        start: "2026-09-27T16:00:00Z",
        end: "2026-09-30T16:00:00Z",
        h: 3 / 7,
      },
      {
        id: "2026-09-28:2026-10-01",
        weekStart: "2026-09-28",
        seasonId: "2026-10-01",
        start: "2026-09-30T16:00:00Z",
        end: "2026-10-04T16:00:00Z",
        h: 4 / 7,
      },
    ]);
    expect(segments[0].h + segments[1].h).toBe(1);
  });

  it("puts an as-of instant exactly on a boundary in the new non-empty segment", () => {
    expect(ratingSegmentAt("2026-09-30T16:00:00Z", "2026-10-01")).toEqual({
      id: "2026-09-28:2026-10-01",
      weekStart: "2026-09-28",
      seasonId: "2026-10-01",
      start: "2026-09-30T16:00:00Z",
      end: "2026-10-04T16:00:00Z",
      h: 4 / 7,
    });
  });

  it("does not identify a season before the configured first season boundary", () => {
    expect(ratingSegmentAt("2026-06-30T16:00:00Z", "2026-10-01")).toMatchObject({
      start: "2026-06-30T16:00:00Z",
      end: "2026-07-05T16:00:00Z",
      seasonId: null,
    });
  });

  it("rejects a first season date that bypasses config but is not a quarter start", () => {
    expect(() => ratingSegmentAt("2026-02-01T00:00:00Z", "2026-02-01")).toThrow(
      /firstSeasonStart must be a valid quarter start/
    );
    expect(() =>
      iterateRatingSegments({
        start: "2026-01-04T16:00:00Z",
        end: "2026-01-11T16:00:00Z",
        firstSeasonStart: "2026-02-01",
      })
    ).toThrow(/firstSeasonStart must be a valid quarter start/);
  });

  it("begins at a first mid-week match without creating an earlier or zero-length segment", () => {
    const segments = iterateRatingSegments({
      start: "2026-09-29T16:00:00Z",
      end: "2026-10-04T16:00:00Z",
      firstSeasonStart: "2026-10-01",
    });

    expect(segments.map(({ start, end, weekStart, h }) => ({ start, end, weekStart, h }))).toEqual([
      {
        start: "2026-09-29T16:00:00Z",
        end: "2026-09-30T16:00:00Z",
        weekStart: "2026-09-28",
        h: 1 / 7,
      },
      {
        start: "2026-09-30T16:00:00Z",
        end: "2026-10-04T16:00:00Z",
        weekStart: "2026-09-28",
        h: 4 / 7,
      },
    ]);
    expect(segments[0].id).not.toBe(segments[1].id);
    expect(segments.every((segment) => segment.h > 0)).toBe(true);
  });

  it("keeps a Monday quarterly boundary as one segment", () => {
    expect(ratingSegmentAt("2023-12-31T16:00:00Z", "2024-01-01")).toMatchObject({
      weekStart: "2024-01-01",
      seasonId: "2024-01-01",
      start: "2023-12-31T16:00:00Z",
      end: "2024-01-07T16:00:00Z",
      h: 1,
    });
  });

  it("returns no segment for an empty boundary range", () => {
    expect(
      iterateRatingSegments({
        start: "2026-10-04T16:00:00Z",
        end: "2026-10-04T16:00:00Z",
        firstSeasonStart: "2026-10-01",
      })
    ).toEqual([]);
  });

  it("rejects rating-boundary calculations that would exceed the LocalDate maximum", () => {
    expect(isValidLocalDate("9999-12-30")).toBe(true);
    const expectedError = /cannot calculate a rating boundary after 9999-12-31/;
    expect(() => nextRatingBoundary("9999-12-30")).toThrow(expectedError);
    expect(() => ratingSegmentAt("9999-12-30T00:00:00Z", "9999-10-01")).toThrow(
      expectedError
    );
    expect(() =>
      iterateRatingSegments({
        start: "9999-12-26T16:00:00Z",
        end: "9999-12-30T16:00:00Z",
        firstSeasonStart: "9999-10-01",
      })
    ).toThrow(expectedError);
  });
});
