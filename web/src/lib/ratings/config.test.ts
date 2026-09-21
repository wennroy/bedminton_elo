import { describe, expect, it } from "vitest";
import {
  GLICKO2_DISPLAY_SCALE,
  createRatingConfig,
  ratingConfigVersion,
} from "./config";
import * as ratingConfig from "./config";
import type { RatingConfig } from "./types";

describe("rating config", () => {
  it("converts display ratings and RD to the shared internal coordinates", () => {
    expect(ratingConfig.displayRatingToInternalX).toBeTypeOf("function");
    expect(ratingConfig.displayRdToInternalPhi).toBeTypeOf("function");
    expect(ratingConfig.displayRatingToInternalX(1000)).toBe(0);
    expect(
      ratingConfig.displayRatingToInternalX(1000 + 2 * GLICKO2_DISPLAY_SCALE)
    ).toBe(1);
    expect(ratingConfig.displayRdToInternalPhi(2 * GLICKO2_DISPLAY_SCALE)).toBe(1);
  });

  it("creates the pinned p1 defaults in display scale", () => {
    const config: RatingConfig = createRatingConfig({
      firstSeasonStart: "2024-01-01",
    });

    expect(GLICKO2_DISPLAY_SCALE).toBe(173.7178);
    expect(config).toEqual({
      algorithmVersion: "glicko2-doubles-v1",
      paramsVersion: "p1",
      firstSeasonStart: "2024-01-01",
      timeZone: "Asia/Shanghai",
      initialRating: 1000,
      initialRd: 180,
      minRd: 60,
      maxRd: 250,
      initialVolatility: 0.06,
      tau: 0.3,
      seasonLower: 900,
      seasonUpper: 1100,
      seasonRetention: 0.75,
      seasonRdFloor: 90,
    });
    expect(ratingConfigVersion(config)).toBe(
      "glicko2-doubles-v1|p1|2024-01-01|Asia%2FShanghai|1000|180|60|250|0.06|0.3|900|1100|0.75|90"
    );
  });

  it.each([
    ["NaN", { initialRating: Number.NaN }, /initialRating must be a finite number/],
    ["infinity", { maxRd: Infinity }, /maxRd must be a finite number/],
    ["negative volatility", { initialVolatility: -0.06 }, /greater than zero/],
    ["zero tau", { tau: 0 }, /tau must be greater than zero/],
    ["reversed RD bounds", { minRd: 251 }, /minRd must be less than or equal to maxRd/],
    ["initial RD below its minimum", { initialRd: 59 }, /initialRd must be within minRd and maxRd/],
    ["season RD floor below its minimum", { seasonRdFloor: 59 }, /seasonRdFloor must be within minRd and maxRd/],
    ["retention over one", { seasonRetention: 1.01 }, /seasonRetention must be between zero and one/],
    ["a non-quarter date", { firstSeasonStart: "2024-02-01" }, /valid YYYY-MM-DD quarter start/],
    ["year zero", { firstSeasonStart: "0000-01-01" }, /valid YYYY-MM-DD quarter start/],
  ] as const)("rejects %s with a clear error", (_name, overrides, expectedError) => {
    expect(() =>
      createRatingConfig({ firstSeasonStart: "2024-01-01", ...overrides })
    ).toThrow(expectedError);
  });

  it.each([
    ["initial rating", { initialRating: -1 }, /initialRating must be greater than or equal to zero/],
    ["minimum RD", { minRd: -1 }, /minRd must be greater than or equal to zero/],
    ["season lower bound", { seasonLower: -1 }, /seasonLower must be greater than or equal to zero/],
  ] as const)("rejects a negative %s", (_name, overrides, expectedError) => {
    expect(() =>
      createRatingConfig({ firstSeasonStart: "2024-01-01", ...overrides })
    ).toThrow(expectedError);
  });

  it("allows zero at valid non-negative boundaries", () => {
    expect(
      createRatingConfig({
        firstSeasonStart: "2024-01-01",
        initialRating: 0,
        minRd: 0,
        seasonLower: 0,
        seasonRetention: 0,
      })
    ).toMatchObject({
      initialRating: 0,
      minRd: 0,
      seasonLower: 0,
      seasonRetention: 0,
    });
  });

  it("changes the cache version when effective parameters change", () => {
    const base = createRatingConfig({ firstSeasonStart: "2024-01-01" });
    const changedTau = createRatingConfig({
      firstSeasonStart: "2024-01-01",
      tau: 0.4,
    });

    expect(ratingConfigVersion(changedTau)).not.toBe(ratingConfigVersion(base));
  });
});
