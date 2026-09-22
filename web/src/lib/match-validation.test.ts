import { describe, expect, it } from "vitest";
import { assertValidMatchInput, MatchValidationError } from "./match-validation";

const KNOWN = new Set([1, 2, 3, 4, 5]);

const VALID = {
  pa1: 1,
  pa2: 2,
  pb1: 3,
  pb2: 4,
  scoreA: 21,
  scoreB: 15,
  playedAt: "2026-09-21",
};

function expectInvalid(input: Partial<typeof VALID>, code: string) {
  try {
    assertValidMatchInput({ ...VALID, ...input }, KNOWN);
  } catch (error) {
    expect(error).toBeInstanceOf(MatchValidationError);
    expect((error as MatchValidationError).code).toBe(code);
    return;
  }
  throw new Error(`expected MatchValidationError(${code})`);
}

describe("assertValidMatchInput", () => {
  it("接受合法输入（含未来日期与超过 21 分的比分）", () => {
    expect(() =>
      assertValidMatchInput({ ...VALID, playedAt: "2099-01-01", scoreA: 30 }, KNOWN)
    ).not.toThrow();
  });

  it("拒绝非整数/非安全整数球员 ID", () => {
    expectInvalid({ pa1: 1.5 }, "invalid_player_id");
    expectInvalid({ pa1: Number.NaN }, "invalid_player_id");
    expectInvalid({ pb2: Number.MAX_SAFE_INTEGER + 1 }, "invalid_player_id");
  });

  it("拒绝重复球员", () => {
    expectInvalid({ pb1: 1 }, "duplicate_player");
    expectInvalid({ pa2: 4 }, "duplicate_player");
  });

  it("拒绝目录外的球员 ID", () => {
    expectInvalid({ pb2: 999 }, "unknown_player");
    expectInvalid({ pa1: 0 }, "unknown_player");
  });

  it("拒绝非法日历日期（含 2026-02-30 这类形式合法但不存在的日期）", () => {
    expectInvalid({ playedAt: "2026-02-30" }, "invalid_date");
    expectInvalid({ playedAt: "2026-13-01" }, "invalid_date");
    expectInvalid({ playedAt: "2026-1-1" }, "invalid_date");
    expectInvalid({ playedAt: "not-a-date" }, "invalid_date");
  });

  it("拒绝负分/非整数比分/平分", () => {
    expectInvalid({ scoreA: -1 }, "invalid_score");
    expectInvalid({ scoreB: 21.5 }, "invalid_score");
    expectInvalid({ scoreA: 21, scoreB: 21 }, "invalid_score");
  });
});
