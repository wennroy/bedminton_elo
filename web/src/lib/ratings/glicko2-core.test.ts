import { describe, expect, it } from "vitest";
import golden from "../../../test/golden/glicko2-reference.json";
import { updateGlicko2 } from "./glicko2-core";

const OFFICIAL_GLICKO2_SCALE = 173.7178;

describe("Glicko-2 core", () => {
  it("matches the official worked example in standard Glicko-2 coordinates", () => {
    const example = golden.officialExample;
    const updated = updateGlicko2(
      {
        x: (example.input.rating - 1500) / OFFICIAL_GLICKO2_SCALE,
        phi: example.input.rd / OFFICIAL_GLICKO2_SCALE,
        sigma: example.input.sigma,
      },
      example.observations.map((observation) => ({
        opponentX: (observation.rating - 1500) / OFFICIAL_GLICKO2_SCALE,
        opponentPhi: observation.rd / OFFICIAL_GLICKO2_SCALE,
        result: observation.result as 0 | 1,
      })),
      example.tau
    );

    expect(Math.abs(updated.x * OFFICIAL_GLICKO2_SCALE + 1500 - example.expected.rating)).toBeLessThanOrEqual(0.01);
    expect(Math.abs(updated.phi * OFFICIAL_GLICKO2_SCALE - example.expected.rd)).toBeLessThanOrEqual(0.01);
    expect(Math.abs(updated.sigma - example.expected.sigma)).toBeLessThanOrEqual(0.00001);
  });

  it("only expands phi during an empty period, including from zero phi", () => {
    const state = { x: 3.5, phi: 0, sigma: Number.MIN_VALUE };

    const updated = updateGlicko2(state, [], 0.5);

    expect(updated.x).toBe(state.x);
    expect(updated.phi).toBe(state.sigma);
    expect(updated.sigma).toBe(state.sigma);
  });

  it("reports a typed error when an empty-period variance cannot be represented", () => {
    let caught: unknown;
    try {
      updateGlicko2(
        { x: 0, phi: Number.MAX_VALUE, sigma: Number.MAX_VALUE },
        [],
        0.5
      );
    } catch (error) {
      caught = error;
    }

    expect(caught).toMatchObject({ code: "GLICKO2_CONVERGENCE_FAILURE" });
  });

  it("keeps an update finite for extreme but finite coordinate differences", () => {
    const updated = updateGlicko2(
      { x: Number.MAX_VALUE, phi: 0.5, sigma: 0.06 },
      [{ opponentX: -Number.MAX_VALUE, opponentPhi: Number.MAX_VALUE, result: 0 }],
      0.5
    );

    expect(Number.isFinite(updated.x)).toBe(true);
    expect(Number.isFinite(updated.phi)).toBe(true);
    expect(Number.isFinite(updated.sigma)).toBe(true);
  });

  it("treats an unrepresentable observation variance as an empty period", () => {
    const state = { x: 0, phi: Number.MAX_VALUE, sigma: 0.06 };

    const updated = updateGlicko2(
      state,
      [{ opponentX: 770, opponentPhi: 0.5, result: 1 }],
      0.5
    );

    expect(updated).toEqual({ x: state.x, phi: state.phi, sigma: state.sigma });
  });

  it.each([
    { observations: [{ opponentX: 0, opponentPhi: Number.MAX_VALUE, result: 1 as const }] },
    { observations: [{ opponentX: 740, opponentPhi: 0.5, result: 1 as const }] },
  ])("reports a typed error when an observed fallback cannot expand variance", ({ observations }) => {
    let caught: unknown;
    try {
      updateGlicko2(
        { x: 0, phi: Number.MAX_VALUE, sigma: Number.MAX_VALUE },
        observations,
        0.5
      );
    } catch (error) {
      caught = error;
    }

    expect(caught).toMatchObject({ code: "GLICKO2_CONVERGENCE_FAILURE" });
  });

  it("retains a representable subnormal information contribution from a huge phi", () => {
    const updated = updateGlicko2(
      { x: 0, phi: 0.5, sigma: 0.06 },
      [{ opponentX: 0, opponentPhi: 1e154, result: 1 }],
      0.5
    );

    const expectedX = 2.299897593848988e-155;
    expect(updated.x).toBeGreaterThan(0);
    expect(Math.abs(updated.x - expectedX) / expectedX).toBeLessThan(1e-12);
  });

  it.each([
    ["a non-finite state x", { x: Number.NaN, phi: 0.5, sigma: 0.06 }, [], 0.5],
    ["a negative state phi", { x: 0, phi: -0.5, sigma: 0.06 }, [], 0.5],
    ["a zero state sigma", { x: 0, phi: 0.5, sigma: 0 }, [], 0.5],
    ["a non-positive tau", { x: 0, phi: 0.5, sigma: 0.06 }, [], 0],
    [
      "a non-finite opponent x",
      { x: 0, phi: 0.5, sigma: 0.06 },
      [{ opponentX: Infinity, opponentPhi: 0.5, result: 1 }],
      0.5,
    ],
    [
      "a negative opponent phi",
      { x: 0, phi: 0.5, sigma: 0.06 },
      [{ opponentX: 0, opponentPhi: -0.5, result: 1 }],
      0.5,
    ],
    [
      "a non-binary result",
      { x: 0, phi: 0.5, sigma: 0.06 },
      [{ opponentX: 0, opponentPhi: 0.5, result: 0.5 }],
      0.5,
    ],
  ] as const)("rejects %s with an invalid-input error code", (_name, state, observations, tau) => {
    let caught: unknown;
    try {
      updateGlicko2(state as never, observations as never, tau as never);
    } catch (error) {
      caught = error;
    }

    expect(caught).toMatchObject({ code: "GLICKO2_INVALID_INPUT" });
  });

  it("reports a typed convergence failure when the root solver limit is exhausted", () => {
    const updateWithSolverLimit = updateGlicko2 as unknown as (
      state: { x: number; phi: number; sigma: number },
      observations: readonly { opponentX: number; opponentPhi: number; result: 0 | 1 }[],
      tau: number,
      options: { maxIterations: number }
    ) => unknown;
    let caught: unknown;
    try {
      updateWithSolverLimit(
        { x: 0, phi: 1.1512924985234674, sigma: 0.06 },
        [{ opponentX: -0.5, opponentPhi: 0.5, result: 1 }],
        0.5,
        { maxIterations: 0 }
      );
    } catch (error) {
      caught = error;
    }

    expect(caught).toMatchObject({ code: "GLICKO2_CONVERGENCE_FAILURE" });
  });

  it("does not allow a solver limit above the production cap", () => {
    const updateWithSolverLimit = updateGlicko2 as unknown as (
      state: { x: number; phi: number; sigma: number },
      observations: readonly { opponentX: number; opponentPhi: number; result: 0 | 1 }[],
      tau: number,
      options: { maxIterations: number }
    ) => unknown;
    let caught: unknown;
    try {
      updateWithSolverLimit(
        { x: 0, phi: 0.5, sigma: 0.06 },
        [{ opponentX: 0, opponentPhi: 0.5, result: 1 }],
        0.5,
        { maxIterations: 1001 }
      );
    } catch (error) {
      caught = error;
    }

    expect(caught).toMatchObject({ code: "GLICKO2_INVALID_INPUT" });
  });

  it("handles a zero-phi player with an observation", () => {
    const updated = updateGlicko2(
      { x: 0, phi: 0, sigma: 0.06 },
      [{ opponentX: 0, opponentPhi: 0.5, result: 1 }],
      0.5
    );

    expect(updated.x).toBeGreaterThan(0);
    expect(updated.phi).toBeGreaterThan(0);
    expect(updated.sigma).toBeGreaterThan(0);
  });

  it("reports a typed numerical failure instead of returning zero for an extreme tau", () => {
    let caught: unknown;
    try {
      updateGlicko2(
        { x: 0, phi: 0, sigma: 0.06 },
        [{ opponentX: 0, opponentPhi: 0, result: 1 }],
        1e154
      );
    } catch (error) {
      caught = error;
    }

    expect(caught).toMatchObject({ code: "GLICKO2_CONVERGENCE_FAILURE" });
  });

  it.each([0, 1] as const)("keeps a %s result finite at an extreme logit", (result) => {
    const updated = updateGlicko2(
      { x: Number.MAX_VALUE, phi: 0.5, sigma: 0.06 },
      [{ opponentX: -Number.MAX_VALUE, opponentPhi: 0.5, result }],
      0.5
    );

    expect(Number.isFinite(updated.x)).toBe(true);
    expect(Number.isFinite(updated.phi)).toBe(true);
    expect(Number.isFinite(updated.sigma)).toBe(true);
  });

  it("does not apply a display-scale RD ceiling to standard phi", () => {
    const state = { x: 0, phi: 2, sigma: 0.06 };
    const updated = updateGlicko2(state, [], 0.5);

    expect(updated.phi).toBeCloseTo(Math.hypot(state.phi, state.sigma), 12);
    expect(updated.phi * OFFICIAL_GLICKO2_SCALE).toBeGreaterThan(250);
  });

  it("does not mutate the state or observation array", () => {
    const state = Object.freeze({ x: 0, phi: 0.5, sigma: 0.06 });
    const observations = Object.freeze([
      Object.freeze({ opponentX: 0.25, opponentPhi: 0.5, result: 1 as const }),
    ]);

    updateGlicko2(state, observations, 0.5);

    expect(state).toEqual({ x: 0, phi: 0.5, sigma: 0.06 });
    expect(observations).toEqual([{ opponentX: 0.25, opponentPhi: 0.5, result: 1 }]);
  });

  it("keeps the smallest positive volatility finite with an observation", () => {
    const updated = updateGlicko2(
      { x: 0, phi: 0.5, sigma: Number.MIN_VALUE },
      [{ opponentX: 0, opponentPhi: 0.5, result: 1 }],
      0.5
    );

    expect(Number.isFinite(updated.x)).toBe(true);
    expect(updated.phi).toBeGreaterThan(0);
    expect(updated.sigma).toBe(Number.MIN_VALUE);
  });
});
