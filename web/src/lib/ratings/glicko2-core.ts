const ROOT_TOLERANCE = 0.000001;
const MAX_SOLVER_ITERATIONS = 1000;

export interface Glicko2State {
  readonly x: number;
  readonly phi: number;
  readonly sigma: number;
}

export interface Glicko2Observation {
  readonly opponentX: number;
  readonly opponentPhi: number;
  readonly result: 0 | 1;
}

/** Limits both volatility-solver phases for deterministic tests. Omit in production. */
export interface Glicko2UpdateOptions {
  readonly maxIterations?: number;
}

export type Glicko2ErrorCode = "GLICKO2_INVALID_INPUT" | "GLICKO2_CONVERGENCE_FAILURE";

export class Glicko2InputError extends Error {
  readonly code = "GLICKO2_INVALID_INPUT" as const;

  constructor(message: string) {
    super(message);
    this.name = "Glicko2InputError";
  }
}

export class Glicko2ConvergenceError extends Error {
  readonly code = "GLICKO2_CONVERGENCE_FAILURE" as const;

  constructor(message: string) {
    super(message);
    this.name = "Glicko2ConvergenceError";
  }
}

/** Updates a standard Glicko-2 state in its native x/phi/sigma coordinates. */
export function updateGlicko2(
  state: Glicko2State,
  observations: readonly Glicko2Observation[],
  tau: number,
  options: Glicko2UpdateOptions = {}
): Glicko2State {
  validateState(state);
  validateObservations(observations);
  assertFinitePositive(tau, "tau");
  const maxIterations = validateMaxIterations(options);

  if (observations.length === 0) {
    return assertFiniteUpdatedState(expandVariance(state));
  }

  let information = 0;
  let scoreDifference = 0;
  for (const observation of observations) {
    const g = glickoG(observation.opponentPhi);
    const probability = logistic(glickoLogit(g, state.x, observation.opponentX));
    information += g ** 2 * probability * (1 - probability);
    scoreDifference += g * (observation.result - probability);
  }

  if (information === 0) {
    return assertFiniteUpdatedState(expandVariance(state));
  }

  const variance = 1 / information;
  const delta = variance * scoreDifference;
  if (!Number.isFinite(variance) || !Number.isFinite(delta)) {
    return assertFiniteUpdatedState(expandVariance(state));
  }
  const sigma = solveVolatility(state.phi, state.sigma, variance, delta, tau, maxIterations);
  const phiStar = Math.hypot(state.phi, sigma);
  const phi = 1 / Math.sqrt(1 / phiStar ** 2 + 1 / variance);
  return assertFiniteUpdatedState({ x: state.x + phi ** 2 * scoreDifference, phi, sigma });
}

function expandVariance(state: Glicko2State): Glicko2State {
  return { x: state.x, phi: Math.hypot(state.phi, state.sigma), sigma: state.sigma };
}

function assertFiniteUpdatedState(state: Glicko2State): Glicko2State {
  if (
    !Number.isFinite(state.x) ||
    !Number.isFinite(state.phi) ||
    state.phi <= 0 ||
    !Number.isFinite(state.sigma) ||
    state.sigma <= 0
  ) {
    throw new Glicko2ConvergenceError("Glicko-2 update could not produce a finite state");
  }
  return state;
}

function glickoG(phi: number): number {
  return 1 / Math.hypot(1, (Math.sqrt(3) / Math.PI) * phi);
}

function glickoLogit(g: number, x: number, opponentX: number): number {
  return g === 0 ? 0 : g * (x - opponentX);
}

function validateState(state: Glicko2State): void {
  if (typeof state !== "object" || state === null) {
    throw new Glicko2InputError("state must be an object");
  }
  assertFiniteNumber(state.x, "state.x");
  assertFiniteNonNegative(state.phi, "state.phi");
  assertFinitePositive(state.sigma, "state.sigma");
}

function validateObservations(observations: readonly Glicko2Observation[]): void {
  if (!Array.isArray(observations)) {
    throw new Glicko2InputError("observations must be an array");
  }
  for (const [index, observation] of observations.entries()) {
    if (typeof observation !== "object" || observation === null) {
      throw new Glicko2InputError(`observations[${index}] must be an object`);
    }
    assertFiniteNumber(observation.opponentX, `observations[${index}].opponentX`);
    assertFiniteNonNegative(observation.opponentPhi, `observations[${index}].opponentPhi`);
    if (observation.result !== 0 && observation.result !== 1) {
      throw new Glicko2InputError(`observations[${index}].result must be 0 or 1`);
    }
  }
}

function validateMaxIterations(options: Glicko2UpdateOptions): number {
  if (typeof options !== "object" || options === null) {
    throw new Glicko2InputError("options must be an object");
  }
  const maxIterations = options.maxIterations ?? MAX_SOLVER_ITERATIONS;
  if (!Number.isInteger(maxIterations) || maxIterations < 0 || maxIterations > MAX_SOLVER_ITERATIONS) {
    throw new Glicko2InputError(`maxIterations must be an integer from 0 to ${MAX_SOLVER_ITERATIONS}`);
  }
  return maxIterations;
}

function assertFiniteNumber(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Glicko2InputError(`${name} must be a finite number`);
  }
}

function assertFiniteNonNegative(value: unknown, name: string): asserts value is number {
  assertFiniteNumber(value, name);
  if (value < 0) {
    throw new Glicko2InputError(`${name} must be greater than or equal to zero`);
  }
}

function assertFinitePositive(value: unknown, name: string): asserts value is number {
  assertFiniteNumber(value, name);
  if (value <= 0) {
    throw new Glicko2InputError(`${name} must be greater than zero`);
  }
}

function logistic(value: number): number {
  if (value >= 0) {
    return 1 / (1 + Math.exp(-value));
  }
  const exponent = Math.exp(value);
  return exponent / (1 + exponent);
}

function solveVolatility(
  phi: number,
  sigma: number,
  variance: number,
  delta: number,
  tau: number,
  maxIterations: number
): number {
  const a = 2 * Math.log(sigma);
  const phiSquared = phi ** 2;
  const deltaSquared = delta ** 2;
  const f = (value: number) => {
    const exponent = Math.exp(value);
    const denominator = phiSquared + variance + exponent;
    return (
      (exponent * (deltaSquared - phiSquared - variance - exponent)) /
        (2 * denominator ** 2) -
      ((value - a) / tau) / tau
    );
  };

  let lower = a;
  let upper: number;
  if (deltaSquared > phiSquared + variance) {
    upper = Math.log(deltaSquared - phiSquared - variance);
  } else {
    let k = 1;
    upper = a - k * tau;
    let bracketIterations = 0;
    while (evaluateVolatilityFunction(f, upper) < 0) {
      if (bracketIterations >= maxIterations) {
        throw new Glicko2ConvergenceError("Glicko-2 volatility bracket did not converge");
      }
      k += 1;
      upper = a - k * tau;
      bracketIterations += 1;
    }
  }

  let fLower = evaluateVolatilityFunction(f, lower);
  let fUpper = evaluateVolatilityFunction(f, upper);
  let rootIterations = 0;
  while (Math.abs(upper - lower) > ROOT_TOLERANCE) {
    if (rootIterations >= maxIterations) {
      throw new Glicko2ConvergenceError("Glicko-2 volatility root did not converge");
    }
    const candidate = lower + ((lower - upper) * fLower) / (fUpper - fLower);
    if (!Number.isFinite(candidate)) {
      throw new Glicko2ConvergenceError("Glicko-2 volatility root became non-finite");
    }
    const fCandidate = evaluateVolatilityFunction(f, candidate);
    if (fCandidate * fUpper <= 0) {
      lower = upper;
      fLower = fUpper;
    } else {
      fLower /= 2;
    }
    upper = candidate;
    fUpper = fCandidate;
    rootIterations += 1;
  }
  const solvedSigma = Math.exp(lower / 2);
  if (!Number.isFinite(solvedSigma) || solvedSigma <= 0) {
    throw new Glicko2ConvergenceError("Glicko-2 volatility root produced an invalid sigma");
  }
  return solvedSigma;
}

function evaluateVolatilityFunction(
  f: (value: number) => number,
  value: number
): number {
  const result = f(value);
  if (!Number.isFinite(result)) {
    throw new Glicko2ConvergenceError("Glicko-2 volatility solver became non-finite");
  }
  return result;
}
