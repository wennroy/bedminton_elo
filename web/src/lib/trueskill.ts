export interface TrueSkillPlayer {
  mu: number;
  sigma: number;
}

export interface TrueSkillOutcome {
  win: number;
  draw: number;
  loss: number;
}

export interface TrueSkillSnapshot {
  date: string;
  playerId: string;
  mu: number;
  sigma: number;
}

export const TS_MU = 25.0;
export const TS_SIGMA = 8.333;
export const TS_BETA = TS_SIGMA / 2.0;
export const TS_DRAW_PROBABILITY = 0.0;

export function createPlayer(
  mu: number = TS_MU,
  sigma: number = TS_SIGMA
): TrueSkillPlayer {
  return { mu, sigma };
}

function erf(x: number): number {
  // Abramowitz & Stegun formula 7.1.26, accurate to ~1.5e-7.
  const sign = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + 0.3275911 * ax);
  const y =
    1.0 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
      0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-ax * ax));
  return sign * y;
}

function pdf(x: number): number {
  return Math.exp(-(x * x) / 2.0) / Math.sqrt(2.0 * Math.PI);
}

function cdf(x: number): number {
  return 0.5 * (1.0 + erf(x / Math.sqrt(2.0)));
}

function teamParams(team: TrueSkillPlayer[], beta: number) {
  const mu = team.reduce((sum, p) => sum + p.mu, 0);
  const sigmaSq = team.reduce((sum, p) => sum + p.sigma ** 2, 0) + team.length * beta ** 2;
  return { mu, sigmaSq };
}

export function rateTeam(
  teamA: TrueSkillPlayer[],
  teamB: TrueSkillPlayer[],
  result: 1 | -1,
  beta: number = TS_BETA
): void {
  const a = teamParams(teamA, beta);
  const b = teamParams(teamB, beta);
  const c = Math.sqrt(a.sigmaSq + b.sigmaSq);
  const t = (a.mu - b.mu) / c;

  let v: number;
  let w: number;
  let teamAUpdate: number;
  let teamBUpdate: number;

  if (result === 1) {
    const cdfT = cdf(t);
    v = cdfT > 1e-10 ? pdf(t) / cdfT : 0.0;
    w = v * (v + t);
    teamAUpdate = v;
    teamBUpdate = -v;
  } else {
    const cdfNegT = cdf(-t);
    v = cdfNegT > 1e-10 ? pdf(-t) / cdfNegT : 0.0;
    w = v * (v - t);
    teamAUpdate = -v;
    teamBUpdate = v;
  }

  for (const p of teamA) {
    const factor = (p.sigma ** 2 + beta ** 2) / c;
    p.mu += factor * teamAUpdate;
    p.sigma *= Math.sqrt(
      Math.max(1.0 - ((p.sigma ** 2 + beta ** 2) / c ** 2) * w, 1e-10)
    );
  }

  for (const p of teamB) {
    const factor = (p.sigma ** 2 + beta ** 2) / c;
    p.mu += factor * teamBUpdate;
    p.sigma *= Math.sqrt(
      Math.max(1.0 - ((p.sigma ** 2 + beta ** 2) / c ** 2) * w, 1e-10)
    );
  }
}

export function predictTeamOutcome(
  teamA: TrueSkillPlayer[],
  teamB: TrueSkillPlayer[],
  drawProbability: number = TS_DRAW_PROBABILITY,
  beta: number = TS_BETA
): TrueSkillOutcome {
  const a = teamParams(teamA, beta);
  const b = teamParams(teamB, beta);
  const deltaMu = a.mu - b.mu;
  const deltaSigmaSq = a.sigmaSq + b.sigmaSq;
  const deltaSigma = Math.sqrt(deltaSigmaSq);

  const drawMargin = 0.0;
  void drawProbability;

  const pWin = cdf((deltaMu - drawMargin) / deltaSigma);
  const pLoss = 1.0 - cdf((deltaMu + drawMargin) / deltaSigma);
  const pDraw = Math.max(0.0, 1.0 - pWin - pLoss);

  return { win: pWin, draw: pDraw, loss: pLoss };
}

export function predictTeamOutcomeWin(
  teamA: TrueSkillPlayer[],
  teamB: TrueSkillPlayer[],
  drawProbability: number = TS_DRAW_PROBABILITY,
  beta: number = TS_BETA
): number {
  return predictTeamOutcome(teamA, teamB, drawProbability, beta).win;
}

export function recomputeTrueSkills(matches: import("./elo").Match[]): {
  players: Record<string, TrueSkillPlayer>;
  snapshots: TrueSkillSnapshot[];
} {
  const players: Record<string, TrueSkillPlayer> = {};
  const snapshots: TrueSkillSnapshot[] = [];
  let lastDate: string | null = null;

  function recordDayBoundary(date: string | null) {
    if (date === null) return;
    for (const [playerId, player] of Object.entries(players)) {
      snapshots.push({
        date,
        playerId,
        mu: player.mu,
        sigma: player.sigma,
      });
    }
  }

  for (const match of matches) {
    if (lastDate === null || lastDate !== match.date) {
      recordDayBoundary(lastDate);
      lastDate = match.date;
    }

    for (const playerId of [match.a1, match.a2, match.b1, match.b2]) {
      if (players[playerId] === undefined) {
        players[playerId] = createPlayer();
      }
    }

    const teamA = [players[match.a1], players[match.a2]];
    const teamB = [players[match.b1], players[match.b2]];
    const result: 1 | -1 = match.scoreA > match.scoreB ? 1 : -1;
    rateTeam(teamA, teamB, result);
  }

  recordDayBoundary(lastDate);

  return { players, snapshots };
}
