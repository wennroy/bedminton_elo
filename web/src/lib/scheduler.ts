import { predictTeamOutcomeWin, type TrueSkillPlayer } from "./trueskill";

export interface ScheduledMatch {
  a1: string;
  a2: string;
  b1: string;
  b2: string;
}

export interface SchedulerMetrics {
  alphaVar: number;
  bestLoss: number;
  meanCloseness: number;
  maxCloseness: number;
  entropy: number;
}

export interface SchedulerResult {
  schedule: ScheduledMatch[];
  metrics: SchedulerMetrics;
}

export interface SchedulerOptions {
  playerIds: string[];
  matches: number;
  players: TrueSkillPlayer[];
  seed: number;
  lambda: number;
  iters?: number;
  startTemp?: number;
  alphaDecay?: number;
  floor?: number;
}

export type PRNG = () => number;

export function mulberry32(seed: number): PRNG {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >> 15), t | 1);
    r ^= r + Math.imul(r ^ (r >> 7), r | 61);
    return ((r ^ (r >> 14)) >> 0) / 4294967296;
  };
}

function chaosByEntropy(lst: string[]): number {
  const n = lst.length;
  if (n <= 1) return 1.0;
  const freqs = new Map<string, number>();
  for (const item of lst) {
    freqs.set(item, (freqs.get(item) ?? 0) + 1);
  }
  let h = 0;
  for (const cnt of freqs.values()) {
    const p = cnt / n;
    h -= p * Math.log(p);
  }
  return 1 - h / Math.log(n);
}

function normalizeMatch(match: ScheduledMatch): ScheduledMatch {
  const t1 = [match.a1, match.a2].sort();
  const t2 = [match.b1, match.b2].sort();
  if (t1.join(",") <= t2.join(",")) {
    return { a1: t1[0], a2: t1[1], b1: t2[0], b2: t2[1] };
  }
  return { a1: t2[0], a2: t2[1], b1: t1[0], b2: t1[1] };
}

function normalizeSchedule(schedule: ScheduledMatch[]): ScheduledMatch[] {
  return schedule.map(normalizeMatch);
}

function randomInitialSchedule(
  playerIds: string[],
  m: number,
  rng: PRNG
): ScheduledMatch[] {
  const schedule: ScheduledMatch[] = [];
  for (let i = 0; i < m; i++) {
    const team = sample(playerIds, 4, rng);
    const team1 = sample(team, 2, rng);
    const team2 = team.filter((p) => !team1.includes(p));
    schedule.push({
      a1: team1[0],
      a2: team1[1],
      b1: team2[0],
      b2: team2[1],
    });
  }
  return normalizeSchedule(schedule);
}

function sample<T>(arr: T[], k: number, rng: PRNG): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, k);
}

function choice<T>(arr: T[], rng: PRNG): T {
  return arr[Math.floor(rng() * arr.length)];
}

function computeLoss(
  schedule: ScheduledMatch[],
  playerIds: string[],
  players: TrueSkillPlayer[],
  lambdaWeight: number
): {
  totalLoss: number;
  alphaVar: number;
  closenessSum: number;
  closenessList: number[];
  entropy: number;
} {
  const n = playerIds.length;
  const counts = new Map<string, number>();
  for (const id of playerIds) {
    counts.set(id, 0);
  }
  const closenessList: number[] = [];
  const scheduleTeam: string[] = [];
  const matchTuples: string[] = [];

  for (const match of schedule) {
    const team1 = [match.a1, match.a2];
    const team2 = [match.b1, match.b2];
    scheduleTeam.push([...team1].sort().join(","));
    scheduleTeam.push([...team2].sort().join(","));
    matchTuples.push(`(${[...team1].sort().join(",")})vs(${[...team2].sort().join(",")})`);

    for (const id of team1.concat(team2)) {
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }

    const p = predictTeamOutcomeWin(
      team1.map((id) => players[playerIds.indexOf(id)]),
      team2.map((id) => players[playerIds.indexOf(id)])
    );
    const closeness = Math.abs(p - 0.5);
    closenessList.push(closeness);
  }

  const countValues = Array.from(counts.values());
  const meanCount = countValues.reduce((a, b) => a + b, 0) / n;
  const alphaVar =
    countValues.reduce((sum, c) => sum + (c - meanCount) ** 2, 0) / n;

  const entropy =
    chaosByEntropy(scheduleTeam) + chaosByEntropy(matchTuples);

  const closenessSum = closenessList.reduce((a, b) => a + b, 0);

  const totalLoss =
    alphaVar +
    lambdaWeight * entropy +
    (1 - lambdaWeight) * (closenessSum / closenessList.length) * 2;

  return { totalLoss, alphaVar, closenessSum, closenessList, entropy };
}

function getNeighbor(
  schedule: ScheduledMatch[],
  playerIds: string[],
  rng: PRNG
): ScheduledMatch[] {
  const newSchedule = schedule.map((m) => ({ ...m }));
  const m = newSchedule.length;
  const operation = choice(["swap_match", "swap_player", "reshuffle_team"], rng);

  if (operation === "swap_match" && m >= 2) {
    const [i, j] = sample(Array.from({ length: m }, (_, idx) => idx), 2, rng);
    const mi = newSchedule[i];
    const mj = newSchedule[j];
    const allI = [mi.a1, mi.a2, mi.b1, mi.b2];
    const allJ = [mj.a1, mj.a2, mj.b1, mj.b2];
    const pa = choice(allI, rng);
    const pb = choice(allJ, rng);

    const replace = (team: string[], oldId: string, newId: string) =>
      team.map((x) => (x === oldId ? newId : x));

    const t1a = replace([mi.a1, mi.a2], pa, pb);
    const t2a = replace([mi.b1, mi.b2], pa, pb);
    const t1b = replace([mj.a1, mj.a2], pb, pa);
    const t2b = replace([mj.b1, mj.b2], pb, pa);

    if (
      new Set([...t1a, ...t2a]).size === 4 &&
      new Set([...t1b, ...t2b]).size === 4
    ) {
      newSchedule[i] = { a1: t1a[0], a2: t1a[1], b1: t2a[0], b2: t2a[1] };
      newSchedule[j] = { a1: t1b[0], a2: t1b[1], b1: t2b[0], b2: t2b[1] };
    }
  } else if (operation === "swap_player") {
    const i = Math.floor(rng() * m);
    const match = newSchedule[i];
    const allInMatch = [match.a1, match.a2, match.b1, match.b2];
    const outside = playerIds.filter((p) => !allInMatch.includes(p));
    if (outside.length > 0) {
      if (rng() < 0.5) {
        const old = choice([match.a1, match.a2], rng);
        const newId = choice(outside, rng);
        const team1 = [match.a1, match.a2].map((x) => (x === old ? newId : x));
        match.a1 = team1[0];
        match.a2 = team1[1];
      } else {
        const old = choice([match.b1, match.b2], rng);
        const newId = choice(outside, rng);
        const team2 = [match.b1, match.b2].map((x) => (x === old ? newId : x));
        match.b1 = team2[0];
        match.b2 = team2[1];
      }
    }
  } else if (operation === "reshuffle_team") {
    const i = Math.floor(rng() * m);
    const match = newSchedule[i];
    const all4 = [match.a1, match.a2, match.b1, match.b2];
    const newTeam1 = sample(all4, 2, rng);
    const newTeam2 = all4.filter((p) => !newTeam1.includes(p));
    match.a1 = newTeam1[0];
    match.a2 = newTeam1[1];
    match.b1 = newTeam2[0];
    match.b2 = newTeam2[1];
  }

  return normalizeSchedule(newSchedule);
}

export function optimizeSchedule(options: SchedulerOptions): SchedulerResult {
  const {
    playerIds,
    matches: m,
    players,
    seed,
    lambda: lambdaWeight,
    iters = 5000,
    startTemp = 1.0,
    alphaDecay = 0.995,
    floor = 1e-4,
  } = options;

  const rng = mulberry32(seed);
  let bestSchedule = randomInitialSchedule(playerIds, m, rng);
  let best = computeLoss(bestSchedule, playerIds, players, lambdaWeight);
  let currentSchedule = bestSchedule;
  let currentLoss = best.totalLoss;
  let T = startTemp;

  for (let k = 0; k < iters; k++) {
    const neighbor = getNeighbor(currentSchedule, playerIds, rng);
    const neighborLoss = computeLoss(
      neighbor,
      playerIds,
      players,
      lambdaWeight
    );

    if (
      neighborLoss.totalLoss < currentLoss ||
      rng() < Math.exp((currentLoss - neighborLoss.totalLoss) / T)
    ) {
      currentSchedule = neighbor;
      currentLoss = neighborLoss.totalLoss;

      if (neighborLoss.totalLoss < best.totalLoss) {
        bestSchedule = neighbor;
        best = neighborLoss;
      }
    }

    T *= alphaDecay;
    if (T < floor) break;
  }

  const meanCloseness =
    best.closenessList.reduce((a, b) => a + b, 0) / best.closenessList.length;
  const maxCloseness = Math.max(...best.closenessList);

  return {
    schedule: bestSchedule,
    metrics: {
      alphaVar: best.alphaVar,
      bestLoss: best.totalLoss,
      meanCloseness,
      maxCloseness,
      entropy: best.entropy,
    },
  };
}
