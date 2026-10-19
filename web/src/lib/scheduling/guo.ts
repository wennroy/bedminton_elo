import { createRng } from "./random";
import type { ScheduledMatch } from "./types";

/** Browser-safe tournament contract. IDs always use the scheduling API's string form. */
export interface GuoStanding {
  playerId: string;
  played: number;
  wins: number;
  pointsFor: number;
  pointsAgainst: number;
  netPoints: number;
  expectedWins: number;
  overperformance: number;
  drawOrder: number;
}
export type GuoRoundKind = "initial" | "promotion" | "cross";
export interface GuoRound {
  roundNo: number;
  generationId: number;
  kind: GuoRoundKind;
  groupA: string[];
  groupB: string[];
  standingsA: GuoStanding[];
  standingsB: GuoStanding[];
  complete: boolean;
}
export interface GuoProjectState {
  rounds: GuoRound[];
  maxRounds: 4;
  canAdvance: boolean;
  nextRoundKind: "promotion" | "cross" | null;
}
export interface GuoSnapshot {
  mode: "guo";
  roundNo: number;
  kind: GuoRoundKind;
  groupA: string[];
  groupB: string[];
  drawOrder: Record<string, number>;
  seed: number;
  /** Ranking used to create this round; remains fixed if earlier scores change. */
  sourceStandings: { groupA: GuoStanding[]; groupB: GuoStanding[] } | null;
  /** Exact generated teams, retained as historical input. */
  schedule: ScheduledMatch[];
}

function assertRoster(ids: string[]): void {
  if (ids.length !== 8 || new Set(ids).size !== 8) throw new Error("郭老师分配方法仅支持 8 名不同球员");
}
export function createGuoDraw(ids: string[], seed: number): Record<string, number> {
  assertRoster(ids);
  const shuffled = [...ids].sort((a, b) => Number(a) - Number(b));
  const rng = createRng(seed);
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return Object.fromEntries(shuffled.map((id, i) => [id, i]));
}
export function createGuoGroups(ids: string[], ratings: Record<string, number>, drawOrder: Record<string, number>): { groupA: string[]; groupB: string[] } {
  assertRoster(ids);
  const sorted = [...ids].sort((a, b) => (ratings[b] - ratings[a]) || drawOrder[a] - drawOrder[b]);
  return { groupA: sorted.slice(0, 4), groupB: sorted.slice(4) };
}
export function createGuoSchedule(groupA: string[], groupB: string[]): ScheduledMatch[] {
  assertRoster([...groupA, ...groupB]);
  if (groupA.length !== 4 || groupB.length !== 4) throw new Error("每组必须有 4 人");
  const pairings = (p: string[]): ScheduledMatch[] => [
    { a1: p[0], a2: p[1], b1: p[2], b2: p[3] },
    { a1: p[0], a2: p[2], b1: p[1], b2: p[3] },
    { a1: p[0], a2: p[3], b1: p[1], b2: p[2] },
  ];
  const a = pairings(groupA), b = pairings(groupB);
  // Two courts can play each group's corresponding fixture simultaneously.
  return a.flatMap((match, i) => [match, b[i]]);
}
export function rankGuoGroup(rows: GuoStanding[]): GuoStanding[] {
  return [...rows].sort((a, b) => b.wins - a.wins || b.netPoints - a.netPoints ||
    // Ignore sub-ulp summation noise when symmetric forecasts tie mathematically.
    (Math.abs(b.overperformance - a.overperformance) > 1e-12 ? b.overperformance - a.overperformance : 0) || a.drawOrder - b.drawOrder);
}
export function nextGuoGroups(a: GuoStanding[], b: GuoStanding[], kind: "promotion" | "cross"): { groupA: string[]; groupB: string[] } {
  const ids = (r: GuoStanding[]) => r.map((row) => row.playerId);
  return kind === "cross"
    ? { groupA: ids([...a.slice(0, 2), ...b.slice(2)]), groupB: ids([...a.slice(2), ...b.slice(0, 2)]) }
    : { groupA: ids([...a.slice(0, 2), ...b.slice(0, 2)]), groupB: ids([...a.slice(2), ...b.slice(2)]) };
}
export function isGuoSnapshot(input: unknown): input is GuoSnapshot {
  return input !== null && typeof input === "object" && (input as { mode?: unknown }).mode === "guo";
}

interface GuoFixture {
  generationId: number;
  pa1: string; pa2: string; pb1: string; pb2: string;
  status: string;
  scoreA: number | null; scoreB: number | null;
  generatedWinRate: number | null;
}
export function deriveGuoState(generations: { id: number; input: unknown }[], fixtures: GuoFixture[]): GuoProjectState | undefined {
  const snapshots = generations.filter((g): g is { id: number; input: GuoSnapshot } => isGuoSnapshot(g.input));
  if (snapshots.length === 0) return undefined;
  const rounds = snapshots.map(({ id, input }): GuoRound => {
    const matches = fixtures.filter((f) => f.generationId === id);
    const rows = new Map([...input.groupA, ...input.groupB].map((playerId) => [playerId, {
      playerId, played: 0, wins: 0, pointsFor: 0, pointsAgainst: 0,
      netPoints: 0, expectedWins: 0, overperformance: 0, drawOrder: input.drawOrder[playerId],
    }]));
    for (const f of matches) {
      if (f.status !== "recorded" || f.scoreA === null || f.scoreB === null) continue;
      for (const [players, own, other, expected] of [
        [[f.pa1, f.pa2], f.scoreA, f.scoreB, f.generatedWinRate],
        [[f.pb1, f.pb2], f.scoreB, f.scoreA, f.generatedWinRate === null ? null : 1 - f.generatedWinRate],
      ] as const) {
        for (const playerId of players) {
          const row = rows.get(playerId);
          if (!row) continue;
          row.played++; row.wins += own > other ? 1 : 0;
          row.pointsFor += own; row.pointsAgainst += other;
          row.netPoints = row.pointsFor - row.pointsAgainst;
          row.expectedWins += expected ?? 0.5;
          row.overperformance = row.wins - row.expectedWins;
        }
      }
    }
    return { roundNo: input.roundNo, generationId: id, kind: input.kind,
      groupA: input.groupA, groupB: input.groupB,
      standingsA: rankGuoGroup(input.groupA.map((p) => rows.get(p)!)),
      standingsB: rankGuoGroup(input.groupB.map((p) => rows.get(p)!)),
      complete: matches.length === 6 && matches.every((f) => f.status === "recorded" && f.scoreA !== null && f.scoreB !== null),
    };
  });
  const last = rounds[rounds.length - 1];
  return { rounds, maxRounds: 4, canAdvance: last.complete && last.roundNo < 4,
    nextRoundKind: last.roundNo >= 4 ? null : last.roundNo === 2 ? "cross" : "promotion" };
}
