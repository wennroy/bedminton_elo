import type { ScheduledMatch } from "./types";
import type { PRNG } from "./random";

/**
 * 公平配额模块：最优配额分配、构造场次、等价转移、独立校验。
 *
 * 硬条件：0<=x_i<=M、sum x_i=4M、每场四人互异且来自当前名单。
 * 目标：最小化 F = sum_i (c_i + x_i)^2（c_i = effectiveAppearances）。
 *
 * 口径说明：模块只看 effectiveAppearances，不理解 fairnessBase/playedAtJoin
 * ——「新人从加入时公平」由存储层折算成 effective 输入后驱动本模块。
 */

export interface QuotaResult {
  /** 每人本批新增场数 x_i */
  quotas: Record<string, number>;
  /** 最优公平目标值 F* = sum_i (c_i + x_i)^2 */
  F: number;
}

function pickRandom<T>(candidates: T[], rng: PRNG): T {
  if (candidates.length === 0) {
    throw new Error("内部错误：空候选集无法随机选取");
  }
  return candidates[Math.floor(rng() * candidates.length)];
}

/**
 * 逐名额贪心：每次把下一个名额给「尚未达上限 M 且边际代价
 * 2(c_i+x_i)+1 最小」的人；同代价用 rng 打破。
 *
 * 代价函数凸且可分离，贪心按最小边际代价分配即全局最优。
 * 同起点无历史时配额退化为 floor(4M/N)/ceil(4M/N)；
 * 历史差异无法靠有限场数消除时不强行拉平（由目标函数自然体现）。
 */
export function computeQuotas(input: {
  playerIds: string[];
  appearances: Record<string, number>;
  matches: number;
  rng: PRNG;
}): QuotaResult {
  const { playerIds, appearances, matches: M, rng } = input;
  const n = playerIds.length;
  if (n < 4) throw new Error(`computeQuotas: 至少需要 4 名选手，实际 ${n}`);
  if (!Number.isSafeInteger(M) || M < 1) {
    throw new Error(`computeQuotas: matches 必须是正整数，实际 ${M}`);
  }

  const quotas: Record<string, number> = {};
  for (const id of playerIds) {
    const c = appearances[id];
    if (!Number.isSafeInteger(c) || c < 0) {
      throw new Error(`computeQuotas: appearances['${id}'] 必须是非负整数，实际 ${c}`);
    }
    quotas[id] = 0;
  }

  const slots = 4 * M;
  for (let s = 0; s < slots; s++) {
    let bestCost = Infinity;
    let best: string[] = [];
    for (const id of playerIds) {
      if (quotas[id] >= M) continue;
      const cost = 2 * (appearances[id] + quotas[id]) + 1;
      if (cost < bestCost) {
        bestCost = cost;
        best = [id];
      } else if (cost === bestCost) {
        best.push(id);
      }
    }
    if (best.length === 0) {
      throw new Error("computeQuotas: 配额上限导致名额无法分完（名单不足 4 人）");
    }
    quotas[pickRandom(best, rng)]++;
  }

  let F = 0;
  for (const id of playerIds) {
    F += (appearances[id] + quotas[id]) ** 2;
  }
  return { quotas, F };
}

/**
 * 按剩余配额从高到低取四个不同人组成一场（同配额 rng 打破平手），
 * 减配额后重复 M 次。队内/场内分配确定性：前两人为 A 队，后两人为 B 队；
 * 具体分队在 T2 搜索中还会调整。
 *
 * 可行性：每轮总有人可取——不变式「任何人剩余配额 <= 剩余场数」成立
 * （若某轮有 5 人剩余等于场数 k，则总和 >= 5k > 4k，矛盾），
 * 故选满四人时不会提前耗尽。
 */
export function buildFairFixtures(input: {
  playerIds: string[];
  quotas: Record<string, number>;
  matches: number;
  rng: PRNG;
}): ScheduledMatch[] {
  const { playerIds, quotas, matches: M, rng } = input;
  const remaining: Record<string, number> = {};
  let totalRemaining = 0;
  for (const id of playerIds) {
    const q = quotas[id];
    if (!Number.isSafeInteger(q) || q < 0) {
      throw new Error(`buildFairFixtures: quotas['${id}'] 必须是非负整数，实际 ${q}`);
    }
    remaining[id] = q;
    totalRemaining += q;
  }
  if (totalRemaining !== 4 * M) {
    throw new Error(
      `buildFairFixtures: 配额总和 ${totalRemaining} != 4*M ${4 * M}`
    );
  }

  const fixtures: ScheduledMatch[] = [];
  for (let m = 0; m < M; m++) {
    const chosen: string[] = [];
    for (let k = 0; k < 4; k++) {
      let bestVal = -Infinity;
      let candidates: string[] = [];
      for (const id of playerIds) {
        if (chosen.includes(id)) continue;
        const r = remaining[id];
        if (r > bestVal) {
          bestVal = r;
          candidates = [id];
        } else if (r === bestVal) {
          candidates.push(id);
        }
      }
      if (bestVal <= 0) {
        throw new Error(
          `buildFairFixtures: 第 ${m + 1} 场无法凑满四人（剩余配额不足）`
        );
      }
      const pick = pickRandom(candidates, rng);
      chosen.push(pick);
      remaining[pick]--;
    }
    fixtures.push({ a1: chosen[0], a2: chosen[1], b1: chosen[2], b2: chosen[3] });
  }
  return fixtures;
}

export interface TransferOutcome {
  quotas: Record<string, number>;
  fixtures: ScheduledMatch[];
  /** 是否成功转移（无可行转移时为 false，输入原样返回副本） */
  transferred: boolean;
  from?: string;
  to?: string;
  matchIndex?: number;
}

/**
 * 等价转移：把 donor 的一个名额转给 receiver。可行条件：
 * - donor 最终 effective 恰比 receiver 高 1（F 不变：2(c_j+x_j+1)-2(c_i+x_i)+...=0）
 * - donor 配额 >=1、receiver 配额 < M（不超上限）
 * - 存在一场含 donor 但不含 receiver（同场不重复）
 *
 * 在所有可行 (donor, receiver, matchIndex) 中 rng 均匀选取一个；
 * 无可行转移时 transferred=false。单次只转一个名额，调用方可循环。
 */
export function transferQuota(input: {
  playerIds: string[];
  appearances: Record<string, number>;
  quotas: Record<string, number>;
  fixtures: ScheduledMatch[];
  matches: number;
  rng: PRNG;
}): TransferOutcome {
  const { playerIds, appearances, quotas, fixtures, matches: M, rng } = input;

  interface Triple {
    from: string;
    to: string;
    matchIndex: number;
  }
  const feasible: Triple[] = [];

  for (const donor of playerIds) {
    if ((quotas[donor] ?? 0) < 1) continue;
    const donorEff = appearances[donor] + quotas[donor];
    for (const receiver of playerIds) {
      if (receiver === donor) continue;
      if ((quotas[receiver] ?? 0) >= M) continue;
      const receiverEff = appearances[receiver] + quotas[receiver];
      if (donorEff !== receiverEff + 1) continue;
      for (let mi = 0; mi < fixtures.length; mi++) {
        const f = fixtures[mi];
        const four = [f.a1, f.a2, f.b1, f.b2];
        if (four.includes(donor) && !four.includes(receiver)) {
          feasible.push({ from: donor, to: receiver, matchIndex: mi });
        }
      }
    }
  }

  if (feasible.length === 0) {
    return {
      quotas: { ...quotas },
      fixtures: fixtures.map((f) => ({ ...f })),
      transferred: false,
    };
  }

  const pick = pickRandom(feasible, rng);
  const newQuotas = { ...quotas };
  newQuotas[pick.from]--;
  newQuotas[pick.to]++;
  const newFixtures = fixtures.map((f) => ({ ...f }));
  const f = newFixtures[pick.matchIndex];
  for (const key of ["a1", "a2", "b1", "b2"] as const) {
    if (f[key] === pick.from) {
      f[key] = pick.to;
      break;
    }
  }

  return {
    quotas: newQuotas,
    fixtures: newFixtures,
    transferred: true,
    from: pick.from,
    to: pick.to,
    matchIndex: pick.matchIndex,
  };
}

/**
 * 最终独立校验，不满足抛错：
 * 场数=M、每场四人互异且来自名单、各人实际新增 == quotas、F == expectedF。
 */
export function validateFairness(input: {
  playerIds: string[];
  appearances: Record<string, number>;
  matches: number;
  quotas: Record<string, number>;
  fixtures: ScheduledMatch[];
  expectedF: number;
}): void {
  const { playerIds, appearances, matches: M, quotas, fixtures, expectedF } = input;
  const roster = new Set(playerIds);

  if (fixtures.length !== M) {
    throw new Error(
      `validateFairness: 场数 ${fixtures.length} != M ${M}`
    );
  }

  const actual: Record<string, number> = {};
  for (const id of playerIds) actual[id] = 0;

  for (let i = 0; i < fixtures.length; i++) {
    const f = fixtures[i];
    const four = [f.a1, f.a2, f.b1, f.b2];
    if (new Set(four).size !== 4) {
      throw new Error(`validateFairness: 第 ${i + 1} 场四人不互异`);
    }
    for (const id of four) {
      if (!roster.has(id)) {
        throw new Error(`validateFairness: 第 ${i + 1} 场出现名单外选手 '${id}'`);
      }
      actual[id]++;
    }
  }

  for (const id of playerIds) {
    if (actual[id] !== quotas[id]) {
      throw new Error(
        `validateFairness: '${id}' 实际新增 ${actual[id]} != 配额 ${quotas[id]}`
      );
    }
  }

  let F = 0;
  for (const id of playerIds) {
    F += (appearances[id] + quotas[id]) ** 2;
  }
  if (F !== expectedF) {
    throw new Error(`validateFairness: F ${F} != F* ${expectedF}`);
  }
}
