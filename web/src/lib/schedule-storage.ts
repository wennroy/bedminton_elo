/**
 * 配对结果的按身份持久化:和「我是谁」同一套 localStorage 机制,
 * key 按当前身份区分;未选身份时共用 anon。只有用户点「清空」
 * 或再次「生成配对」才会被覆盖。
 */

export interface ScheduledMatch {
  a1: string;
  a2: string;
  b1: string;
  b2: string;
  winRate: number;
}

export interface ScheduleResult {
  schedule: ScheduledMatch[];
  metrics: {
    alphaVar: number;
    bestLoss: number;
    meanCloseness: number;
    maxCloseness: number;
    entropy: number;
  };
  names: Record<string, string>;
}

export interface StoredSchedule {
  playerIds: number[];
  matches: number;
  seed: number;
  lambda: number;
  result: ScheduleResult;
  savedAt: string;
}

const KEY_PREFIX = "badminton:schedule:";

function storageKey(userId: number | null): string {
  return `${KEY_PREFIX}${userId ?? "anon"}`;
}

/** 纯校验函数:坏数据一律返回 null,绝不抛异常。 */
export function parseStoredSchedule(raw: string): StoredSchedule | null {
  try {
    const data = JSON.parse(raw);
    if (!data || typeof data !== "object") return null;
    if (
      !Array.isArray(data.playerIds) ||
      !data.playerIds.every((x: unknown) => typeof x === "number")
    ) {
      return null;
    }
    if (
      typeof data.matches !== "number" ||
      typeof data.seed !== "number" ||
      typeof data.lambda !== "number" ||
      typeof data.savedAt !== "string"
    ) {
      return null;
    }
    const r = data.result;
    if (!r || typeof r !== "object") return null;
    if (
      !Array.isArray(r.schedule) ||
      !r.schedule.every(
        (m: unknown) =>
          m !== null &&
          typeof m === "object" &&
          ["a1", "a2", "b1", "b2"].every(
            (k) => typeof (m as Record<string, unknown>)[k] === "string"
          ) &&
          typeof (m as Record<string, unknown>).winRate === "number"
      )
    ) {
      return null;
    }
    if (!r.metrics || typeof r.metrics !== "object") return null;
    if (!r.names || typeof r.names !== "object") return null;
    return data as StoredSchedule;
  } catch {
    return null;
  }
}

export function loadSchedule(userId: number | null): StoredSchedule | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(storageKey(userId));
  return raw ? parseStoredSchedule(raw) : null;
}

export function saveSchedule(
  userId: number | null,
  data: StoredSchedule
): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(storageKey(userId), JSON.stringify(data));
}

export function clearSchedule(userId: number | null): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(storageKey(userId));
}
