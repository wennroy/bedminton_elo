import type Database from "better-sqlite3";
import { getDb } from "./db";

export interface Signup {
  id: number;
  sessionDate: string;
  playerId: number;
  name: string;
  partySize: number;
  createdAt: string;
}

export interface SignupSummary {
  count: number;
  totalPeople: number;
}

const EIGHT_HOURS_MS = 8 * 3600_000;
const DAY_MS = 24 * 3600_000;
const WEDNESDAY = 3;
const CUTOFF_HOUR = 20;

/**
 * 当前开放报名的场次日期（YYYY-MM-DD）。
 * 固定每周三局：上海时间周三 20:00 前 → 当周周三；之后 → 下周三。
 * 容器 TZ 不可靠，用 UTC+8 偏移后读 getUTC*，不依赖服务器本地时区。
 */
export function getActiveSessionDate(now: Date): string {
  const sh = new Date(now.getTime() + EIGHT_HOURS_MS);
  const day = sh.getUTCDay();
  const daysAhead =
    day === WEDNESDAY && sh.getUTCHours() < CUTOFF_HOUR
      ? 0
      : ((WEDNESDAY - day + 7) % 7) || 7;
  const target = new Date(sh.getTime() + daysAhead * DAY_MS);
  const y = target.getUTCFullYear();
  const m = String(target.getUTCMonth() + 1).padStart(2, "0");
  const d = String(target.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

/** "2026-09-09" → "9月9日（周三）"（场次恒为周三，无需算星期） */
export function formatSessionDate(sessionDate: string): string {
  const [, m, d] = sessionDate.split("-").map(Number);
  return `${m}月${d}日（周三）`;
}

export function listSignups(
  sessionDate: string,
  db?: Database.Database
): Signup[] {
  const conn = db ?? getDb();
  return conn
    .prepare(
      `SELECT s.id,
              s.session_date AS sessionDate,
              s.player_id AS playerId,
              p.name,
              s.party_size AS partySize,
              s.created_at AS createdAt
         FROM signups s
         JOIN players p ON p.id = s.player_id
        WHERE s.session_date = ?
        ORDER BY s.created_at ASC, s.id ASC`
    )
    .all(sessionDate) as Signup[];
}

export function upsertSignup(
  sessionDate: string,
  playerId: number,
  partySize: 1 | 2,
  db?: Database.Database
): void {
  const conn = db ?? getDb();
  conn
    .prepare(
      `INSERT INTO signups (session_date, player_id, party_size)
       VALUES (?, ?, ?)
       ON CONFLICT (session_date, player_id)
       DO UPDATE SET party_size = excluded.party_size`
    )
    .run(sessionDate, playerId, partySize);
}

export function removeSignup(
  sessionDate: string,
  playerId: number,
  db?: Database.Database
): void {
  const conn = db ?? getDb();
  conn
    .prepare(`DELETE FROM signups WHERE session_date = ? AND player_id = ?`)
    .run(sessionDate, playerId);
}

export function signupSummary(
  sessionDate: string,
  db?: Database.Database
): SignupSummary {
  const conn = db ?? getDb();
  return conn
    .prepare(
      `SELECT COUNT(*) AS count,
              COALESCE(SUM(party_size), 0) AS totalPeople
         FROM signups
        WHERE session_date = ?`
    )
    .get(sessionDate) as SignupSummary;
}
