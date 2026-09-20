import type { MatchWithNames } from "@/lib/repo";

export type MatchFilters = {
  query: string;
  playerId: number | null;
  from: string;
  to: string;
};

const CALENDAR_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;
const SQLITE_DATETIME = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/;
const ISO_DATETIME =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?$/;

function isValidDateTime(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number
): boolean {
  const date = new Date(0);
  date.setUTCFullYear(year, month - 1, day);
  date.setUTCHours(hour, minute, second, 0);

  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day &&
    date.getUTCHours() === hour &&
    date.getUTCMinutes() === minute &&
    date.getUTCSeconds() === second
  );
}

function isCalendarDate(value: string): boolean {
  const parts = CALENDAR_DATE.exec(value);
  if (!parts) return false;

  const [, year, month, day] = parts.map(Number);
  return isValidDateTime(year, month, day, 0, 0, 0);
}

function createdAtTimestamp(value: string): number | null {
  const sqliteParts = SQLITE_DATETIME.exec(value);
  if (sqliteParts) {
    const [, year, month, day, hour, minute, second] = sqliteParts.map(Number);
    if (!isValidDateTime(year, month, day, hour, minute, second)) return null;

    const date = new Date(0);
    date.setUTCFullYear(year, month - 1, day);
    date.setUTCHours(hour, minute, second, 0);
    return date.getTime();
  }

  const isoParts = ISO_DATETIME.exec(value);
  if (!isoParts) return null;
  const year = Number(isoParts[1]);
  const month = Number(isoParts[2]);
  const day = Number(isoParts[3]);
  const hour = Number(isoParts[4]);
  const minute = Number(isoParts[5]);
  const second = Number(isoParts[6] ?? "0");
  if (!isValidDateTime(year, month, day, hour, minute, second)) return null;

  const timestamp = Date.parse(value);
  return Number.isNaN(timestamp) ? null : timestamp;
}

function compareMatches(a: MatchWithNames, b: MatchWithNames): number {
  if (a.playedAt !== b.playedAt) return a.playedAt < b.playedAt ? 1 : -1;

  const aCreatedAt = createdAtTimestamp(a.createdAt);
  const bCreatedAt = createdAtTimestamp(b.createdAt);
  if (aCreatedAt !== null && bCreatedAt === null) return -1;
  if (aCreatedAt === null && bCreatedAt !== null) return 1;
  if (aCreatedAt !== null && bCreatedAt !== null && aCreatedAt !== bCreatedAt) {
    return bCreatedAt - aCreatedAt;
  }

  return b.id - a.id;
}

export function filterMatches(
  matches: readonly MatchWithNames[],
  filters: MatchFilters
): MatchWithNames[] {
  const query = filters.query.trim().toLowerCase();
  const { playerId, from, to } = filters;

  if ((from && !isCalendarDate(from)) || (to && !isCalendarDate(to)) || (from && to && from > to)) {
    return [];
  }

  return matches
    .filter((match) => {
      const playerMatches =
        playerId === null ||
        match.pa1 === playerId ||
        match.pa2 === playerId ||
        match.pb1 === playerId ||
        match.pb2 === playerId;
      if (!playerMatches) return false;

      const queryMatches =
        !query ||
        [match.pa1Name, match.pa2Name, match.pb1Name, match.pb2Name].some((name) =>
          name.toLowerCase().includes(query)
        );
      if (!queryMatches) return false;

      return (!from || match.playedAt >= from) && (!to || match.playedAt <= to);
    })
    .sort(compareMatches);
}
