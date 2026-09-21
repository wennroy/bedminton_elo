import type { LocalDate, RatingSegment } from "./types";

const LOCAL_DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/;
const ISO_INSTANT_PATTERN = /^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?(Z|[+-]\d{2}:\d{2})$/;
const DAY_MILLISECONDS = 24 * 60 * 60 * 1000;
const LOCAL_DATE_SEARCH_WINDOW_MILLISECONDS = 2 * DAY_MILLISECONDS;
const MAX_LOCAL_DATE = "9999-12-31";
const MAX_LOCAL_DATE_YEAR = 9999;

const shanghaiDateFormatter = new Intl.DateTimeFormat("en-US-u-ca-gregory-nu-latn", {
  timeZone: "Asia/Shanghai",
  era: "short",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

export interface RatingSegmentRange {
  /** Inclusive Shanghai-midnight ISO instant. */
  start: string;
  /** Exclusive Shanghai-midnight ISO instant. */
  end: string;
  /** The first quarterly boundary eligible to identify a rating season. */
  firstSeasonStart: LocalDate;
}

/** Returns whether a value is a strict, year-one-or-later Gregorian date. */
export function isValidLocalDate(value: unknown): value is LocalDate {
  if (typeof value !== "string") return false;

  const match = LOCAL_DATE_PATTERN.exec(value);
  if (match === null) return false;

  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  return year >= 1 && month >= 1 && month <= 12 && day >= 1 && day <= daysInMonth(year, month);
}

/** Asserts that a value is a strict, year-one-or-later Gregorian date. */
export function assertLocalDate(value: unknown): asserts value is LocalDate {
  if (!isValidLocalDate(value)) {
    throw new RangeError("expected a valid YYYY-MM-DD Gregorian date");
  }
}

/** Converts an ISO instant to its Gregorian calendar date in Asia/Shanghai. */
export function shanghaiLocalDateFromInstant(instant: string): LocalDate {
  return shanghaiLocalDateFromDate(new Date(assertIsoInstant(instant)));
}

/**
 * Returns the earliest instant whose Shanghai local date is `localDate`.
 * Usually this is Shanghai midnight; if a historical transition skipped
 * midnight, it is the first valid instant later that day.
 */
export function shanghaiMidnightIso(localDate: LocalDate): string {
  assertLocalDate(localDate);
  const nominalUtcMidnightMilliseconds =
    (dayNumber(localDate) - dayNumber("1970-01-01")) * DAY_MILLISECONDS;
  let before = nominalUtcMidnightMilliseconds - LOCAL_DATE_SEARCH_WINDOW_MILLISECONDS;
  let atOrAfter = nominalUtcMidnightMilliseconds + LOCAL_DATE_SEARCH_WINDOW_MILLISECONDS;
  while (atOrAfter - before > 1) {
    const middle = before + Math.floor((atOrAfter - before) / 2);
    if (compareCivilDates(shanghaiCivilDateFromDate(new Date(middle)), localDateParts(localDate)) < 0) {
      before = middle;
    } else {
      atOrAfter = middle;
    }
  }
  if (compareCivilDates(shanghaiCivilDateFromDate(new Date(atOrAfter)), localDateParts(localDate)) !== 0) {
    throw new RangeError("local date has no valid instant in Asia/Shanghai");
  }
  return new Date(atOrAfter).toISOString().replace(".000Z", "Z");
}

/** Returns the Monday beginning the Shanghai week containing a local date. */
export function weekStart(localDate: LocalDate): LocalDate {
  assertLocalDate(localDate);
  const monday = dayNumber("1970-01-05");
  const weekdayOffset = modulo(dayNumber(localDate) - monday, 7);
  return localDateFromDayNumber(dayNumber(localDate) - weekdayOffset);
}

/** Returns the first day of the natural quarter containing a local date. */
export function quarterStart(localDate: LocalDate): LocalDate {
  const { year, month } = localDateParts(localDate);
  return formatLocalDate(year, Math.floor((month - 1) / 3) * 3 + 1, 1);
}

/**
 * Returns the Monday immediately after the week containing a local date.
 * Throws when that boundary cannot be represented as a four-digit LocalDate.
 */
export function nextWeekStart(localDate: LocalDate): LocalDate {
  const currentWeekStart = weekStart(localDate);
  if (currentWeekStart > "9999-12-24") {
    throw boundaryOverflowError();
  }
  return addDays(currentWeekStart, 7);
}

/**
 * Returns the first day of the natural quarter after a local date's quarter.
 * Throws when that boundary cannot be represented as a four-digit LocalDate.
 */
export function nextQuarterStart(localDate: LocalDate): LocalDate {
  const { year, month } = localDateParts(quarterStart(localDate));
  if (year === 9999 && month === 10) {
    throw boundaryOverflowError();
  }
  return month === 10 ? formatLocalDate(year + 1, 1, 1) : formatLocalDate(year, month + 3, 1);
}

/** Returns the first weekly or quarterly boundary strictly after a local date. */
export function nextRatingBoundary(localDate: LocalDate): LocalDate {
  const nextWeek = nextWeekStart(localDate);
  if (quarterStart(localDate) === "9999-10-01") return nextWeek;
  const nextQuarter = nextQuarterStart(localDate);
  return nextWeek < nextQuarter ? nextWeek : nextQuarter;
}

/** Returns whether a Shanghai match date occurs after an as-of instant's Shanghai date. */
export function isFutureShanghaiLocalDate(playedAt: LocalDate, asOf: string): boolean {
  assertLocalDate(playedAt);
  return playedAt > shanghaiLocalDateFromInstant(asOf);
}

/** Returns the non-empty weekly/quarterly segment containing an as-of instant. */
export function ratingSegmentAt(asOf: string, firstSeasonStart: LocalDate): RatingSegment {
  assertFirstSeasonStart(firstSeasonStart);
  const asOfDate = shanghaiLocalDateFromInstant(asOf);
  const segmentStart = maxLocalDate(weekStart(asOfDate), quarterStart(asOfDate));
  return createRatingSegment(segmentStart, nextRatingBoundary(segmentStart), firstSeasonStart);
}

/**
 * Iterates non-empty calendar segments in `[start, end)`, splitting on every
 * Shanghai Monday and natural-quarter boundary. Inputs must be canonical
 * Shanghai-midnight ISO instants so `h` is always whole local days divided by 7.
 */
export function iterateRatingSegments({
  start,
  end,
  firstSeasonStart,
}: RatingSegmentRange): RatingSegment[] {
  assertFirstSeasonStart(firstSeasonStart);
  const startDate = localDateAtShanghaiMidnight(start);
  const endDate = localDateAtShanghaiMidnight(end);
  if (startDate > endDate) {
    throw new RangeError("segment range start must not be after end");
  }

  const segments: RatingSegment[] = [];
  for (let cursor = startDate; cursor < endDate; ) {
    const boundary = nextRatingBoundary(cursor);
    const segmentEnd = boundary < endDate ? boundary : endDate;
    segments.push(createRatingSegment(cursor, segmentEnd, firstSeasonStart));
    cursor = segmentEnd;
  }
  return segments;
}

function daysInMonth(year: number, month: number): number {
  if (month === 2) return isLeapYear(year) ? 29 : 28;
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}

function isLeapYear(year: number): boolean {
  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
}

function dayNumber(localDate: LocalDate): number {
  const { year, month, day } = localDateParts(localDate);
  const adjustedYear = year - (month <= 2 ? 1 : 0);
  const era = Math.floor(adjustedYear / 400);
  const yearOfEra = adjustedYear - era * 400;
  const monthPrime = month > 2 ? month - 3 : month + 9;
  const dayOfYear = Math.floor((153 * monthPrime + 2) / 5) + day - 1;
  return era * 146097 + yearOfEra * 365 + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100) + dayOfYear;
}

function localDateFromDayNumber(dayNumber: number): LocalDate {
  const era = Math.floor(dayNumber / 146097);
  const dayOfEra = dayNumber - era * 146097;
  const yearOfEra = Math.floor(
    (dayOfEra - Math.floor(dayOfEra / 1460) + Math.floor(dayOfEra / 36524) - Math.floor(dayOfEra / 146096)) / 365
  );
  const dayOfYear =
    dayOfEra -
    (yearOfEra * 365 + Math.floor(yearOfEra / 4) - Math.floor(yearOfEra / 100));
  const monthPrime = Math.floor((5 * dayOfYear + 2) / 153);
  const day = dayOfYear - Math.floor((153 * monthPrime + 2) / 5) + 1;
  const month = monthPrime < 10 ? monthPrime + 3 : monthPrime - 9;
  const year = yearOfEra + era * 400 + (month <= 2 ? 1 : 0);
  return formatLocalDate(year, month, day);
}

function addDays(localDate: LocalDate, days: number): LocalDate {
  return localDateFromDayNumber(dayNumber(localDate) + days);
}

function formatLocalDate(year: number, month: number, day: number): LocalDate {
  if (year < 1 || year > MAX_LOCAL_DATE_YEAR) {
    throw new RangeError("local date must remain within years 0001 through 9999");
  }
  return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function modulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

function localDateParts(localDate: LocalDate): { year: number; month: number; day: number } {
  assertLocalDate(localDate);
  const match = LOCAL_DATE_PATTERN.exec(localDate);
  if (match === null) throw new Error("unreachable");
  return { year: Number(match[1]), month: Number(match[2]), day: Number(match[3]) };
}

function assertIsoInstant(value: string): string {
  const match = ISO_INSTANT_PATTERN.exec(value);
  if (match === null || !isValidIsoCalendarDate(match[1])) {
    throw new RangeError("expected an ISO instant with an explicit time zone");
  }

  const hour = Number(match[2]);
  const minute = Number(match[3]);
  const second = match[4] === undefined ? 0 : Number(match[4]);
  const offset = match[6];
  const offsetHour = offset === "Z" ? 0 : Number(offset.slice(1, 3));
  const offsetMinute = offset === "Z" ? 0 : Number(offset.slice(4, 6));
  if (
    hour > 23 ||
    minute > 59 ||
    second > 59 ||
    offsetHour > 23 ||
    offsetMinute > 59 ||
    !Number.isFinite(Date.parse(value))
  ) {
    throw new RangeError("expected an ISO instant with an explicit time zone");
  }
  return value;
}

function isValidIsoCalendarDate(value: string): boolean {
  const match = LOCAL_DATE_PATTERN.exec(value);
  if (match === null) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  return month >= 1 && month <= 12 && day >= 1 && day <= daysInMonth(year, month);
}

function assertFirstSeasonStart(value: LocalDate): void {
  assertLocalDate(value);
  const { month, day } = localDateParts(value);
  if (day !== 1 || ![1, 4, 7, 10].includes(month)) {
    throw new RangeError("firstSeasonStart must be a valid quarter start");
  }
}

function localDateAtShanghaiMidnight(instant: string): LocalDate {
  const localDate = shanghaiLocalDateFromInstant(instant);
  if (instant !== shanghaiMidnightIso(localDate)) {
    throw new RangeError("segment range boundaries must be canonical Shanghai-midnight ISO instants");
  }
  return localDate;
}

function formatParts(
  formatter: Intl.DateTimeFormat,
  date: Date
): Record<string, string> {
  return Object.fromEntries(
    formatter
      .formatToParts(date)
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, part.value])
  );
}

function shanghaiLocalDateFromDate(date: Date): LocalDate {
  const { year, month, day } = shanghaiCivilDateFromDate(date);
  return formatLocalDate(year, month, day);
}

function shanghaiCivilDateFromDate(date: Date): { year: number; month: number; day: number } {
  const parts = formatParts(shanghaiDateFormatter, date);
  const displayedYear = Number(parts.year);
  return {
    year: parts.era === "BC" ? 1 - displayedYear : displayedYear,
    month: Number(parts.month),
    day: Number(parts.day),
  };
}

function compareCivilDates(
  first: { year: number; month: number; day: number },
  second: { year: number; month: number; day: number }
): number {
  if (first.year !== second.year) return first.year - second.year;
  if (first.month !== second.month) return first.month - second.month;
  return first.day - second.day;
}

function createRatingSegment(
  start: LocalDate,
  end: LocalDate,
  firstSeasonStart: LocalDate
): RatingSegment {
  const segmentWeekStart = weekStart(start);
  return {
    id: `${segmentWeekStart}:${start}`,
    weekStart: segmentWeekStart,
    seasonId: start < firstSeasonStart ? null : quarterStart(start),
    start: shanghaiMidnightIso(start),
    end: shanghaiMidnightIso(end),
    h: (dayNumber(end) - dayNumber(start)) / 7,
  };
}

function maxLocalDate(first: LocalDate, second: LocalDate): LocalDate {
  return first > second ? first : second;
}

function boundaryOverflowError(): RangeError {
  return new RangeError(`cannot calculate a rating boundary after ${MAX_LOCAL_DATE}`);
}
