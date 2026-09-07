/**
 * 每周名句：静态原创句子池，按上海时区（UTC+8）周一 00:00 归一到当周，
 * 用周序号对池长取模确定当期句子。不接 LLM / DB，刷新与设备无关。
 */

export interface WeeklyQuote {
  /** 当周周一（上海时区），YYYY-MM-DD */
  weekStart: string;
  /** 形如 "2026.09.07 — 09.13" */
  dateLabel: string;
  text: string;
}

/** 原创短句池：12–26 字，与打球 / 配合 / 练习有关，克制不说教。全站仅此一处励志文案。 */
export const WEEKLY_QUOTES: string[] = [
  "把注意力留给下一拍。",
  "球还没落地，就还有下一种可能。",
  "最好的配合，是有人接住你的下一拍。",
  "比分会归零，练习不会。",
  "先站稳，再出手。",
  "有些答案，要多打一场才知道。",
  "场上多一拍耐心，场下少一分遗憾。",
  "好搭档会在你跑动之前启动。",
  "练够一百次，才算会了这一拍。",
  "脚步到位了，手上自然就有答案。",
];

const EIGHT_HOURS_MS = 8 * 3600_000;
const DAY_MS = 24 * 3600_000;
const WEEK_MS = 7 * DAY_MS;

/** 归一到上海时区当周周一 00:00（以 UTC 时间戳表示）。 */
function mondayOf(date: Date): Date {
  const shanghai = new Date(date.getTime() + EIGHT_HOURS_MS);
  const day = shanghai.getUTCDay();
  return new Date(
    Date.UTC(
      shanghai.getUTCFullYear(),
      shanghai.getUTCMonth(),
      shanghai.getUTCDate() - ((day + 6) % 7)
    )
  );
}

export function quoteForWeek(date: Date, offsetWeeks = 0): WeeklyQuote {
  const start = mondayOf(date);
  start.setUTCDate(start.getUTCDate() + offsetWeeks * 7);
  const end = new Date(start.getTime() + 6 * DAY_MS);

  const weekIndex = Math.floor(start.getTime() / WEEK_MS);
  const poolSize = WEEKLY_QUOTES.length;
  const text = WEEKLY_QUOTES[((weekIndex % poolSize) + poolSize) % poolSize];

  const startIso = start.toISOString().slice(0, 10);
  const endIso = end.toISOString().slice(0, 10);
  return {
    weekStart: startIso,
    dateLabel: `${startIso.replaceAll("-", ".")} — ${endIso
      .slice(5)
      .replace("-", ".")}`,
    text,
  };
}
