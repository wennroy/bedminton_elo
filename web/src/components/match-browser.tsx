"use client";

import * as React from "react";
import Link from "next/link";
import { PlayerAvatar } from "@/components/player-avatar";
import { filterMatches, type MatchFilters } from "@/lib/match-browser";
import type { MatchWithNames, Player } from "@/lib/repo";
import { cn } from "@/lib/utils";

const PAGE_SIZE = 20;
const CALENDAR_DATE = /^(\d{4})-(\d{2})-(\d{2})$/;

const EMPTY_FILTERS: MatchFilters = {
  query: "",
  playerId: null,
  from: "",
  to: "",
};

function isCalendarDate(value: string): boolean {
  const parts = CALENDAR_DATE.exec(value);
  if (!parts) return false;

  const [, year, month, day] = parts.map(Number);
  const date = new Date(0);
  date.setUTCFullYear(year, month - 1, day);
  date.setUTCHours(0, 0, 0, 0);

  return (
    date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
  );
}

function formatMatchDate(date: string): string {
  return date.replaceAll("-", ".");
}

function winnerLabel(match: MatchWithNames): string {
  if (match.scoreA > match.scoreB) return "A 队获胜";
  if (match.scoreB > match.scoreA) return "B 队获胜";
  return "平局";
}

interface MatchBrowserProps {
  matches: MatchWithNames[];
  players: Player[];
}

export function MatchBrowser({ matches, players }: MatchBrowserProps) {
  const [filters, setFilters] = React.useState<MatchFilters>(EMPTY_FILTERS);
  const [visibleCount, setVisibleCount] = React.useState(PAGE_SIZE);

  const fromInvalid = filters.from !== "" && !isCalendarDate(filters.from);
  const toInvalid = filters.to !== "" && !isCalendarDate(filters.to);
  const invalidRange =
    !fromInvalid &&
    !toInvalid &&
    filters.from !== "" &&
    filters.to !== "" &&
    filters.from > filters.to;
  const invalidDateFilter = fromInvalid || toInvalid || invalidRange;

  const filteredMatches = React.useMemo(
    () => (invalidDateFilter ? [] : filterMatches(matches, filters)),
    [filters, invalidDateFilter, matches]
  );
  const visibleMatches = filteredMatches.slice(0, visibleCount);
  const hasMore = visibleCount < filteredMatches.length;

  const fromError = fromInvalid
    ? "开始日期格式无效，请输入有效的 YYYY-MM-DD 日期。"
    : null;
  const toError = toInvalid
    ? "结束日期格式无效，请输入有效的 YYYY-MM-DD 日期。"
    : null;
  const fromDescription = [
    fromError ? "match-browser-from-error" : null,
    invalidRange ? "match-browser-date-range-error" : null,
  ]
    .filter(Boolean)
    .join(" ") || undefined;
  const toDescription = [
    toError ? "match-browser-to-error" : null,
    invalidRange ? "match-browser-date-range-error" : null,
  ]
    .filter(Boolean)
    .join(" ") || undefined;

  function changeQuery(query: string) {
    setFilters((current) => ({ ...current, query }));
    setVisibleCount(PAGE_SIZE);
  }

  function changePlayer(playerId: number | null) {
    setFilters((current) => ({ ...current, playerId }));
    setVisibleCount(PAGE_SIZE);
  }

  function changeDate(key: "from" | "to", value: string) {
    setFilters((current) => ({ ...current, [key]: value }));
    setVisibleCount(PAGE_SIZE);
  }

  function clearFilters() {
    setFilters(EMPTY_FILTERS);
    setVisibleCount(PAGE_SIZE);
  }

  return (
    <section className="rounded-2xl border border-border bg-card p-[18px] min-[761px]:p-[25px]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[16px]">
            查找比赛
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            固定以 A 队和 B 队展示每场比赛。
          </p>
        </div>
        <p
          aria-live="polite"
          className="inline-flex min-h-9 items-center rounded-[7px] bg-secondary px-3 text-xs font-bold text-foreground"
        >
          共 {filteredMatches.length} 场比赛
        </p>
      </div>

      <div className="mt-5 border-t border-border pt-5">
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-full min-[761px]:w-[280px]">
            <label
              htmlFor="match-browser-query"
              className="mb-1.5 block text-xs font-semibold text-card-foreground"
            >
              搜索球员
            </label>
            <input
              id="match-browser-query"
              type="search"
              value={filters.query}
              onChange={(event) => changeQuery(event.target.value)}
              placeholder="输入球员姓名"
              className="h-10 w-full rounded-[8px] border border-input bg-background px-3 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>

          <div className="w-full min-[461px]:w-[calc(50%-0.375rem)] min-[761px]:w-[164px]">
            <label
              htmlFor="match-browser-from"
              className="mb-1.5 block text-xs font-semibold text-card-foreground"
            >
              开始日期
            </label>
            <input
              id="match-browser-from"
              type="date"
              value={filters.from}
              onChange={(event) => changeDate("from", event.target.value)}
              aria-invalid={fromInvalid || invalidRange}
              aria-describedby={fromDescription}
              className="h-10 w-full rounded-[8px] border border-input bg-background px-3 text-sm text-foreground outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring aria-[invalid=true]:border-destructive"
            />
            {fromError && (
              <p id="match-browser-from-error" role="alert" className="mt-1.5 text-xs text-destructive">
                {fromError}
              </p>
            )}
          </div>

          <div className="w-full min-[461px]:w-[calc(50%-0.375rem)] min-[761px]:w-[164px]">
            <label
              htmlFor="match-browser-to"
              className="mb-1.5 block text-xs font-semibold text-card-foreground"
            >
              结束日期
            </label>
            <input
              id="match-browser-to"
              type="date"
              value={filters.to}
              onChange={(event) => changeDate("to", event.target.value)}
              aria-invalid={toInvalid || invalidRange}
              aria-describedby={toDescription}
              className="h-10 w-full rounded-[8px] border border-input bg-background px-3 text-sm text-foreground outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring aria-[invalid=true]:border-destructive"
            />
            {toError && (
              <p id="match-browser-to-error" role="alert" className="mt-1.5 text-xs text-destructive">
                {toError}
              </p>
            )}
          </div>

          <button
            type="button"
            onClick={clearFilters}
            className="min-h-10 rounded-[8px] border border-border px-3 text-xs font-bold text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            清空筛选
          </button>
        </div>

        {invalidRange && (
          <p
            id="match-browser-date-range-error"
            role="alert"
            className="mt-3 text-xs text-destructive"
          >
            开始日期不能晚于结束日期
          </p>
        )}

        <fieldset className="mt-5">
          <legend className="text-xs font-semibold text-card-foreground">按球员筛选</legend>
          <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label="选择球员">
            <button
              type="button"
              aria-pressed={filters.playerId === null}
              onClick={() => changePlayer(null)}
              className={cn(
                "min-h-9 max-w-full break-all rounded-[7px] border border-border px-3 text-xs whitespace-normal transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                filters.playerId === null
                  ? "bg-secondary font-bold text-foreground"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground"
              )}
            >
              全部球员
            </button>
            {players.map((player) => (
              <button
                key={player.id}
                type="button"
                aria-pressed={filters.playerId === player.id}
                onClick={() => changePlayer(player.id)}
                className={cn(
                  "min-h-9 max-w-full break-all rounded-[7px] border border-border px-3 text-xs whitespace-normal transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  filters.playerId === player.id
                    ? "bg-secondary font-bold text-foreground"
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                )}
              >
                {player.name}
              </button>
            ))}
          </div>
        </fieldset>
      </div>

      <div className="mt-6">
        {matches.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-muted/30 py-[45px] text-center text-sm text-muted-foreground">
            还没有比赛记录
          </div>
        ) : invalidDateFilter ? (
          <div className="rounded-xl border border-dashed border-border bg-muted/30 py-[45px] text-center text-sm text-muted-foreground">
            请修正日期筛选后查看比赛。
          </div>
        ) : filteredMatches.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-muted/30 py-[45px] text-center text-sm text-muted-foreground">
            <p>没有符合条件的比赛</p>
            <button
              type="button"
              onClick={clearFilters}
              className="mt-3 min-h-9 rounded-[7px] border border-border px-3 text-xs font-bold text-foreground transition-colors hover:bg-secondary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              清空筛选
            </button>
          </div>
        ) : (
          <>
            <ol className="space-y-3" aria-label="比赛记录">
              {visibleMatches.map((match) => (
                <li key={match.id}>
                  <MatchCard match={match} />
                </li>
              ))}
            </ol>
            <div className="mt-3 flex flex-wrap items-center justify-center gap-3 border-t border-border pt-3 text-center">
              {hasMore ? (
                <>
                  <span className="text-xs text-muted-foreground">
                    已展示 {visibleMatches.length} / {filteredMatches.length} 场
                  </span>
                  <button
                    type="button"
                    onClick={() => setVisibleCount((count) => count + PAGE_SIZE)}
                    className="min-h-10 rounded-[8px] border border-border px-4 text-xs font-bold text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    加载更多
                  </button>
                </>
              ) : (
                <span className="inline-flex min-h-10 items-center text-xs text-muted-foreground">
                  已展示全部 {filteredMatches.length} 场比赛
                </span>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function MatchCard({ match }: { match: MatchWithNames }) {
  const teamA = [
    { id: match.pa1, name: match.pa1Name },
    { id: match.pa2, name: match.pa2Name },
  ];
  const teamB = [
    { id: match.pb1, name: match.pb1Name },
    { id: match.pb2, name: match.pb2Name },
  ];

  return (
    <article className="rounded-xl border border-border bg-background p-4 min-[761px]:p-[18px]">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <span className="block text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
            比赛日期
          </span>
          <time
            dateTime={match.playedAt}
            className="mt-0.5 block text-xs font-semibold text-card-foreground"
          >
            {formatMatchDate(match.playedAt)}
          </time>
        </div>
        <span className="rounded-[5px] bg-secondary px-[7px] py-1 text-[10px] font-bold text-card-foreground">
          {winnerLabel(match)}
        </span>
      </header>

      <div className="mt-3 grid gap-3 min-[761px]:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] min-[761px]:items-center min-[761px]:gap-5">
        <Team label="A 队" players={teamA} />
        <div className="flex min-w-[7rem] items-baseline justify-center rounded-[9px] bg-secondary px-3 py-2 font-num text-[25px] font-bold leading-none text-card-foreground">
          <span>{match.scoreA}</span>
          <span className="px-2 text-sm text-muted-foreground">:</span>
          <span>{match.scoreB}</span>
        </div>
        <Team label="B 队" players={teamB} alignEnd />
      </div>
    </article>
  );
}

function Team({
  label,
  players,
  alignEnd = false,
}: {
  label: string;
  players: { id: number; name: string }[];
  alignEnd?: boolean;
}) {
  return (
    <div className={cn("min-w-0", alignEnd && "min-[761px]:text-right")}>
      <span className="text-[9px] font-bold tracking-[1.5px] text-muted-foreground">{label}</span>
      <div className="mt-1.5 flex min-w-0 flex-col gap-1">
        {players.map((player) => (
          <Link
            key={player.id}
            href={`/players/${player.id}`}
            className={cn(
              "flex min-w-0 items-center gap-2 text-xs font-medium text-card-foreground transition-colors hover:text-win focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              alignEnd && "min-[761px]:justify-end"
            )}
          >
            <PlayerAvatar name={player.name} size="xs" className="size-6 shrink-0 text-[10px]" />
            <span className="min-w-0 break-words">{player.name}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
