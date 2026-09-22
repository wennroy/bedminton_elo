import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import {
  playerFunStats,
  playerMatches,
  playerRelations,
  playerSummary,
  type PlayerMatchRecord,
  type StatsData,
} from "@/lib/stats";
import { INITIAL_RATING } from "@/lib/elo";
import { getWeekRange } from "@/lib/weekly";
import { listMatchesByDate, listPlayers, type MatchWithNames } from "@/lib/repo";
import { loadRatingView } from "@/lib/rating-view";
import type { RatingIssue } from "@/lib/ratings/types";
import { RatingBoundaryRefresh } from "@/components/rating-boundary-refresh";
import { RatingModeControl } from "@/components/rating-mode-control";
import { RatingStatus, toRatingStatusInput } from "@/components/rating-status";
import { ProfileHeader, MoreMetrics } from "@/components/fun-stats";
import { PlayerTrend, RecentForm } from "@/components/player-trend";
import { PlayerRelations } from "@/components/player-relations";
import { PlayerMatchHistory } from "@/components/player-match-history";
import {
  RatingPeriodDetails,
  type RatingPeriodFact,
} from "@/components/rating-period-details";
import { cn } from "@/lib/utils";

export const dynamic = "force-dynamic";

interface PlayerPageProps {
  params: Promise<{ id: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

function todayString(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

const ISSUE_REASON_TEXT: Record<RatingIssue["reason"], string> = {
  invalid_date: "日期无效",
  invalid_score: "比分无效",
  duplicate_player: "同一球员重复出场",
  unknown_player: "包含未知球员",
};

/** glicko2 比赛记录：事实字段同旧口径，delta 占位 0（展示走 matchEstimates 的预估）。 */
function buildGlicko2MatchRecords(
  playerId: number,
  matches: readonly MatchWithNames[],
  estimateById: Readonly<Record<string, { changes: readonly { playerId: number; delta: number }[] } | undefined>>
): { records: PlayerMatchRecord[]; matchEstimates: Record<string, number | null> } {
  const records: PlayerMatchRecord[] = [];
  const matchEstimates: Record<string, number | null> = {};
  for (let i = matches.length - 1; i >= 0; i--) {
    const m = matches[i];
    const teamA = [m.pa1, m.pa2];
    const teamB = [m.pb1, m.pb2];
    const inA = teamA.includes(playerId);
    const inB = teamB.includes(playerId);
    if (!inA && !inB) continue;
    const aWon = m.scoreA > m.scoreB;
    const teammates = (inA ? teamA : teamB)
      .filter((id) => id !== playerId)
      .map((id) =>
        id === m.pa1
          ? m.pa1Name
          : id === m.pa2
            ? m.pa2Name
            : id === m.pb1
              ? m.pb1Name
              : m.pb2Name
      );
    records.push({
      id: m.id,
      date: m.playedAt,
      teammates,
      opponents: inA ? [m.pb1Name, m.pb2Name] : [m.pa1Name, m.pa2Name],
      scoreFor: inA ? m.scoreA : m.scoreB,
      scoreAgainst: inA ? m.scoreB : m.scoreA,
      won: inA ? aWon : !aWon,
      delta: 0,
    });
    const change = estimateById[String(m.id)]?.changes.find(
      (c) => c.playerId === playerId
    );
    matchEstimates[String(m.id)] = change ? Math.round(change.delta) : null;
  }
  return { records, matchEstimates };
}

export default async function PlayerPage({
  params,
  searchParams,
}: PlayerPageProps) {
  const { id: idParam } = await params;
  const sp = await searchParams;
  const playerId = Number(idParam);
  if (!Number.isFinite(playerId)) {
    notFound();
  }

  // 全页一次 loadRatingView：glicko2 分支纯消费投影，legacy 分支 view 为旧 StatsData。
  const result = loadRatingView({
    rating: typeof sp.rating === "string" ? sp.rating : undefined,
  });
  const isLegacy = result.model === "legacy";
  const ratingQuery = isLegacy ? "" : "?rating=glicko2";

  const metricCell =
    "px-5 py-[17px] min-[761px]:px-[25px] min-[761px]:py-[22px]";
  const metricLabel = "text-[11px] text-muted-foreground max-[760px]:text-[10px]";
  const metricValue =
    "mt-[5px] mb-1 flex items-center gap-[9px] font-num text-[38px] leading-[1.1] text-card-foreground min-[761px]:text-[43px]";
  const metricUnit = "text-[19px]";
  const metricSub = "text-[10px] text-muted-foreground max-[760px]:text-[9px]";

  if (isLegacy) {
    const data = result.view;
    const summary = playerSummary(playerId, data);
    if (!summary) {
      notFound();
    }

    const funStats = playerFunStats(playerId, data);
    const matches = playerMatches(playerId, data);
    const relations = playerRelations(playerId, data);

    // 排名：ELO 降序，同分按 id 升序（与排行榜口径一致）
    const eloOf = (id: number) =>
      Math.round(data.ratings.get(id)?.elo ?? INITIAL_RATING);
    const ordered = [...data.players].sort(
      (a, b) => eloOf(b.id) - eloOf(a.id) || a.id - b.id
    );
    const rank = ordered.findIndex((p) => p.id === playerId) + 1;

    // 本周变化（口径同首页）：当前 ELO − 本周一之前最后一个快照，无快照按初始分
    const { weekStart } = getWeekRange(todayString());
    let lastBeforeWeek: number | null = null;
    let hasHistory = false;
    const trendPoints: { date: string; elo: number }[] = [];
    for (const h of data.eloHistory) {
      if (h.playerId !== String(playerId)) continue;
      hasHistory = true;
      trendPoints.push({ date: h.date, elo: h.elo });
      if (h.date < weekStart) lastBeforeWeek = h.elo;
    }
    const currentElo = Math.round(summary.elo);
    const weekDelta = hasHistory
      ? currentElo - (lastBeforeWeek ?? INITIAL_RATING)
      : 0;

    const switcherPlayers = ordered.map((p) => ({
      id: p.id,
      name: p.name,
      elo: eloOf(p.id),
    }));

    const lastMatchDate = matches[0]?.date;
    const hasMatches = summary.totalMatches > 0;

    return (
      <div className="flex flex-col gap-5 min-[761px]:gap-6">
        <RatingBoundaryRefresh nextBoundary={null} />

        <div className="flex items-center justify-between gap-3">
          <Link
            href="/players"
            className="inline-flex min-h-9 items-center gap-[7px] text-xs text-muted-foreground transition-colors hover:text-win"
          >
            <ArrowLeft className="size-[15px]" strokeWidth={1.65} />
            所有球员
          </Link>
          <RatingModeControl current={result.model} />
        </div>

        <RatingStatus {...toRatingStatusInput(result)} />

        <ProfileHeader
          id={playerId}
          name={summary.name}
          rank={rank}
          players={switcherPlayers}
        />

        <section
          aria-label="球员核心数据"
          className="grid grid-cols-2 rounded-2xl border border-border bg-card min-[761px]:grid-cols-[1.2fr_1fr_1fr_1fr]"
        >
          <div
            className={cn(
              metricCell,
              "border-border max-[760px]:border-r max-[760px]:border-b min-[761px]:border-r"
            )}
          >
            <div className={metricLabel}>当前 ELO</div>
            <div className={metricValue}>
              <span>{currentElo}</span>
              <span
                className={cn(
                  "inline-flex items-center rounded-[5px] px-[7px] py-1 font-sans text-[10px] font-bold",
                  weekDelta > 0
                    ? "bg-win-bg text-win"
                    : weekDelta < 0
                      ? "bg-loss-bg text-loss"
                      : "bg-secondary text-muted-foreground"
                )}
              >
                {weekDelta > 0 ? `+${weekDelta}` : weekDelta}
              </span>
            </div>
            <div className={metricSub}>
              较 {weekStart.slice(5).replace("-", ".")} 前 · 最高{" "}
              {funStats.peakElo}
            </div>
          </div>
          <div
            className={cn(
              metricCell,
              "border-border max-[760px]:border-b min-[761px]:border-r"
            )}
          >
            <div className={metricLabel}>生涯胜率</div>
            <div className={metricValue}>
              {hasMatches ? (
                <>
                  <span>{summary.winRate}</span>
                  <span className={metricUnit}>%</span>
                </>
              ) : (
                <span>—</span>
              )}
            </div>
            <div className={metricSub}>
              {summary.wins} 胜 / {summary.losses} 负
            </div>
          </div>
          <div className={cn(metricCell, "border-border max-[760px]:border-r min-[761px]:border-r")}>
            <div className={metricLabel}>累计出场</div>
            <div className={metricValue}>
              <span>{summary.totalMatches}</span>
              <span className={metricUnit}>场</span>
            </div>
            <div className={metricSub}>生涯双打比赛</div>
          </div>
          <div className={metricCell}>
            <div className={metricLabel}>当前状态</div>
            <div className={metricValue}>
              {funStats.currentStreakType === "none" ? (
                <span>—</span>
              ) : (
                <>
                  <span>{funStats.currentStreak}</span>
                  <span className={metricUnit}>
                    连{funStats.currentStreakType === "win" ? "胜" : "负"}
                  </span>
                </>
              )}
            </div>
            <div className={metricSub}>
              {lastMatchDate
                ? `最近一场 · ${lastMatchDate.slice(5).replace("-", ".")}`
                : "还没有比赛"}
            </div>
          </div>
        </section>

        <div className="grid items-start gap-[18px] min-[761px]:grid-cols-[minmax(0,1.95fr)_minmax(260px,1fr)] min-[761px]:gap-[22px]">
          <PlayerTrend
            model="legacy"
            playerName={summary.name}
            points={trendPoints}
          />
          <RecentForm
            matches={matches}
            avgPointDiff={funStats.avgPointDiff}
            peakElo={funStats.peakElo}
            currentElo={currentElo}
          />
        </div>

        <PlayerRelations
          playerName={summary.name}
          partners={relations.partners}
          opponents={relations.opponents}
        />

        <PlayerMatchHistory matches={matches} playerName={summary.name} />

        <MoreMetrics
          model="legacy"
          mu={summary.mu}
          sigma={summary.sigma}
          longestWinStreak={funStats.longestWinStreak}
          peakElo={funStats.peakElo}
          peakEloDate={funStats.peakEloDate}
        />
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // glicko2：一次 loadRatingView 的投影；事实口径沿用旧 stats 函数。
  // -------------------------------------------------------------------------
  const view = result.view;
  const row = view.players.find((p) => p.playerId === playerId);
  if (result.freshness !== "unavailable" && !row) {
    notFound();
  }

  const directory = listPlayers();
  const playerName =
    row?.name ?? directory.find((p) => p.id === playerId)?.name;
  if (!playerName) {
    notFound();
  }

  // 比赛事实：glicko2 也需要名单/比分/胜负（投影只带评分事件）。
  const matches = listMatchesByDate();
  const { records, matchEstimates } = buildGlicko2MatchRecords(
    playerId,
    matches,
    view.matchEstimatesById
  );
  const matchFacts: Record<number, RatingPeriodFact> = {};
  for (const r of records) {
    matchFacts[r.id] = {
      teammates: r.teammates,
      opponents: r.opponents,
      scoreFor: r.scoreFor,
      scoreAgainst: r.scoreAgainst,
      won: r.won,
    };
  }

  // 事实口径指标（胜场/连胜/搭档/对手）：旧 stats 函数只消费比赛事实，
  // 空 ratings/eloHistory/tsPlayers 不被这些字段读取（评分值一律取自投影）。
  const factData: StatsData = {
    players: directory,
    matches,
    ratings: new Map(),
    eloHistory: [],
    tsPlayers: {},
  };
  const factSummary = playerSummary(playerId, factData);
  const funStats = playerFunStats(playerId, factData);
  const relations = playerRelations(playerId, factData);

  // 每周 Final 展示值（weekly_final 点），供周明细的「段末 Final」。
  const finalBySegment: Record<string, number> = {};
  for (const point of view.points) {
    if (point.kind !== "weekly_final" || point.playerId !== playerId) continue;
    finalBySegment[point.segment] = Math.round(point.r);
  }

  // 本周逐场预估合计（当前区段内该球员的 Estimated 变化之和）。
  const currentSegment = view.weekSegments.find(
    (segment) => segment.segmentId === result.currentSegmentId
  );
  let weekEstDelta = 0;
  if (row?.status === "estimated" && currentSegment) {
    for (const estimate of currentSegment.matches) {
      const change = estimate.changes.find((c) => c.playerId === playerId);
      if (change) weekEstDelta += change.delta;
    }
    weekEstDelta = Math.round(weekEstDelta);
  }

  const peak = row ? view.peakFinal[String(playerId)] : null;
  const peakDisplay = peak ? Math.round(peak.r) : null;
  const displayRating = row?.displayRating ?? null;
  const unrated = row === undefined || row.status === "unrated";
  const statusLabel =
    row?.status === "estimated"
      ? "本周逐场预估"
      : row?.status === "final"
        ? "沿用最近 Final"
        : null;

  // 切换球员名单：投影可用时附展示评分；服务不可用时回退到目录（未评级）。
  const switcherPlayers = (
    view.players.length > 0
      ? view.players.map((p) => ({ id: p.playerId, name: p.name, elo: p.displayRating }))
      : directory.map((p) => ({ id: p.id, name: p.name, elo: null }))
  );
  const hasMatches = (factSummary?.totalMatches ?? 0) > 0;
  const lastMatchDate = records[0]?.date;

  return (
    <div className="flex flex-col gap-5 min-[761px]:gap-6">
      <RatingBoundaryRefresh
        nextBoundary={result.model === "glicko2" ? result.nextBoundary : null}
      />

      <div className="flex items-center justify-between gap-3">
        <Link
          href={`/players${ratingQuery}`}
          className="inline-flex min-h-9 items-center gap-[7px] text-xs text-muted-foreground transition-colors hover:text-win"
        >
          <ArrowLeft className="size-[15px]" strokeWidth={1.65} />
          所有球员
        </Link>
        <RatingModeControl current={result.model} />
      </div>

      <RatingStatus {...toRatingStatusInput(result)} />

      <ProfileHeader
        id={playerId}
        name={playerName}
        rank={row?.rank ?? null}
        players={switcherPlayers}
        ratingQuery={ratingQuery}
      />

      <section
        aria-label="球员核心数据"
        className="grid grid-cols-2 rounded-2xl border border-border bg-card min-[761px]:grid-cols-[1.2fr_1fr_1fr_1fr]"
      >
        <div
          className={cn(
            metricCell,
            "border-border max-[760px]:border-r max-[760px]:border-b min-[761px]:border-r"
          )}
        >
          <div className={metricLabel}>当前评分</div>
          <div className={metricValue}>
            <span>{unrated ? "—" : displayRating}</span>
            {!unrated ? (
              <span
                className={cn(
                  "inline-flex items-center rounded-[5px] px-[7px] py-1 font-sans text-[10px] font-bold",
                  weekEstDelta > 0
                    ? "bg-win-bg text-win"
                    : weekEstDelta < 0
                      ? "bg-loss-bg text-loss"
                      : "bg-secondary text-muted-foreground"
                )}
              >
                {weekEstDelta > 0 ? `+${weekEstDelta}` : weekEstDelta}
              </span>
            ) : null}
          </div>
          <div className={metricSub}>
            {unrated
              ? "尚未评级 · 完成首场比赛后开始计分"
              : `${statusLabel} · 最高 ${peakDisplay ?? "—"}`}
          </div>
        </div>
        <div
          className={cn(
            metricCell,
            "border-border max-[760px]:border-b min-[761px]:border-r"
          )}
        >
          <div className={metricLabel}>生涯胜率</div>
          <div className={metricValue}>
            {hasMatches ? (
              <>
                <span>{factSummary?.winRate}</span>
                <span className={metricUnit}>%</span>
              </>
            ) : (
              <span>—</span>
            )}
          </div>
          <div className={metricSub}>
            {factSummary?.wins ?? 0} 胜 / {factSummary?.losses ?? 0} 负
          </div>
        </div>
        <div className={cn(metricCell, "border-border max-[760px]:border-r min-[761px]:border-r")}>
          <div className={metricLabel}>累计出场</div>
          <div className={metricValue}>
            <span>{factSummary?.totalMatches ?? 0}</span>
            <span className={metricUnit}>场</span>
          </div>
          <div className={metricSub}>生涯双打比赛</div>
        </div>
        <div className={metricCell}>
          <div className={metricLabel}>当前状态</div>
          <div className={metricValue}>
            {funStats.currentStreakType === "none" ? (
              <span>—</span>
            ) : (
              <>
                <span>{funStats.currentStreak}</span>
                <span className={metricUnit}>
                  连{funStats.currentStreakType === "win" ? "胜" : "负"}
                </span>
              </>
            )}
          </div>
          <div className={metricSub}>
            {lastMatchDate
              ? `最近一场 · ${lastMatchDate.slice(5).replace("-", ".")}`
              : "还没有比赛"}
          </div>
        </div>
      </section>

      <div className="grid items-start gap-[18px] min-[761px]:grid-cols-[minmax(0,1.95fr)_minmax(260px,1fr)] min-[761px]:gap-[22px]">
        <PlayerTrend
          model="glicko2"
          playerName={playerName}
          playerId={playerId}
          view={view}
          currentSegmentId={result.currentSegmentId ?? ""}
        />
        <RecentForm
          matches={records}
          avgPointDiff={funStats.avgPointDiff}
          peakElo={peakDisplay ?? displayRating ?? 0}
          currentElo={displayRating ?? 0}
          peakLabel="生涯最高评分"
        />
      </div>

      {view.issues.length > 0 ? (
        <section
          aria-label="未参与评分的记录"
          className="rounded-2xl border border-dashed border-loss/60 p-3.5 text-xs"
        >
          <div className="text-[13px] font-bold text-loss">
            {view.issues.length} 条记录未参与新版评分
          </div>
          <ul className="mt-1.5 space-y-1 text-muted-foreground">
            {view.issues.map((issue) => {
              const playedAt = matches.find((m) => m.id === issue.matchId)?.playedAt;
              return (
                <li key={`${issue.matchId}:${issue.reason}`}>
                  比赛 #{issue.matchId}
                  {playedAt ? `（${playedAt.replaceAll("-", ".")}）` : ""} ·{" "}
                  {ISSUE_REASON_TEXT[issue.reason]}，未参与评分。
                </li>
              );
            })}
          </ul>
        </section>
      ) : null}

      <RatingPeriodDetails
        playerId={playerId}
        segments={view.weekSegments}
        finalBySegment={finalBySegment}
        matchFacts={matchFacts}
      />

      <PlayerRelations
        playerName={playerName}
        partners={relations.partners}
        opponents={relations.opponents}
      />

      <PlayerMatchHistory
        matches={records}
        playerName={playerName}
        model="glicko2"
        matchEstimates={matchEstimates}
      />

      <MoreMetrics
        model="glicko2"
        rd={row?.rd ?? null}
        longestWinStreak={funStats.longestWinStreak}
        peak={peak}
      />
    </div>
  );
}
