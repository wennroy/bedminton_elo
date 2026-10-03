import type { RatingViewWeekSegment } from "@/lib/ratings/view-types";
import { formatTrendSeasonLabel } from "@/lib/ratings/chart-data";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * 个人周评分明细（glicko2 只读展示）：消费 view.weekSegments，
 * 按周区段列出该球员的逐场 Estimated、段末 Final/周校准与赛季重置。
 * Final 不摊到各场——逐场只显示单场预估，校准与重置是独立事件行。
 * 纯展示组件：数据全部来自服务端投影，不在此处结算；
 * 默认整体收起（原生 details/summary，无 JS），收起时摘要一行 Final/本周合计。
 */

/** 该球员在某场比赛的事实（阵容/比分/胜负），由页面按 matchId 传入。 */
export interface RatingPeriodFact {
  teammates: string[];
  opponents: string[];
  scoreFor: number;
  scoreAgainst: number;
  won: boolean;
}

interface RatingPeriodDetailsProps {
  playerId: number;
  /** view.weekSegments：跨季周被拆成两段，重置挂在触发它的那一段。 */
  segments: readonly RatingViewWeekSegment[];
  /** 该球员每周 Final 展示值（segmentId → 取整），来自 view.points 的 weekly_final。 */
  finalBySegment: Readonly<Record<string, number>>;
  /** matchId → 该场事实；重放含但名单查询缺失的比赛（已删球员）可能缺项。 */
  matchFacts: Readonly<Record<number, RatingPeriodFact>>;
}

const deltaClass = (delta: number) =>
  cn(
    "font-num",
    delta > 0 ? "text-win" : delta < 0 ? "text-loss" : "text-muted-foreground"
  );

const formatDelta = (delta: number) => (delta > 0 ? `+${delta}` : `${delta}`);

function seasonLabel(seasonId: string | null): string {
  return seasonId === null ? "尚未进入赛季" : formatTrendSeasonLabel(seasonId);
}

export function RatingPeriodDetails({
  playerId,
  segments,
  finalBySegment,
  matchFacts,
}: RatingPeriodDetailsProps) {
  const key = String(playerId);

  // 只列出与该球员有关的区段：有其逐场变化、校准条目或重置变化。
  const relevant = segments.filter((segment) => {
    if (segment.matches.some((m) => m.changes.some((c) => c.playerId === playerId))) {
      return true;
    }
    if (segment.correction[key] !== undefined) return true;
    return segment.reset?.changes.some((c) => c.playerId === playerId) ?? false;
  });

  if (relevant.length === 0) {
    return (
      <section className="rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
        <div className="mb-[18px]">
          <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
            周评分明细
          </h2>
          <div className="mt-[3px] text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
            RATING PERIODS
          </div>
        </div>
        <div className="grid min-h-[120px] place-items-center rounded-[10px] bg-secondary text-xs text-muted-foreground">
          还没有比赛记录，记一场后这里会按周列出逐场预估与正式结算。
        </div>
      </section>
    );
  }

  const pillClass =
    "inline-flex items-center rounded-[5px] px-[7px] py-1 text-[10px] font-bold";

  // 收起态摘要：最近一个已结算段的 Final + 进行中段的本周合计净变化。
  const newestFirst = [...relevant].reverse();
  const latestFinal = newestFirst
    .map((segment) => finalBySegment[segment.segmentId])
    .find((final) => final !== undefined);
  const currentSegment = newestFirst.find(
    (segment) => segment.correction[key] === undefined
  );
  const currentNet = currentSegment
    ? currentSegment.matches.reduce(
        (sum, estimate) =>
          sum +
          Math.round(
            estimate.changes.find((c) => c.playerId === playerId)?.delta ?? 0
          ),
        0
      )
    : null;
  const summaryLine = [
    latestFinal !== undefined ? `Final ${latestFinal}` : null,
    currentNet !== null
      ? `本周合计 ${formatDelta(currentNet)}（进行中）`
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <section className="overflow-hidden rounded-2xl border border-border bg-card">
      {/* 原生 details：键盘可及（Enter/Space 切换），默认收起无动画依赖。 */}
      <details className="group">
        <summary className="block cursor-pointer list-none px-[18px] pt-[19px] min-[761px]:px-[25px] min-[761px]:pt-[22px] [&::-webkit-details-marker]:hidden">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
                周评分明细
              </h2>
              <div className="mt-[3px] text-[9px] font-bold tracking-[1.5px] text-muted-foreground">
                RATING PERIODS
              </div>
            </div>
            <ChevronDown
              aria-hidden="true"
              className="size-4 shrink-0 text-muted-foreground transition-transform group-open:rotate-180"
            />
          </div>
          <p className="mt-2 font-num text-[11px] leading-[1.7] text-muted-foreground">
            {summaryLine || "展开查看逐场预估与正式结算"}
          </p>
        </summary>

        <p className="mt-2 px-[18px] text-[11px] leading-[1.7] text-muted-foreground min-[761px]:px-[25px]">
          逐场为 Estimated 预估；周一正式结算给出 Final 与校准（可能上调或下调）；赛季重置单列，不计入比赛表现。
        </p>

      <div className="mt-[18px]">
        {newestFirst.map((segment) => {
          const isCurrent = segment.correction[key] === undefined;
          const matchRows = segment.matches
            .map((estimate) => ({
              estimate,
              change: estimate.changes.find((c) => c.playerId === playerId),
            }))
            .filter((row) => row.change !== undefined);
          const matchTotal = matchRows.reduce(
            (sum, row) => sum + Math.round(row.change!.delta),
            0
          );
          const correction = segment.correction[key];
          const resetChange = segment.reset?.changes.find(
            (c) => c.playerId === playerId
          );
          const final = finalBySegment[segment.segmentId];
          const net =
            matchTotal + (correction !== undefined ? Math.round(correction) : 0);

          return (
            <article
              key={segment.segmentId}
              className="border-t border-border px-[18px] py-4 min-[761px]:px-[25px]"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-[13px] font-bold text-card-foreground">
                  {segment.weekStart.slice(5).replace("-", ".")} 当周
                  <span className="ml-2 text-[11px] font-normal text-muted-foreground">
                    {seasonLabel(segment.seasonId)}
                  </span>
                </h3>
                <span
                  className={cn(
                    pillClass,
                    isCurrent
                      ? "bg-win-bg text-win"
                      : "bg-secondary text-muted-foreground"
                  )}
                >
                  {isCurrent ? "进行中" : "已结算"}
                </span>
              </div>
              {/* 已结算段：逐场数字永远是单场预估，不摊 Final——去徽标、加一句口径说明。 */}
              {isCurrent ? null : (
                <p className="mt-1 text-[10px] leading-snug text-muted-foreground">
                  逐场为单场预估，正式结果见校准与 Final 行。
                </p>
              )}

              <div className="mt-3 space-y-2 text-[11px]">
                {matchRows.map(({ estimate, change }) => {
                  const fact = matchFacts[estimate.matchId];
                  const delta = Math.round(change!.delta);
                  return (
                    <div
                      key={estimate.eventId}
                      className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1"
                    >
                      <span className="text-muted-foreground">
                        {estimate.playedAt.slice(5).replace("-", ".")}
                        {fact ? (
                          <>
                            {" "}
                            我方 {fact.teammates.join(" / ") || "（无搭档）"}
                            {" vs "}
                            {fact.opponents.join(" / ")}
                            <span className="font-num">
                              {" "}
                              {fact.scoreFor}:{fact.scoreAgainst}
                            </span>
                          </>
                        ) : (
                          <> 比赛 #{estimate.matchId}</>
                        )}
                      </span>
                      <span className="flex items-center gap-2">
                        {/* 「预估」徽标仅保留给进行中段；已结算段的逐场口径由段头说明。 */}
                        {isCurrent ? (
                          <span className="rounded-[4px] bg-secondary px-1.5 py-0.5 text-[9px] font-bold text-muted-foreground">
                            预估
                          </span>
                        ) : null}
                        <span className={deltaClass(delta)}>
                          {formatDelta(delta)}
                        </span>
                      </span>
                    </div>
                  );
                })}

                {correction !== undefined ? (
                  <div className="flex items-center justify-between gap-3 border-t border-dashed border-border pt-2">
                    <span className="text-muted-foreground">
                      周正式结算 · 校准（Final − 预估）
                    </span>
                    <span className={deltaClass(Math.round(correction))}>
                      {formatDelta(Math.round(correction))}
                    </span>
                  </div>
                ) : null}

                {final !== undefined ? (
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">段末 Final</span>
                    <span className="font-num font-semibold text-card-foreground">
                      {final}
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">
                      周一正式结算后给出 Final
                    </span>
                    <span className="font-num text-muted-foreground">—</span>
                  </div>
                )}

                {resetChange ? (
                  <div className="mt-1 flex flex-wrap items-center justify-between gap-2 rounded-[8px] bg-secondary px-3 py-2">
                    <span className="text-muted-foreground">
                      赛季重置 · 进入 {seasonLabel(segment.reset!.seasonId)}：
                      {Math.round(resetChange.before.r)} →{" "}
                      {Math.round(resetChange.after.r)}
                    </span>
                    <span className={deltaClass(Math.round(resetChange.delta))}>
                      {formatDelta(Math.round(resetChange.delta))}
                    </span>
                  </div>
                ) : null}

                <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-2 text-[10px] text-muted-foreground">
                  <span>
                    本周合计{" "}
                    <strong
                      className={cn(
                        "font-num font-semibold",
                        net > 0
                          ? "text-win"
                          : net < 0
                            ? "text-loss"
                            : "text-foreground"
                      )}
                    >
                      {formatDelta(net)}
                    </strong>
                  </span>
                  <span>
                    逐场 {formatDelta(matchTotal)}
                    {correction !== undefined
                      ? ` · 校准 ${formatDelta(Math.round(correction))}`
                      : ""}
                    {resetChange
                      ? ` · 重置 ${formatDelta(Math.round(resetChange.delta))}`
                      : ""}
                  </span>
                </div>
              </div>
            </article>
          );
        })}
      </div>
      </details>
    </section>
  );
}
