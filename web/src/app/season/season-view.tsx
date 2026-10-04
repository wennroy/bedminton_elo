"use client";

import * as React from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { PlayerAvatar } from "@/components/player-avatar";
import type {
  SeasonPageData,
  SeasonRatingParams,
  SeasonStats,
} from "@/lib/season";
import type { FunMatch } from "@/lib/weekly";
import { formatTrendSeasonLabel } from "@/lib/ratings/chart-data";
import {
  Flame,
  Sparkles,
  Swords,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

interface SeasonViewProps {
  data: SeasonPageData;
  /** 当前生效的赛季参数（简介卡展示，不硬编码数字）。 */
  params: SeasonRatingParams;
}

export function SeasonView({ data, params }: SeasonViewProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  function handleSeasonChange(value: string) {
    const next = new URLSearchParams(searchParams.toString());
    next.set("season", value);
    router.push(`/season?${next.toString()}`);
  }

  const selected =
    data.stats?.seasonId ?? data.currentSeasonId ?? data.seasons[0] ?? "";

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-end gap-2">
        <span className="text-xs text-muted-foreground">选择赛季</span>
        <select
          value={selected}
          onChange={(e) => handleSeasonChange(e.target.value)}
          className="h-9 rounded-lg border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {data.seasons.map((seasonId) => (
            <option key={seasonId} value={seasonId}>
              {formatTrendSeasonLabel(seasonId)}
            </option>
          ))}
        </select>
      </div>

      {data.seasons.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          暂无赛季数据
        </div>
      ) : data.stats === null ? (
        <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          未找到{" "}
          {formatTrendSeasonLabel(data.requestedSeasonId ?? selected)}{" "}
          的赛季数据
        </div>
      ) : (
        <SeasonSections stats={data.stats} params={params} />
      )}
    </div>
  );
}

function SeasonSections({
  stats,
  params,
}: {
  stats: SeasonStats;
  params: SeasonRatingParams;
}) {
  const { fun } = stats;
  const hasFun = fun.closest || fun.blowout || fun.streakKing || fun.upset;
  const asOfLabel = formatAsOfLabel(stats.asOfLocalDate);

  return (
    <>
      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <div className="text-sm text-muted-foreground">
            {stats.start} ~ {stats.end}
          </div>
          {stats.inProgress ? (
            <span className="rounded-[5px] border border-dashed border-border px-1.5 py-0.5 text-[10px] font-bold text-muted-foreground">
              进行中 · 截至 {asOfLabel}
            </span>
          ) : (
            <span className="rounded-[5px] bg-win-bg px-1.5 py-0.5 text-[10px] font-bold text-win">
              已结束
            </span>
          )}
        </div>
        <div className="text-2xl font-bold text-card-foreground">
          {stats.label} 赛季战报
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <h2 className="mb-2 text-sm font-bold text-card-foreground">赛季制度</h2>
        <ul className="list-disc space-y-1 pl-4 text-xs text-muted-foreground">
          <li>自然季度划分，每季一个独立榜单。</li>
          <li>
            每周一结算：周内战绩为预估（Estimated），周一凌晨出正式值（Final）。
          </li>
          <li>
            跨季软重置：新赛季开始时，超出 {params.seasonLower}–
            {params.seasonUpper} 的部分按 {params.seasonRetention} 倍保留，评分不确定性（RD）下限{" "}
            {params.seasonRdFloor}。
          </li>
        </ul>
      </div>

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-lg font-bold text-foreground">赛季排名</h2>
          <span className="text-[10px] text-muted-foreground">
            新版 Glicko-2 · 模型 {stats.version.split("|")[0]}
            {stats.freshness === "stale"
              ? " · 以上为最后成功结算的结果，恢复后自动更新"
              : ""}
          </span>
        </div>
        {stats.rating.length === 0 ? (
          <Empty />
        ) : (
          <div className="space-y-2">
            {stats.rating.map((row) => (
              <RankRow
                key={row.playerId}
                rank={row.rank ?? 0}
                name={row.name}
                sub={`${row.matchesPlayed} 场`}
                value={
                  <span className="flex flex-wrap items-center justify-end gap-x-2">
                    {row.change !== null ? (
                      <span
                        className={row.change >= 0 ? "text-win" : "text-loss"}
                      >
                        {row.change >= 0 ? "+" : ""}
                        {row.change}
                        {row.change >= 0 ? (
                          <TrendingUp className="ml-1 inline size-4" />
                        ) : (
                          <TrendingDown className="ml-1 inline size-4" />
                        )}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">
                        本季新加入
                      </span>
                    )}
                    <span className="font-bold">{row.endR}</span>
                  </span>
                }
              />
            ))}
          </div>
        )}
        <p className="text-[10px] text-muted-foreground">
          排名与涨跌按期初（季首重置后）→ 期末（{stats.inProgress ? "当前评分" : "季末周 Final"}）计算，并列同名次。
        </p>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">出勤榜</h2>
        {stats.attendance.length === 0 ? (
          <Empty />
        ) : (
          <div className="space-y-2">
            {stats.attendance.slice(0, 5).map((s, i) => (
              <RankRow
                key={s.playerId}
                rank={i + 1}
                name={s.name}
                value={`${s.matches} 场`}
              />
            ))}
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">战绩王</h2>
        {stats.winKing.length === 0 ? (
          <Empty />
        ) : (
          <div className="space-y-2">
            {stats.winKing.slice(0, 5).map((s, i) => (
              <RankRow
                key={s.playerId}
                rank={i + 1}
                name={s.name}
                value={`${s.wins} 胜 ${s.losses} 负`}
              />
            ))}
          </div>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">最佳组合</h2>
        {stats.bestPair ? (
          <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex -space-x-2">
                <PlayerAvatar name={stats.bestPair.playerA} size="sm" />
                <PlayerAvatar name={stats.bestPair.playerB} size="sm" />
              </div>
              <div className="text-lg font-bold text-card-foreground">
                {stats.bestPair.playerA} / {stats.bestPair.playerB}
              </div>
            </div>
            <div className="text-sm text-muted-foreground">
              {stats.bestPair.wins} 胜 {stats.bestPair.total - stats.bestPair.wins}{" "}
              负 · {Math.round(stats.bestPair.winRate * 100)}% 胜率
            </div>
          </div>
        ) : (
          <Empty />
        )}
        <p className="text-[10px] text-muted-foreground">
          季内搭档 ≥5 场入围，按胜率排名。
        </p>
      </section>

      {hasFun && (
        <section className="flex flex-col gap-3">
          <h2 className="text-lg font-bold text-foreground">赛季趣闻</h2>
          <div className="grid grid-cols-2 gap-3">
            {fun.closest && (
              <FunCard icon={Swords} title="最胶着一战">
                <FunMatchLine match={fun.closest} />
              </FunCard>
            )}
            {fun.blowout && (
              <FunCard icon={Flame} title="赛季惨案">
                <FunMatchLine match={fun.blowout} />
              </FunCard>
            )}
            {fun.streakKing && (
              <FunCard icon={Zap} title="赛季连胜王">
                <div className="font-bold text-card-foreground">
                  {fun.streakKing.name}
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {fun.streakKing.streak} 连胜
                </div>
              </FunCard>
            )}
            {fun.upset && (
              <FunCard icon={Sparkles} title="赛季最大冷门">
                <FunMatchLine match={fun.upset} />
                <div className="mt-1 text-xs font-medium text-win">
                  胜率仅 {Math.round(fun.upset.winnerWinProb * 100)}%
                </div>
              </FunCard>
            )}
          </div>
        </section>
      )}
    </>
  );
}

function RankRow({
  rank,
  name,
  value,
  sub,
}: {
  rank: number;
  name: string;
  value: React.ReactNode;
  sub?: string;
}) {
  const medalColors = [
    "bg-amber-100 text-amber-700 ring-amber-200 dark:bg-amber-400/15 dark:text-amber-300 dark:ring-amber-400/30",
    "bg-slate-100 text-slate-700 ring-slate-200 dark:bg-slate-400/15 dark:text-slate-300 dark:ring-slate-400/30",
    "bg-orange-100 text-orange-800 ring-orange-200 dark:bg-orange-400/15 dark:text-orange-300 dark:ring-orange-400/30",
  ];
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-border bg-card p-3 shadow-sm">
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold ring-1 ${
          medalColors[rank - 1] ?? "bg-muted text-muted-foreground"
        }`}
      >
        {rank}
      </div>
      <PlayerAvatar name={name} size="xs" />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium text-card-foreground">
          {name}
        </span>
        {sub && (
          <span className="block text-[10px] text-muted-foreground">{sub}</span>
        )}
      </span>
      <span className="text-sm tabular-nums font-semibold text-card-foreground">
        {value}
      </span>
    </div>
  );
}

function FunCard({
  icon: Icon,
  title,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
      <div className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-muted-foreground">
        <Icon className="size-4" />
        {title}
      </div>
      {children}
    </div>
  );
}

function FunMatchLine({ match }: { match: FunMatch }) {
  const aWon = match.scoreA > match.scoreB;
  const winTeam = aWon ? match.teamA : match.teamB;
  const loseTeam = aWon ? match.teamB : match.teamA;
  const winScore = aWon ? match.scoreA : match.scoreB;
  const loseScore = aWon ? match.scoreB : match.scoreA;
  return (
    <div className="text-sm leading-snug text-card-foreground">
      <span className="font-bold">
        {winTeam[0]} / {winTeam[1]}
      </span>{" "}
      <span className="tabular-nums font-semibold">
        {winScore}:{loseScore}
      </span>{" "}
      <span className="text-muted-foreground">
        {loseTeam[0]} / {loseTeam[1]}
      </span>
      <div className="mt-1 text-xs text-muted-foreground">{match.date}</div>
    </div>
  );
}

function Empty() {
  return (
    <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
      本季无数据
    </div>
  );
}

/** asOfLocalDate（YYYY-MM-DD）→ 「M月d日」。 */
function formatAsOfLabel(localDate: string): string {
  const [, month, day] = localDate.split("-");
  return `${Number(month)}月${Number(day)}日`;
}
