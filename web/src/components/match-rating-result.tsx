"use client";

import * as React from "react";
import { EloDeltaCard, type EloDeltaPlayer } from "@/components/elo-delta-card";

/** POST 录分响应 rating 字段的客户端形状（MatchRatingField 三态）。 */
export interface RatingChange {
  playerId: number;
  before: { r: number; rd: number; volatility: number };
  after: { r: number; rd: number; volatility: number };
  delta: number;
}

export type MatchRatingFieldClient =
  | {
      state: "ready";
      model: "glicko2";
      configVersion: string;
      asOf: string;
      segmentId: string;
      estimate: {
        matchId: number;
        playedAt: string;
        preWinA: number;
        changes: RatingChange[];
      };
      historyRecomputed: boolean;
    }
  | { state: "pending"; model: "glicko2"; reason: string }
  | { state: "not_effective"; model: "glicko2"; reason: string };

type Slot = number | null;

/** not_effective 的面向球友说明（record-form 与项目记分共用）。 */
export function notEffectiveDetail(rating: { reason: string }): string {
  if (rating.reason.includes("not yet effective")) {
    return "该比赛日期晚于当前评分时点，尚未产生新版计分；未来比赛不展示涨跌，到对应时段结算后生效。";
  }
  if (rating.reason === "rating config not initialized") {
    return "新版评分未启用，本次记录未产生新版计分。";
  }
  return `该记录未产生新版计分：${rating.reason}`;
}

/** 成功弹层标题下的描述行。 */
export function ratingResultDescription(rating: MatchRatingFieldClient | null): string {
  if (rating === null) return "四位球员的 ELO 已更新";
  if (rating.state === "ready") {
    return "新版评分已更新（Estimated 预估，周一正式结算）";
  }
  if (rating.state === "pending") return "比赛已保存，新版积分暂未更新";
  return "该记录未产生新版计分";
}

/** ready：按提交槽位顺序把 estimate.changes 映射为展示行。 */
export function estimatePlayers(
  rating: Extract<MatchRatingFieldClient, { state: "ready" }>,
  slotIds: [Slot, Slot, Slot, Slot],
  playerMap: Map<number, { id: number; name: string }>
): EloDeltaPlayer[] {
  const byId = new Map(
    rating.estimate.changes.map((change) => [change.playerId, change])
  );
  const players: EloDeltaPlayer[] = [];
  for (const id of slotIds) {
    if (id === null) continue;
    const change = byId.get(id);
    if (!change) continue;
    players.push({
      id,
      name: playerMap.get(id)?.name ?? "?",
      before: change.before.r,
      after: change.after.r,
    });
  }
  return players;
}

interface MatchRatingResultProps {
  rating: MatchRatingFieldClient | null;
  /** legacy ELO before/after 行（legacyAvailable=false 时不展示、不伪造）。 */
  legacyPlayers: EloDeltaPlayer[];
  legacyAvailable: boolean;
  /** 提交时的四个槽位（顺序展示新版评分变化）。 */
  slotIds: [Slot, Slot, Slot, Slot];
  playerMap: Map<number, { id: number; name: string }>;
}

/**
 * 录分成功后的评分结果区：新版评分三态（ready/pending/not_effective）+
 * legacy ELO 前后参考。record-form 与项目记分页共用，保证口径一致。
 */
export function MatchRatingResult({
  rating,
  legacyPlayers,
  legacyAvailable,
  slotIds,
  playerMap,
}: MatchRatingResultProps) {
  return (
    <div className="flex flex-col gap-3">
      {rating?.state === "ready" ? (
        <>
          <EloDeltaCard
            players={estimatePlayers(rating, slotIds, playerMap)}
            caption={`新版评分 · Estimated 预估 · 模型 ${rating.configVersion}`}
          />
          {rating.historyRecomputed ? (
            <p className="text-center text-[10px] leading-relaxed text-muted-foreground">
              该场属于已结算的历史区段：补录已重算下游当前分，
              以上为该场自身的单场变化。
            </p>
          ) : null}
        </>
      ) : null}
      {rating?.state === "pending" ? (
        <p className="rounded-xl border border-dashed border-loss/50 bg-loss-bg p-3 text-xs leading-relaxed text-loss">
          比赛已保存，新版积分暂未更新（{rating.reason}
          ）。评分恢复后会自动更新，无需重复提交。
        </p>
      ) : null}
      {rating?.state === "not_effective" ? (
        <p className="rounded-xl border border-dashed border-border bg-muted/30 p-3 text-xs leading-relaxed text-muted-foreground">
          {notEffectiveDetail(rating)}
        </p>
      ) : null}
      {legacyAvailable && legacyPlayers.length > 0 ? (
        <EloDeltaCard
          players={legacyPlayers}
          caption={
            rating?.state === "ready" || rating?.state === "pending"
              ? "Legacy ELO 参考"
              : "ELO"
          }
        />
      ) : (
        <p className="text-center text-xs text-muted-foreground">
          旧版 ELO 前后对比暂不可用。
        </p>
      )}
    </div>
  );
}
