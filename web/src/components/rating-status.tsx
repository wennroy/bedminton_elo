import type { LoadRatingViewResult } from "@/lib/rating-view";
import { shanghaiWallClockFromInstant } from "@/lib/ratings/calendar";
import type { RatingStatus } from "@/lib/ratings/types";
import { cn } from "@/lib/utils";

/** asOf（ISO instant）→ 上海墙钟「M月D日 HH:MM」，服务端/客户端同一口径。 */
export function formatStatusInstant(iso: string): string {
  const { month, day, hours, minutes } = shanghaiWallClockFromInstant(iso);
  return `${month}月${day}日 ${hours}:${minutes}`;
}

/**
 * 三态输入，直接对应 LoadRatingViewResult 的三个分支；
 * legacy 分支不显示新版状态（组件返回 null）。
 */
export type RatingStatusInput =
  | { model: "legacy" }
  | {
      model: "glicko2";
      freshness: "fresh" | "stale";
      asOf: string;
      version: string;
    }
  | { model: "glicko2"; freshness: "unavailable"; reason: string };

/**
 * 从 loadRatingView 结果提取状态组件所需字段；legacy 的 view 含
 * Map 等不可序列化结构，不能整体传给客户端组件。
 */
export function toRatingStatusInput(
  result: LoadRatingViewResult
): RatingStatusInput {
  if (result.model === "legacy") return { model: "legacy" };
  if (result.freshness === "unavailable") {
    return {
      model: "glicko2",
      freshness: "unavailable",
      reason: result.reason,
    };
  }
  return {
    model: "glicko2",
    freshness: result.freshness,
    asOf: result.asOf,
    version: result.version,
  };
}

export interface RatingStatusBanner {
  tone: "ok" | "warn" | "error";
  title: string;
  /** 补充说明：stale 的旧时点语义 / unavailable 的失败原因。 */
  detail: string | null;
  /** 版本与更新时点行；unavailable 为 null。 */
  meta: string | null;
}

/** 新版评分三态 → 状态条文案；legacy 返回 null（不显示新版状态）。 */
export function ratingStatusBanner(
  input: RatingStatusInput
): RatingStatusBanner | null {
  if (input.model === "legacy") return null;
  if (input.freshness === "unavailable") {
    return {
      tone: "error",
      title: "新版评分暂不可用",
      detail: input.reason,
      meta: null,
    };
  }
  if (input.freshness === "fresh") {
    return {
      tone: "ok",
      title: "新版评分运行中",
      detail: null,
      // 版本全串（含引擎参数指纹）留在 API/ETag，UI 只显首段。
      meta: `模型 ${input.version.split("|")[0]} · 更新于 ${formatStatusInstant(input.asOf)}`,
    };
  }
  // stale：asOf 已如实为最后成功时点，不冒充当前。
  const at = formatStatusInstant(input.asOf);
  return {
    tone: "warn",
    title: "新版评分暂未更新",
    detail: `以上为 ${at} 最后成功结算的结果；评分恢复后会自动更新。`,
    // 版本全串（含引擎参数指纹）留在 API/ETag，UI 只显首段。
    meta: `模型 ${input.version.split("|")[0]} · 最后成功 ${at}`,
  };
}

export interface RatingStatusExplainer {
  status: RatingStatus;
  label: string;
  description: string;
}

/** Estimated / Final / 未评级：三种状态各自准确且不相同的说明。 */
export const RATING_STATUS_EXPLAINERS: ReadonlyArray<RatingStatusExplainer> = [
  {
    status: "estimated",
    label: "Estimated 预估",
    description:
      "本周内逐场即时估计的分数；周一正式结算时会按整周表现统一校准，可能上调或下调。",
  },
  {
    status: "final",
    label: "Final 正式",
    description: "每周一结算的正式分数，为该周结束后的最终结果。",
  },
  {
    status: "unrated",
    label: "未评级",
    description: "还没有产生有效比赛，暂无评分；首次参赛后开始计分。",
  },
];

const toneClass: Record<RatingStatusBanner["tone"], string> = {
  ok: "border-border",
  warn: "border-dashed border-loss/60",
  error: "border-loss bg-loss-bg",
};

/**
 * 新版评分状态条：fresh 正常 / stale 旧快照 / unavailable 显示原因；legacy 不渲染。
 * showLegend：是否附 Estimated/Final/未评级三行说明（默认 false，仅首页传入）。
 */
export function RatingStatus({
  showLegend = false,
  ...input
}: RatingStatusInput & { showLegend?: boolean }) {
  const banner = ratingStatusBanner(input);
  if (banner === null) return null;

  return (
    <section
      aria-label="评分状态"
      className={cn("rounded-2xl border p-3.5 text-xs", toneClass[banner.tone])}
    >
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span
          aria-hidden="true"
          className={cn(
            "size-1.5 shrink-0 rounded-full",
            banner.tone === "ok" ? "bg-win" : "bg-loss"
          )}
        />
        <span
          className={cn(
            "text-[13px] font-bold",
            banner.tone === "ok" ? "text-foreground" : "text-loss"
          )}
        >
          {banner.title}
        </span>
        {banner.meta ? (
          <span className="text-muted-foreground">{banner.meta}</span>
        ) : null}
      </div>
      {banner.detail ? (
        <p className="mt-1.5 text-muted-foreground">{banner.detail}</p>
      ) : null}
      {showLegend &&
      input.model === "glicko2" &&
      input.freshness !== "unavailable" ? (
        <ul className="mt-2.5 space-y-1.5 border-t border-border pt-2.5">
          {RATING_STATUS_EXPLAINERS.map((item) => (
            <li key={item.status} className="flex gap-2">
              <span className="shrink-0 font-medium text-card-foreground">
                {item.label}
              </span>
              <span className="text-muted-foreground">
                {item.description}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
