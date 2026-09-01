import type { PlayerFunStats } from "@/lib/stats";
import {
  Flame,
  Trophy,
  HeartHandshake,
  Skull,
  Mountain,
  Scale,
} from "lucide-react";

interface FunStatsProps {
  stats: PlayerFunStats;
}

export function FunStats({ stats }: FunStatsProps) {
  const streakText =
    stats.currentStreakType === "none"
      ? "—"
      : `${stats.currentStreak} 连${stats.currentStreakType === "win" ? "胜" : "败"}`;
  const streakColor =
    stats.currentStreakType === "win"
      ? "text-emerald-600"
      : stats.currentStreakType === "loss"
        ? "text-rose-600"
        : "text-card-foreground";

  const diff = stats.avgPointDiff;
  const diffText = `${diff > 0 ? "+" : ""}${diff.toFixed(1)}`;
  const diffColor =
    diff > 0 ? "text-emerald-600" : diff < 0 ? "text-rose-600" : "text-card-foreground";

  return (
    <div className="grid grid-cols-2 gap-3">
      <StatCard
        icon={<Flame className="size-4" />}
        label="当前势头"
        value={streakText}
        valueClass={streakColor}
      />
      <StatCard
        icon={<Trophy className="size-4" />}
        label="最长连胜"
        value={`${stats.longestWinStreak} 连胜`}
      />
      <StatCard
        icon={<HeartHandshake className="size-4" />}
        label="黄金搭档"
        value={stats.bestPartner?.name ?? "数据不足"}
        sub={
          stats.bestPartner
            ? `${stats.bestPartner.winRate}%（${stats.bestPartner.wins}/${stats.bestPartner.total} 场）`
            : "搭档满 3 场解锁"
        }
      />
      <StatCard
        icon={<Skull className="size-4" />}
        label="头号克星"
        value={stats.nemesis?.name ?? "数据不足"}
        sub={
          stats.nemesis
            ? `胜率 ${stats.nemesis.winRate}%（${stats.nemesis.wins} 胜 ${stats.nemesis.losses} 负）`
            : "交手满 3 场解锁"
        }
      />
      <StatCard
        icon={<Mountain className="size-4" />}
        label="ELO 峰值"
        value={String(stats.peakElo)}
        sub={stats.peakEloDate ? `${stats.peakEloDate} 达成` : undefined}
      />
      <StatCard
        icon={<Scale className="size-4" />}
        label="场均净胜分"
        value={diffText}
        valueClass={diffColor}
      />
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  sub,
  valueClass = "text-card-foreground",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  valueClass?: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-2xl border border-border bg-card p-3 shadow-sm">
      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {icon}
        {label}
      </div>
      <span className={`text-lg font-bold tabular-nums ${valueClass}`}>
        {value}
      </span>
      {sub && <span className="text-[10px] text-muted-foreground">{sub}</span>}
    </div>
  );
}
