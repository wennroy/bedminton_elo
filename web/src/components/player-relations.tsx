"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { PlayerAvatar } from "@/components/player-avatar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { RelationRecord } from "@/lib/stats";
import { cn } from "@/lib/utils";

type Kind = "partners" | "opponents";

const KIND_META: Record<
  Kind,
  {
    title: string;
    heroCaption: string;
    rateCaption: string;
    unlock: string;
    dialogTitle: string;
    barColor: string;
  }
> = {
  partners: {
    title: "黄金搭档",
    heroCaption: "搭档胜率最高",
    rateCaption: "搭档胜率",
    unlock: "搭档满 3 场后解锁",
    dialogTitle: "搭档表现",
    barColor: "var(--chart)",
  },
  opponents: {
    title: "值得研究的对手",
    heroCaption: "对阵胜率最低",
    rateCaption: "对阵胜率",
    unlock: "交手满 3 场后解锁",
    dialogTitle: "对手交锋",
    barColor: "var(--team-b)",
  },
};

function RelationPanel({
  kind,
  playerName,
  entries,
}: {
  kind: Kind;
  playerName: string;
  /** 全量列表（含 <3 场），已按该面板口径排序 */
  entries: RelationRecord[];
}) {
  const [open, setOpen] = React.useState(false);
  const meta = KIND_META[kind];
  const qualified = entries.filter((e) => e.total >= 3);
  const hero = qualified[0];

  if (!hero) {
    return (
      <section className="rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
        <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
          {meta.title}
        </h2>
        <div className="py-[45px] text-center text-[13px] text-muted-foreground">
          {meta.unlock}
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
      <div className="mb-[22px] flex items-center justify-between gap-3.5 max-[760px]:mb-[18px]">
        <h2 className="text-lg font-bold text-card-foreground max-[760px]:text-[15px]">
          {meta.title}
        </h2>
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-win"
        >
          全部
          <ArrowRight className="size-[15px]" strokeWidth={1.65} />
        </button>
      </div>

      <Link
        href={`/players/${hero.id}`}
        className={cn(
          "mb-[15px] flex items-center justify-between gap-2.5 rounded-[10px] p-[17px_18px]",
          kind === "partners" ? "bg-win-bg" : "bg-secondary"
        )}
      >
        <span className="flex min-w-0 items-center gap-[11px]">
          <PlayerAvatar name={hero.name} size="xs" />
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold text-foreground">
              {hero.name}
            </span>
            <span className="block text-[10px] text-muted-foreground">
              {meta.heroCaption}
            </span>
          </span>
        </span>
        <span className="text-right">
          <span className="font-num text-[30px] leading-none text-foreground">
            {hero.winRate}
            <span className="text-base">%</span>
          </span>
          <span className="mt-[5px] block text-[10px] text-muted-foreground">
            {meta.rateCaption} · {hero.total} 场
          </span>
        </span>
      </Link>

      <div>
        {qualified.slice(0, 3).map((e) => (
          <div
            key={e.id}
            className="mt-3.5 grid grid-cols-[85px_1fr_50px_52px] items-center gap-2.5 text-[11px] max-[1190px]:grid-cols-[70px_1fr_35px_48px] max-[1190px]:gap-[7px]"
          >
            <Link
              href={`/players/${e.id}`}
              className="flex min-w-0 items-center gap-[7px] font-semibold text-foreground transition-colors hover:text-win"
            >
              <PlayerAvatar
                name={e.name}
                size="xs"
                className="size-[27px] text-[10px]"
              />
              <span className="truncate">{e.name}</span>
            </Link>
            <div
              className="h-[5px] overflow-hidden rounded bg-secondary"
              aria-hidden="true"
            >
              <div
                className="h-full rounded"
                style={{ width: `${e.winRate}%`, background: meta.barColor }}
              />
            </div>
            <span className="text-right font-num text-[15px] text-foreground">
              {e.winRate}%
            </span>
            <span className="text-right text-muted-foreground">
              {e.wins}胜 {e.losses}负
            </span>
          </div>
        ))}
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{meta.dialogTitle}</DialogTitle>
            <DialogDescription>
              {playerName}的生涯数据 · 胜率均为{playerName}一方的胜率。
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[55dvh] overflow-y-auto">
            {entries.map((e) => (
              <Link
                key={e.id}
                href={`/players/${e.id}`}
                onClick={() => setOpen(false)}
                className="flex items-center justify-between gap-3 border-t border-border py-3 first:border-t-0"
              >
                <span className="flex min-w-0 items-center gap-[11px]">
                  <PlayerAvatar name={e.name} size="xs" />
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-semibold text-foreground">
                      {e.name}
                    </span>
                    <span className="block text-[10px] text-muted-foreground">
                      {e.wins} 胜 {e.losses} 负 · {e.total} 场
                      {e.total < 3 ? " · 样本较少" : ""}
                    </span>
                  </span>
                </span>
                <span className="font-num text-xl text-foreground">
                  {e.winRate}%
                </span>
              </Link>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </section>
  );
}

/** 搭档与对手双面板（mock relationships）；胜率均为当前查看球员一方 */
export function PlayerRelations({
  playerName,
  partners,
  opponents,
}: {
  playerName: string;
  partners: RelationRecord[];
  opponents: RelationRecord[];
}) {
  return (
    <section>
      <div className="mb-[13px] flex items-center justify-between gap-3">
        <h2 className="text-base font-bold text-foreground">搭档与对手</h2>
        <small className="text-[11px] text-muted-foreground max-[760px]:text-[9px]">
          生涯数据 · 至少搭档 / 交手 3 场
        </small>
      </div>
      <div className="grid gap-[18px] min-[761px]:grid-cols-2 min-[761px]:gap-[22px]">
        <RelationPanel
          kind="partners"
          playerName={playerName}
          entries={partners}
        />
        <RelationPanel
          kind="opponents"
          playerName={playerName}
          entries={opponents}
        />
      </div>
    </section>
  );
}
