"use client";

import * as React from "react";
import { ChevronDown } from "lucide-react";

interface CollapsibleSectionProps {
  title: string;
  /** 标题右侧的小字摘要,如「本周 12 场」 */
  badge?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export function CollapsibleSection({
  title,
  badge,
  defaultOpen = true,
  children,
}: CollapsibleSectionProps) {
  const [open, setOpen] = React.useState(defaultOpen);

  return (
    <section className="overflow-hidden rounded-2xl border border-border bg-card shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
      >
        <span className="text-lg font-bold text-card-foreground">{title}</span>
        <span className="flex items-center gap-2">
          {badge ? (
            <span className="text-xs text-muted-foreground">{badge}</span>
          ) : null}
          <ChevronDown
            className={`size-4 text-muted-foreground transition-transform duration-200 ${
              open ? "rotate-180" : ""
            }`}
          />
        </span>
      </button>
      {open ? <div className="px-4 pb-4">{children}</div> : null}
    </section>
  );
}
