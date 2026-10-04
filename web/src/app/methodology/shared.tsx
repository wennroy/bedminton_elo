import type { ReactNode } from "react";

/** 公式排版块：等宽字体 + 浅色底，避免引入 LaTeX 依赖。 */
export function Formula({ children }: { children: ReactNode }) {
  return (
    <pre className="overflow-x-auto rounded-xl bg-muted/50 p-4 font-mono text-xs leading-relaxed text-card-foreground">
      {children}
    </pre>
  );
}

export const tableHeaderCell =
  "px-3 pb-3 text-left align-top text-[10px] font-medium text-muted-foreground";
export const tableBodyCell =
  "border-t border-border px-3 py-2.5 align-top text-xs text-card-foreground";

/** 内容页章节卡片，沿用 trends 页的 section 版式。 */
export function Section({
  title,
  intro,
  children,
}: {
  title: string;
  intro?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-border bg-card p-5 min-[761px]:p-[25px]">
      <h2 className="mb-2 text-lg font-bold text-card-foreground">{title}</h2>
      {intro ? (
        <p className="mb-5 text-sm leading-relaxed text-muted-foreground">{intro}</p>
      ) : null}
      <div className="flex flex-col gap-5">{children}</div>
    </section>
  );
}

/** 章节内小节标题。 */
export function SubsectionTitle({ children }: { children: ReactNode }) {
  return (
    <h3 className="text-sm font-bold text-card-foreground">{children}</h3>
  );
}
