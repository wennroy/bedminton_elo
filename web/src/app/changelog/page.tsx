import Link from "next/link";
import { readChangelog, type ChangelogEntry } from "@/lib/changelog";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";

export const dynamic = "force-dynamic";

export default function ChangelogPage() {
  const entries = readChangelog();

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <div className="mb-4 flex items-center gap-2">
        <Button variant="ghost" size="icon-sm" asChild>
          <Link href="/me" aria-label="返回">
            <ChevronLeft className="size-5" />
          </Link>
        </Button>
        <h1 className="text-xl font-bold text-foreground">更新日志</h1>
      </div>

      {entries.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          暂无更新记录
        </div>
      ) : (
        <div>
          {entries.map((entry, index) => (
            <TimelineItem
              key={entry.version}
              entry={entry}
              isLast={index === entries.length - 1}
            />
          ))}
        </div>
      )}
    </main>
  );
}

function TimelineItem({
  entry,
  isLast,
}: {
  entry: ChangelogEntry;
  isLast: boolean;
}) {
  const unreleased = entry.version === "Unreleased";
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <span
          className={`mt-1.5 h-3 w-3 shrink-0 rounded-full ring-4 ring-background ${
            unreleased ? "bg-amber-500" : "bg-primary"
          }`}
        />
        {!isLast && <span className="w-0.5 flex-1 bg-border" />}
      </div>
      <div className="flex-1 pb-6">
        <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
          <div className="mb-2 flex items-center justify-between gap-2">
            {unreleased ? (
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-800">
                未发布
              </span>
            ) : (
              <span className="font-bold text-card-foreground">
                v{entry.version}
              </span>
            )}
            {entry.date && (
              <span className="text-xs text-muted-foreground">{entry.date}</span>
            )}
          </div>
          {entry.whatsNew.length > 0 && (
            <ul className="space-y-1.5">
              {entry.whatsNew.map((item, i) => (
                <li
                  key={i}
                  className="flex gap-2 text-sm text-card-foreground"
                >
                  <span className="mt-[9px] h-1 w-1 shrink-0 rounded-full bg-muted-foreground" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
