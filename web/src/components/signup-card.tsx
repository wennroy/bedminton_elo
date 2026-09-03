import Link from "next/link";
import { ChevronRight } from "lucide-react";
import {
  getActiveSessionDate,
  formatSessionDate,
  signupSummary,
} from "@/lib/signup";

export function SignupCard() {
  const sessionDate = getActiveSessionDate(new Date());
  const { count, totalPeople } = signupSummary(sessionDate);

  return (
    <Link
      href="/signup"
      className="flex items-center justify-between rounded-2xl border border-border bg-card p-4 shadow-sm transition-colors hover:bg-muted/30"
    >
      <div className="flex flex-col gap-1">
        <span className="font-medium text-card-foreground">
          🏸 周三局 · {formatSessionDate(sessionDate)} 18:00–20:00
        </span>
        <span className="text-xs text-muted-foreground">
          已报名 {count} 人（含小伙伴共 {totalPeople} 人）
        </span>
      </div>
      <span className="flex items-center text-sm font-medium text-primary">
        去报名
        <ChevronRight className="size-4" />
      </span>
    </Link>
  );
}
