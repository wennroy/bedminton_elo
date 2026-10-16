import Link from "next/link";
import { BookOpen } from "lucide-react";
import { listPlayers } from "@/lib/repo";
import { NewProjectForm } from "./new-project-form";

export const dynamic = "force-dynamic";

export default function NewScheduleProjectPage() {
  const players = listPlayers();

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="flex items-center justify-between gap-3">
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
            NEW SCHEDULE PROJECT
          </div>
          <Link
            href="/methodology/pairing"
            title="配对方式说明"
            className="inline-flex items-center gap-1 rounded-lg border border-border bg-card px-2 py-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <BookOpen className="size-3" strokeWidth={1.65} />
            <span className="hidden min-[761px]:inline">配对方式</span>
          </Link>
        </div>
        <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
          新建配对项目
        </h1>
        <p className="mt-2 text-xs text-muted-foreground max-[760px]:text-[11px]">
          选择在场球员并安排接下来几场双打；项目保存在服务器上，所有人可见。
        </p>
      </div>
      <NewProjectForm players={players} />
    </div>
  );
}
