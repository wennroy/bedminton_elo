import { listProjects } from "@/lib/schedule-projects";
import { ProjectList } from "./project-list";

export const dynamic = "force-dynamic";

export default async function SchedulePage() {
  // 服务器直读首页（首屏即完整内容），客户端按游标继续翻页；
  // 公共数据一律以 DB 为准，旧的 localStorage 配对缓存不会上传。
  const { projects, nextCursor } = listProjects({ limit: 30 });

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
          SCHEDULE PROJECTS
        </div>
        <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
          配对项目
        </h1>
        <p className="mt-2 text-xs text-muted-foreground max-[760px]:text-[11px]">
          共享的配对安排，所有人看到同一份数据：查看待打场次、标记已打、录入比分。
        </p>
      </div>
      <ProjectList initialProjects={projects} initialNextCursor={nextCursor} />
    </div>
  );
}
