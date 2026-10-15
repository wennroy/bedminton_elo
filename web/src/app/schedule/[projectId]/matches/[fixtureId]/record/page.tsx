import { notFound } from "next/navigation";
import { getDb } from "@/lib/db";
import { getProject } from "@/lib/schedule-projects";
import { buildNamesMap } from "@/lib/schedule-service";
import { FixtureRecordForm } from "./fixture-record-form";

export const dynamic = "force-dynamic";

interface FixtureRecordPageProps {
  params: Promise<{ projectId: string; fixtureId: string }>;
}

export default async function FixtureRecordPage({
  params,
}: FixtureRecordPageProps) {
  const { projectId, fixtureId } = await params;
  const pid = Number(projectId);
  const fid = Number(fixtureId);
  if (!Number.isSafeInteger(pid) || pid < 1 || !Number.isSafeInteger(fid) || fid < 1) {
    notFound();
  }

  let project;
  try {
    project = getProject(pid);
  } catch {
    notFound();
  }
  const fixture = [...project.fixtures, ...project.superseded].find(
    (f) => f.id === fid
  );
  if (!fixture) notFound();

  const names = buildNamesMap(getDb(), project);

  return (
    <div className="flex flex-col gap-5">
      <div>
        <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
          PROJECT MATCH
        </div>
        <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
          项目记分
        </h1>
        <p className="mt-2 text-xs text-muted-foreground max-[760px]:text-[11px]">
          {project.name} · 第 {fixture.displayNo} 场 · 阵容以项目记录为准
        </p>
      </div>
      <FixtureRecordForm
        projectId={pid}
        projectName={project.name}
        fixture={fixture}
        names={names}
      />
    </div>
  );
}
