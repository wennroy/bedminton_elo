import { notFound } from "next/navigation";
import { getDb } from "@/lib/db";
import { listPlayers } from "@/lib/repo";
import { getProject } from "@/lib/schedule-projects";
import { buildNamesMap } from "@/lib/schedule-service";
import { ProjectDetailClient } from "./project-detail";

export const dynamic = "force-dynamic";

interface ScheduleProjectPageProps {
  params: Promise<{ projectId: string }>;
}

export default async function ScheduleProjectPage({
  params,
}: ScheduleProjectPageProps) {
  const { projectId } = await params;
  const id = Number(projectId);
  if (!Number.isSafeInteger(id) || id < 1) notFound();

  let project;
  try {
    project = getProject(id);
  } catch {
    notFound();
  }

  const db = getDb();
  const names = buildNamesMap(db, project);
  const directory = listPlayers();

  return (
    <ProjectDetailClient
      initialProject={project}
      initialNames={names}
      directory={directory}
    />
  );
}
