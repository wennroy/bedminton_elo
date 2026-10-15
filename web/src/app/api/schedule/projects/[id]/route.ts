import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { getProject } from "@/lib/schedule-projects";
import {
  buildNamesMap,
  renameProjectService,
  toScheduleHttp,
} from "@/lib/schedule-service";

export const dynamic = "force-dynamic";

function parseIdParam(idParam: string): number | null {
  const id = Number(idParam);
  if (!Number.isSafeInteger(id) || id < 1) return null;
  return id;
}

/** GET /api/schedule/projects/[id]：项目详情（含成员统计、有效/取代场次、最近代次选项）。 */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: idParam } = await params;
  const id = parseIdParam(idParam);
  if (id === null) {
    return NextResponse.json(
      { error: "Invalid id", code: "validation_error" },
      { status: 400 }
    );
  }

  try {
    const db = getDb();
    const project = getProject(id, db);
    return NextResponse.json({ project, names: buildNamesMap(db, project) });
  } catch (error) {
    const { status, body } = toScheduleHttp(error);
    return NextResponse.json(body, { status });
  }
}

/** PATCH /api/schedule/projects/[id]：仅改名（revision CAS，冲突 409 携带当前 revision）。 */
export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: idParam } = await params;
  const id = parseIdParam(idParam);
  if (id === null) {
    return NextResponse.json(
      { error: "Invalid id", code: "validation_error" },
      { status: 400 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  try {
    const result = renameProjectService({
      projectId: id,
      name: body.name,
      expectedRevision: body.expectedRevision,
      actorId: body.actorId,
      db: getDb(),
    });
    return NextResponse.json(result);
  } catch (error) {
    const { status, body: errorBody } = toScheduleHttp(error);
    return NextResponse.json(errorBody, { status });
  }
}
