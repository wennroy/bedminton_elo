import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { listProjects } from "@/lib/schedule-projects";
import {
  createProjectService,
  toScheduleHttp,
} from "@/lib/schedule-service";

export const dynamic = "force-dynamic";

/** GET /api/schedule/projects：cursor/limit（默认 20，上限 30）公共摘要列表。 */
export async function GET(request: Request) {
  const url = new URL(request.url);
  const cursorParam = url.searchParams.get("cursor");
  const limitParam = url.searchParams.get("limit");

  let cursor: number | undefined;
  if (cursorParam !== null) {
    cursor = Number(cursorParam);
    if (!Number.isSafeInteger(cursor) || cursor < 1) {
      return NextResponse.json(
        { error: "cursor 必须是 >= 1 的整数", code: "validation_error" },
        { status: 400 }
      );
    }
  }
  let limit: number | undefined;
  if (limitParam !== null) {
    limit = Number(limitParam);
    if (!Number.isSafeInteger(limit) || limit < 1) {
      return NextResponse.json(
        { error: "limit 必须是 >= 1 的整数", code: "validation_error" },
        { status: 400 }
      );
    }
  }

  try {
    const { projects, nextCursor } = listProjects(
      { cursor, limit },
      getDb()
    );
    return NextResponse.json({ projects, nextCursor });
  } catch (error) {
    const { status, body } = toScheduleHttp(error);
    return NextResponse.json(body, { status });
  }
}

/** POST /api/schedule/projects：创建项目并生成首批场次（201）。 */
export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  try {
    const result = createProjectService({ ...body, db: getDb() });
    return NextResponse.json(
      {
        project: result.project,
        names: result.names,
        replayed: result.replayed,
      },
      { status: 201 }
    );
  } catch (error) {
    const { status, body: errorBody } = toScheduleHttp(error);
    return NextResponse.json(errorBody, { status });
  }
}
