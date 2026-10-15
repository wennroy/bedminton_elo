import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import {
  markPlayedService,
  toScheduleHttp,
} from "@/lib/schedule-service";

export const dynamic = "force-dynamic";

/** POST .../matches/[fixtureId]/played：pending → played_unscored（补「打过」事实）。 */
export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string; fixtureId: string }> }
) {
  const { id: idParam, fixtureId: fixtureIdParam } = await params;
  const projectId = Number(idParam);
  const fixtureId = Number(fixtureIdParam);
  if (
    !Number.isSafeInteger(projectId) ||
    projectId < 1 ||
    !Number.isSafeInteger(fixtureId) ||
    fixtureId < 1
  ) {
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
    const result = markPlayedService({
      projectId,
      fixtureId,
      expectedFixtureRevision: body.expectedFixtureRevision,
      actorId: body.actorId,
      playedAt: body.playedAt,
      db: getDb(),
    });
    // 合同：状态翻转不产生新资源 → 200（score 才是 201）。
    return NextResponse.json({ fixture: result.fixture }, { status: 200 });
  } catch (error) {
    const { status, body: errorBody } = toScheduleHttp(error);
    return NextResponse.json(errorBody, { status });
  }
}
