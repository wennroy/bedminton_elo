import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import {
  recordFixtureScoreService,
  toScheduleHttp,
} from "@/lib/schedule-service";

export const dynamic = "force-dynamic";

/** POST .../matches/[fixtureId]/score：录比分并挂到场次（201 + MatchRatingField）。 */
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
    const result = recordFixtureScoreService({
      projectId,
      fixtureId,
      scoreA: body.scoreA,
      scoreB: body.scoreB,
      expectedFixtureRevision: body.expectedFixtureRevision,
      actorId: body.actorId,
      playedAt: body.playedAt,
      db: getDb(),
    });
    return NextResponse.json(
      {
        id: result.id,
        fixture: result.fixture,
        before: result.before,
        after: result.after,
        legacyRatingsAvailable: result.legacyRatingsAvailable,
        rating: result.rating,
      },
      { status: 201 }
    );
  } catch (error) {
    const { status, body: errorBody } = toScheduleHttp(error);
    return NextResponse.json(errorBody, { status });
  }
}
