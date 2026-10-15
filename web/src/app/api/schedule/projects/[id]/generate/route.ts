import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import {
  regenerateProjectService,
  toScheduleHttp,
} from "@/lib/schedule-service";

export const dynamic = "force-dynamic";

/** POST /api/schedule/projects/[id]/generate：续排（matches=0 合法，只保存名单）。 */
export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isSafeInteger(id) || id < 1) {
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
    const result = regenerateProjectService({
      projectId: id,
      playerIds: body.playerIds,
      matches: body.matches,
      avoidPlayedRematches: body.avoidPlayedRematches,
      seed: body.seed,
      expectedRevision: body.expectedRevision,
      actorId: body.actorId,
      requestId: body.requestId,
      db: getDb(),
    });
    return NextResponse.json(
      {
        project: result.project,
        names: result.names,
        generationId: result.generationId,
        fixtures: result.fixtures,
        supersededCount: result.supersededCount,
        joinedPlayers: result.joinedPlayers,
        leftPlayers: result.leftPlayers,
        replayed: result.replayed,
      },
      { status: 201 }
    );
  } catch (error) {
    const { status, body: errorBody } = toScheduleHttp(error);
    return NextResponse.json(errorBody, { status });
  }
}
