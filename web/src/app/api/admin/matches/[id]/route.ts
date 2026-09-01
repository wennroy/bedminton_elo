import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import { getDb } from "@/lib/db";
import { isAdminKey } from "@/lib/admin";
import { getMatch } from "@/lib/repo";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const adminKey = request.headers.get("x-admin-key");
  if (!isAdminKey(adminKey)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: idParam } = await params;
  const id = Number(idParam);
  if (!Number.isFinite(id)) {
    return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const scoreA = Number(body.scoreA);
  const scoreB = Number(body.scoreB);
  if (
    [scoreA, scoreB].some((n) => !Number.isFinite(n)) ||
    scoreA < 0 ||
    scoreB < 0
  ) {
    return NextResponse.json(
      { error: "Scores must be non-negative numbers" },
      { status: 400 }
    );
  }
  if (scoreA === scoreB) {
    return NextResponse.json(
      { error: "Scores must not be equal" },
      { status: 400 }
    );
  }

  try {
    const db = getDb();
    const match = getMatch(id, db);
    if (!match) {
      return NextResponse.json({ error: "Match not found" }, { status: 404 });
    }

    db.prepare(
      `UPDATE matches SET score_a = ?, score_b = ? WHERE id = ?`
    ).run(scoreA, scoreB, id);
    revalidatePath("/");
    return NextResponse.json({ success: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
