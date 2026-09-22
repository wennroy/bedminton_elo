import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { isAdminKey } from "@/lib/admin";
import { getMatch } from "@/lib/repo";
import {
  assertValidScores,
  MatchValidationError,
} from "@/lib/match-validation";
import { revalidateRatingPages } from "@/lib/rating-revalidation";

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
  if ([scoreA, scoreB].some((n) => !Number.isFinite(n))) {
    return NextResponse.json(
      { error: "Scores must be non-negative integers" },
      { status: 400 }
    );
  }
  // 与比赛录入共用同一套整数比分规则（非负整数且不相等）。
  try {
    assertValidScores(scoreA, scoreB);
  } catch (error) {
    if (error instanceof MatchValidationError) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    throw error;
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
    revalidateRatingPages();
    return NextResponse.json({ success: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
