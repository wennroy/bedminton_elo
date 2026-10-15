import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { deleteMatch, listMatchesByDate } from "@/lib/repo";
import {
  MatchValidationError,
  type MatchValidationInput,
} from "@/lib/match-validation";
import {
  recordMatch,
  type LegacyRatingEntry,
  type MatchRatingField,
} from "@/lib/match-recording";
import { revalidateRatingPages } from "@/lib/rating-revalidation";

export type { MatchRatingField };

const TEN_MINUTES = 10 * 60 * 1000;

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const db = getDb();
    const matches = listMatchesByDate(db);
    return NextResponse.json(matches);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

interface PostBody extends MatchValidationInput {
  enteredBy?: number | null;
}

function parsePostBody(body: Record<string, unknown>): PostBody | null {
  const pa1 = Number(body.pa1);
  const pa2 = Number(body.pa2);
  const pb1 = Number(body.pb1);
  const pb2 = Number(body.pb2);
  const scoreA = Number(body.scoreA);
  const scoreB = Number(body.scoreB);
  const playedAt = body.playedAt;
  const enteredBy =
    body.enteredBy === undefined || body.enteredBy === null
      ? null
      : Number(body.enteredBy);

  if (
    [pa1, pa2, pb1, pb2, scoreA, scoreB].some((n) => !Number.isFinite(n)) ||
    typeof playedAt !== "string" ||
    (enteredBy !== null && !Number.isFinite(enteredBy))
  ) {
    return null;
  }
  return { pa1, pa2, pb1, pb2, scoreA, scoreB, playedAt, enteredBy };
}

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const input = parsePostBody(body);
  if (!input) {
    return NextResponse.json({ error: "Invalid payload" }, { status: 400 });
  }

  let db: ReturnType<typeof getDb>;
  try {
    db = getDb();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  // 校验失败（含球员目录存在性）→ 400，不落库；写入失败 → 5xx。
  // 评分结果构建失败在 recordMatch 内部降级，绝不影响已保存的比赛。
  let result: {
    id: number;
    before: LegacyRatingEntry[];
    after: LegacyRatingEntry[];
    legacyRatingsAvailable: boolean;
    rating: MatchRatingField;
  };
  try {
    result = recordMatch({
      pa1: input.pa1,
      pa2: input.pa2,
      pb1: input.pb1,
      pb2: input.pb2,
      scoreA: input.scoreA,
      scoreB: input.scoreB,
      playedAt: input.playedAt,
      enteredBy: input.enteredBy,
      db,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    if (error instanceof MatchValidationError) {
      return NextResponse.json({ error: message }, { status: 400 });
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }

  return NextResponse.json(
    {
      id: result.id,
      before: result.before,
      after: result.after,
      legacyRatingsAvailable: result.legacyRatingsAvailable,
      rating: result.rating,
    },
    { status: 201 }
  );
}

export async function DELETE(request: Request) {
  const url = new URL(request.url);
  const idParam = url.searchParams.get("id");
  const id = idParam ? Number(idParam) : NaN;
  if (!Number.isFinite(id)) {
    return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  }

  try {
    const db = getDb();
    const row = db
      .prepare("SELECT created_at AS createdAt FROM matches WHERE id = ?")
      .get(id) as { createdAt: string } | undefined;
    if (!row) {
      return NextResponse.json({ error: "Match not found" }, { status: 404 });
    }

    const createdAt = new Date(`${row.createdAt}Z`).getTime();
    const elapsed = Date.now() - createdAt;
    const adminKey = request.headers.get("x-admin-key");
    const adminPassword = process.env.ADMIN_PASSWORD;
    if (elapsed >= TEN_MINUTES && adminKey !== adminPassword) {
      return NextResponse.json(
        { error: "Cannot delete match after 10 minutes" },
        { status: 403 }
      );
    }

    // 走统一撤回服务：被排阵 fixture 引用时同事务解除关联再删除。
    deleteMatch(id, db);
    revalidateRatingPages();
    return NextResponse.json({ success: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
