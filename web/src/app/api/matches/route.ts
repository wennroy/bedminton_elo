import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import { getDb } from "@/lib/db";
import {
  addMatch,
  listPlayers,
  listMatchesByDate,
  recomputeAllRatings,
  type PlayerRatings,
} from "@/lib/repo";

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

interface PostBody {
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
  enteredBy?: number | null;
}

function isValidDateString(value: unknown): value is string {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}$/.test(value);
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
    !isValidDateString(playedAt) ||
    (enteredBy !== null && !Number.isFinite(enteredBy))
  ) {
    return null;
  }
  return { pa1, pa2, pb1, pb2, scoreA, scoreB, playedAt, enteredBy };
}

function buildPlayerMap(db?: ReturnType<typeof getDb>) {
  const players = listPlayers(db);
  return new Map(players.map((p) => [p.id, p.name]));
}

function ratingsForPlayers(
  ratings: Map<number, PlayerRatings>,
  names: Map<number, string>,
  ids: number[]
) {
  return ids.map((id) => ({
    id,
    name: names.get(id) ?? "?",
    elo: ratings.get(id)?.elo ?? 1000,
  }));
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

  const ids = [input.pa1, input.pa2, input.pb1, input.pb2];
  if (new Set(ids).size !== 4) {
    return NextResponse.json(
      { error: "Four players must be distinct" },
      { status: 400 }
    );
  }
  if (input.scoreA === input.scoreB) {
    return NextResponse.json(
      { error: "Scores must not be equal" },
      { status: 400 }
    );
  }
  if (input.scoreA < 0 || input.scoreB < 0) {
    return NextResponse.json(
      { error: "Scores must be non-negative" },
      { status: 400 }
    );
  }

  try {
    const db = getDb();
    const names = buildPlayerMap(db);
    const before = recomputeAllRatings(db);
    const id = addMatch(
      {
        pa1: input.pa1,
        pa2: input.pa2,
        pb1: input.pb1,
        pb2: input.pb2,
        scoreA: input.scoreA,
        scoreB: input.scoreB,
        playedAt: input.playedAt,
        enteredBy: input.enteredBy,
      },
      db
    );
    const after = recomputeAllRatings(db);
    revalidatePath("/");

    return NextResponse.json(
      {
        id,
        before: ratingsForPlayers(before, names, ids),
        after: ratingsForPlayers(after, names, ids),
      },
      { status: 201 }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 400 });
  }
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

    db.prepare("DELETE FROM matches WHERE id = ?").run(id);
    revalidatePath("/");
    return NextResponse.json({ success: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
