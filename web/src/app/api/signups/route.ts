import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import {
  getActiveSessionDate,
  listSignups,
  signupSummary,
  upsertSignup,
  removeSignup,
} from "@/lib/signup";
import { listPlayers } from "@/lib/repo";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const sessionDate = getActiveSessionDate(new Date());
    const signups = listSignups(sessionDate);
    const { totalPeople } = signupSummary(sessionDate);
    return NextResponse.json({ sessionDate, signups, totalPeople });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

function parsePlayerId(body: Record<string, unknown>): number | null {
  const playerId = Number(body.playerId);
  return Number.isInteger(playerId) && playerId > 0 ? playerId : null;
}

function revalidateSignupPages() {
  revalidatePath("/signup");
  revalidatePath("/");
}

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const playerId = parsePlayerId(body);
  const partySize = Number(body.partySize);
  if (playerId === null || (partySize !== 1 && partySize !== 2)) {
    return NextResponse.json({ error: "Invalid payload" }, { status: 400 });
  }

  try {
    if (!listPlayers().some((p) => p.id === playerId)) {
      return NextResponse.json({ error: "Unknown player" }, { status: 400 });
    }
    upsertSignup(
      getActiveSessionDate(new Date()),
      playerId,
      partySize as 1 | 2
    );
    revalidateSignupPages();
    return NextResponse.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function DELETE(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const playerId = parsePlayerId(body);
  if (playerId === null) {
    return NextResponse.json({ error: "Invalid payload" }, { status: 400 });
  }

  try {
    removeSignup(getActiveSessionDate(new Date()), playerId);
    revalidateSignupPages();
    return NextResponse.json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
