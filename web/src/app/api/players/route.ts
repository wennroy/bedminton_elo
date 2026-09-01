import { NextResponse } from "next/server";
import { listPlayers } from "@/lib/repo";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const players = listPlayers();
    return NextResponse.json(players);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
