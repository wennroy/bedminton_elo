import { NextResponse } from "next/server";
import { addPlayer, listPlayers } from "@/lib/repo";

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

// Trust-based per design: anyone can add a player, no admin key required.
export async function POST(request: Request) {
  try {
    const body = await request.json().catch(() => null);
    const name =
      body && typeof body.name === "string" ? body.name.trim() : "";
    if (!name) {
      return NextResponse.json({ error: "名字不能为空" }, { status: 400 });
    }
    if (name.length > 20) {
      return NextResponse.json(
        { error: "名字最多 20 个字符" },
        { status: 400 }
      );
    }
    const id = addPlayer(name);
    return NextResponse.json({ id, name }, { status: 201 });
  } catch (error) {
    if (error instanceof Error && error.message.includes("UNIQUE constraint")) {
      return NextResponse.json(
        { error: "这个名字已存在，直接在列表里点选即可" },
        { status: 409 }
      );
    }
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
