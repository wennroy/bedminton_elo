import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import { getDb } from "@/lib/db";
import { isAdminKey } from "@/lib/admin";
import { mergePlayers } from "@/lib/repo";

export async function POST(
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

  const toId = Number(body.toId);
  if (!Number.isFinite(toId)) {
    return NextResponse.json({ error: "Invalid toId" }, { status: 400 });
  }
  if (id === toId) {
    return NextResponse.json(
      { error: "Cannot merge a player into itself" },
      { status: 400 }
    );
  }

  try {
    const db = getDb();
    mergePlayers(id, toId, db);
    revalidatePath("/");
    return NextResponse.json({ success: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
