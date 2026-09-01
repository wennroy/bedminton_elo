import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import { getDb } from "@/lib/db";

const TEN_MINUTES = 10 * 60 * 1000;

export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id: idParam } = await params;
  const id = Number(idParam);
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
