import { NextResponse } from "next/server";
import { revalidatePath } from "next/cache";
import { getDb } from "@/lib/db";
import { isAdminKey } from "@/lib/admin";

export async function DELETE(
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

  try {
    const db = getDb();
    const count = db
      .prepare(
        `SELECT COUNT(*) AS c FROM matches
         WHERE pa1 = ? OR pa2 = ? OR pb1 = ? OR pb2 = ?`
      )
      .get(id, id, id, id) as { c: number };
    if (count.c > 0) {
      return NextResponse.json(
        { error: "该球员有比赛记录，请先合并到其他球员" },
        { status: 400 }
      );
    }

    db.prepare(`DELETE FROM players WHERE id = ?`).run(id);
    revalidatePath("/");
    return NextResponse.json({ success: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
