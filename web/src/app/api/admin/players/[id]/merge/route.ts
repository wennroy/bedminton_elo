import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { isAdminKey } from "@/lib/admin";
import { mergePlayerWithScheduleRefs } from "@/lib/schedule-service";
import { revalidateRatingPages } from "@/lib/rating-revalidation";

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
    // 排阵联动合并：同事务预检（目标存在/同场双人对撞/同项目双人成员），
    // 无冲突时迁移 matches + 全部 schedule_* 引用并 bump 受影响项目。
    const result = mergePlayerWithScheduleRefs(id, toId, db);
    revalidateRatingPages();
    return NextResponse.json({ success: true, ...result });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
