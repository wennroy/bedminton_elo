import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { listPlayers } from "@/lib/repo";
import { loadPredictionView } from "@/lib/rating-view";

interface PostBody {
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  rating?: string;
}

function parsePostBody(body: Record<string, unknown>): PostBody | null {
  const pa1 = Number(body.pa1);
  const pa2 = Number(body.pa2);
  const pb1 = Number(body.pb1);
  const pb2 = Number(body.pb2);
  const rawRating = body.rating ?? body.model;
  const rating = typeof rawRating === "string" ? rawRating : undefined;

  if (![pa1, pa2, pb1, pb2].every((n) => Number.isFinite(n))) return null;
  // 四方必须互不相同：重复身份无法构成一场双打。
  if (new Set([pa1, pa2, pb1, pb2]).size !== 4) return null;
  return { pa1, pa2, pb1, pb2, rating };
}

/**
 * 预测路由：包装 loadPredictionView（2026-09-22 用户定：新版预测走服务端，
 * 同一 asOf 工作状态出胜率与赢/输模拟）。选 POST 而非 GET：(1) 与
 * /api/schedule 一致的四方 JSON 负载；(2) 提交时阵容可能已不同于 URL 预填，
 * body 避免查询串拼装/转义；(3) POST 响应不被浏览器/CDN 隐式缓存。
 * stale/unavailable 返回 409 并给出原因，不伪造胜率、不静默退回 Legacy。
 */
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

  const knownIds = new Set(listPlayers(db).map((p) => p.id));
  for (const id of [input.pa1, input.pa2, input.pb1, input.pb2]) {
    if (!knownIds.has(id)) {
      return NextResponse.json(
        { error: `Unknown player id ${id}` },
        { status: 400 }
      );
    }
  }

  let result;
  try {
    result = loadPredictionView({
      pa1: input.pa1,
      pa2: input.pa2,
      pb1: input.pb1,
      pb2: input.pb2,
      rating: input.rating,
      asOf: new Date().toISOString(),
      db,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  if (result.model === "glicko2" && result.freshness !== "fresh") {
    const detail =
      result.freshness === "stale"
        ? `${result.reason}; last good as of ${result.lastGoodAsOf}`
        : result.reason;
    return NextResponse.json(
      {
        error: "glicko2 rating service unavailable",
        state: result.freshness,
        reason: detail,
      },
      { status: 409 }
    );
  }

  return NextResponse.json(result);
}
