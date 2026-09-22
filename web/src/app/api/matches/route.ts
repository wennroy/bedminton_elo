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
import {
  assertValidMatchInput,
  MatchValidationError,
  type MatchValidationInput,
} from "@/lib/match-validation";
import {
  loadGlickoSnapshot,
  RATING_CONFIG_MISSING_REASON,
} from "@/lib/rating-service";
import { shanghaiLocalDateFromInstant } from "@/lib/ratings/calendar";
import type { MatchEstimate } from "@/lib/ratings/types";

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

/** POST 响应中的新版评分判别字段（共享契约：ready/pending/not_effective）。 */
export type MatchRatingField =
  | {
      state: "ready";
      model: "glicko2";
      configVersion: string;
      /** 本次请求捕获的评分时点，贯穿整场评分。 */
      asOf: string;
      segmentId: string;
      /** 该 matchId 自身的单场变化（补录导致的下游当前分变化不冒充该场变化）。 */
      estimate: MatchEstimate;
      /** 该场落在历史区段、触发下游重算（estimate.segmentId !== 当前区段）。 */
      historyRecomputed: boolean;
    }
  | { state: "pending"; model: "glicko2"; reason: string }
  | { state: "not_effective"; model: "glicko2"; reason: string };

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

/**
 * 写入成功后的新版评分判别：
 * - ready：服务 ready 且重放含该 matchId 的单场事件；
 * - not_effective：ready 但无该场事件（未来日期未生效），或新版配置未初始化；
 * - pending：服务失败（unavailable 计算失败 / stale），比赛已保存、积分稍后更新。
 */
function buildRatingField(
  db: ReturnType<typeof getDb>,
  asOf: string,
  matchId: number,
  playedAt: string
): MatchRatingField {
  let result;
  try {
    result = loadGlickoSnapshot(db, asOf);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return {
      state: "pending",
      model: "glicko2",
      reason: `rating service failed: ${message}`,
    };
  }

  if (result.state === "ready") {
    const estimate = result.replay.matchEstimates[String(matchId)];
    if (estimate === undefined) {
      const asOfDate = shanghaiLocalDateFromInstant(asOf);
      return {
        state: "not_effective",
        model: "glicko2",
        reason:
          playedAt > asOfDate
            ? `match date ${playedAt} is after as-of date ${asOfDate}; not yet effective`
            : "match did not produce a glicko2 rating event",
      };
    }
    return {
      state: "ready",
      model: "glicko2",
      configVersion: result.replay.configVersion,
      asOf,
      segmentId: estimate.segmentId,
      estimate,
      historyRecomputed:
        estimate.segmentId !== result.replay.currentSegment.id,
    };
  }

  if (result.state === "stale") {
    return {
      state: "pending",
      model: "glicko2",
      reason: `rating service stale: ${result.reason}`,
    };
  }

  if (result.reason === RATING_CONFIG_MISSING_REASON) {
    return {
      state: "not_effective",
      model: "glicko2",
      reason: RATING_CONFIG_MISSING_REASON,
    };
  }
  return { state: "pending", model: "glicko2", reason: result.reason };
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

  let db: ReturnType<typeof getDb>;
  try {
    db = getDb();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  // 验证（含球员目录存在性）：只有验证失败才返回 4xx。
  try {
    const knownIds = new Set(listPlayers(db).map((p) => p.id));
    assertValidMatchInput(input, knownIds);
  } catch (error) {
    if (error instanceof MatchValidationError) {
      return NextResponse.json({ error: error.message }, { status: 400 });
    }
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  // asOf 在入口捕获一次，贯穿本次评分。
  const asOf = new Date().toISOString();

  // Legacy before/after 是可失败的附加计算：不参加写入事务、不阻止写入。
  let names: Map<number, string> = new Map();
  let before: ReturnType<typeof ratingsForPlayers> = [];
  let after: ReturnType<typeof ratingsForPlayers> = [];
  let legacyRatingsAvailable = true;
  try {
    names = buildPlayerMap(db);
    before = ratingsForPlayers(recomputeAllRatings(db), names, ids);
  } catch {
    before = [];
    legacyRatingsAvailable = false;
  }

  // 数据库写入失败 → 5xx（已成功写入绝不在此后退回错误响应）。
  let id: number;
  try {
    id = addMatch(
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
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  try {
    after = ratingsForPlayers(recomputeAllRatings(db), names, ids);
  } catch {
    after = [];
    legacyRatingsAvailable = false;
  }

  const rating = buildRatingField(db, asOf, id, input.playedAt);
  revalidatePath("/");

  return NextResponse.json(
    {
      id,
      before,
      after,
      legacyRatingsAvailable,
      rating,
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

    db.prepare("DELETE FROM matches WHERE id = ?").run(id);
    revalidatePath("/");
    return NextResponse.json({ success: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
