import { NextResponse } from "next/server";
import { getDb } from "@/lib/db";
import { readRatingConfig } from "@/lib/rating-config";
import { loadGlickoSnapshot, type RatingServiceResult } from "@/lib/rating-service";
import { predictDoubles } from "@/lib/ratings/doubles";
import type { RatingModel } from "@/lib/ratings/types";
import { listPlayers, recomputeAllRatings } from "@/lib/repo";
import { optimizeSchedule, type ScheduledMatch } from "@/lib/scheduler";
import { createPlayer } from "@/lib/trueskill";
import { predictElo } from "@/lib/elo";

interface PostBody {
  playerIds: number[];
  matches: number;
  seed?: number;
  lambda?: number;
  model?: string;
}

interface ScheduleMatchOutput extends ScheduledMatch {
  winRate: number;
}

/** 响应的模型元信息：glicko2 取服务值，legacy 无版本化配置则为 null。 */
interface ScheduleModelMeta {
  model: RatingModel;
  configVersion: string | null;
  asOf: string | null;
  inputHash: string | null;
}

function parsePostBody(body: Record<string, unknown>): PostBody | null {
  const rawIds = body.playerIds;
  const matches = Number(body.matches);
  const seed = body.seed === undefined ? 42 : Number(body.seed);
  const lambda = body.lambda === undefined ? 0.5 : Number(body.lambda);
  const rawModel = body.model ?? body.rating;
  const model = typeof rawModel === "string" ? rawModel : undefined;

  if (!Array.isArray(rawIds) || rawIds.some((id) => !Number.isFinite(Number(id)))) {
    return null;
  }
  const playerIds = rawIds.map((id) => Number(id));

  if (
    playerIds.length < 4 ||
    new Set(playerIds).size !== playerIds.length ||
    !Number.isFinite(matches) ||
    matches < 1 ||
    !Number.isFinite(seed) ||
    !Number.isFinite(lambda) ||
    lambda < 0 ||
    lambda > 1
  ) {
    return null;
  }

  return { playerIds, matches, seed, lambda, model };
}

/** 显式 "glicko2"|"legacy" 优先；非法值/缺省回 activeModel（无配置默认 legacy）。 */
function resolveModel(requested: string | undefined): RatingModel {
  if (requested === "glicko2" || requested === "legacy") return requested;
  return readRatingConfig()?.activeModel ?? "legacy";
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

  const seed = input.seed!;
  const lambda = input.lambda!;

  const players = listPlayers();
  const playerMap = new Map(players.map((p) => [p.id, p]));
  for (const id of input.playerIds) {
    if (!playerMap.has(id)) {
      return NextResponse.json(
        { error: `Unknown player id ${id}` },
        { status: 400 }
      );
    }
  }

  const stringIds = input.playerIds.map(String);
  const names = Object.fromEntries(
    input.playerIds.map((id) => [String(id), playerMap.get(id)!.name])
  );
  const model = resolveModel(input.model);

  // glicko2：优化回调与输出胜率共用同一 ready 快照（同一 asOf、同一
  // replay.current 工作状态），绝不优化一套显示另一套；未知球员（目录有
  // 但未参赛）由引擎按初值处理。stale/unavailable 不伪造胜率、不静默退回。
  if (model === "glicko2") {
    const conn = getDb();
    const asOf = new Date().toISOString();
    // DB 层异常（连接/读取失败）也塑造成结构化 409，不外泄未成形 500。
    let snapshot: RatingServiceResult;
    try {
      snapshot = loadGlickoSnapshot(conn, asOf);
    } catch (error) {
      return NextResponse.json(
        {
          error: "glicko2 rating service unavailable",
          state: "unavailable",
          reason: `rating service failed: ${
            error instanceof Error ? error.message : String(error)
          }`,
        },
        { status: 409 }
      );
    }
    if (snapshot.state !== "ready") {
      const detail =
        snapshot.state === "stale"
          ? `${snapshot.reason}; last good as of ${snapshot.lastGood.asOf}`
          : snapshot.reason;
      return NextResponse.json(
        {
          error: "glicko2 rating service unavailable",
          state: snapshot.state,
          reason: detail,
        },
        { status: 409 }
      );
    }
    const record = readRatingConfig(conn);
    if (record === null) {
      return NextResponse.json(
        {
          error: "glicko2 rating service unavailable",
          state: "unavailable",
          reason: "rating config disappeared after ready snapshot",
        },
        { status: 409 }
      );
    }
    const states = snapshot.replay.current;
    const config = record.config;
    // 优化与展示共用同一个回调：同阵容下预测 DTO、优化回调与 API 概率一致。
    const winProbability = (match: ScheduledMatch) =>
      predictDoubles(
        [Number(match.a1), Number(match.a2)],
        [Number(match.b1), Number(match.b2)],
        states,
        config
      );

    const result = optimizeSchedule({
      playerIds: stringIds,
      matches: input.matches,
      players: [],
      seed,
      lambda,
      winProbability,
    });

    const schedule: ScheduleMatchOutput[] = result.schedule.map((match) => ({
      ...match,
      winRate: winProbability(match),
    }));
    const meta: ScheduleModelMeta = {
      model: "glicko2",
      configVersion: snapshot.replay.configVersion,
      asOf,
      inputHash: snapshot.inputHash,
    };
    return NextResponse.json({ schedule, metrics: result.metrics, names, ...meta });
  }

  // legacy（默认）：TrueSkill 优化 + predictElo 展示胜率，旧语义逐比特保留。
  const ratings = recomputeAllRatings();
  const eloRatings: Record<string, number> = Object.fromEntries(
    [...ratings].map(([id, r]) => [String(id), r.elo])
  );
  const tsPlayers = input.playerIds.map((id) => {
    const r = ratings.get(id);
    return createPlayer(r?.mu ?? 25, r?.sigma ?? 8.333);
  });

  const result = optimizeSchedule({
    playerIds: stringIds,
    matches: input.matches,
    players: tsPlayers,
    seed,
    lambda,
  });

  const schedule: ScheduleMatchOutput[] = result.schedule.map((match) => {
    const winRate = predictElo(
      match.a1,
      match.a2,
      match.b1,
      match.b2,
      eloRatings
    ).teamAWin;
    return { ...match, winRate };
  });
  const meta: ScheduleModelMeta = {
    model: "legacy",
    configVersion: null,
    asOf: null,
    inputHash: null,
  };

  return NextResponse.json({ schedule, metrics: result.metrics, names, ...meta });
}
