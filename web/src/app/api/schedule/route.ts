import { NextResponse } from "next/server";
import { listPlayers, recomputeAllRatings } from "@/lib/repo";
import { optimizeSchedule, type ScheduledMatch } from "@/lib/scheduler";
import { predictTeamOutcomeWin, createPlayer } from "@/lib/trueskill";

interface PostBody {
  playerIds: number[];
  matches: number;
  seed?: number;
  lambda?: number;
}

interface ScheduleMatchOutput extends ScheduledMatch {
  winRate: number;
}

function parsePostBody(body: Record<string, unknown>): PostBody | null {
  const rawIds = body.playerIds;
  const matches = Number(body.matches);
  const seed = body.seed === undefined ? 42 : Number(body.seed);
  const lambda = body.lambda === undefined ? 0.5 : Number(body.lambda);

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

  return { playerIds, matches, seed, lambda };
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

  const ratings = recomputeAllRatings();
  const stringIds = input.playerIds.map(String);
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
    const teamA = [match.a1, match.a2].map((id) => tsPlayers[stringIds.indexOf(id)]
    );
    const teamB = [match.b1, match.b2].map((id) => tsPlayers[stringIds.indexOf(id)]
    );
    const winRate = predictTeamOutcomeWin(teamA, teamB);
    return { ...match, winRate };
  });

  return NextResponse.json({
    schedule,
    metrics: result.metrics,
    names: Object.fromEntries(
      input.playerIds.map((id) => [String(id), playerMap.get(id)!.name])
    ),
  });
}
