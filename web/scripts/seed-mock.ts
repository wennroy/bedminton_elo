import { createDb, getDatabaseUrl } from "../src/lib/db";
import { addPlayer, addMatch, listPlayers, listMatchesByDate } from "../src/lib/repo";
import { mulberry32, type PRNG } from "../src/lib/scheduler";

const NAMES = [
  "张伟",
  "李娜",
  "王强",
  "刘洋",
  "陈静",
  "杨帆",
  "赵敏",
  "黄磊",
  "周婷",
  "吴磊",
];

const SEED = 42;

function sample<T>(arr: T[], k: number, rng: PRNG): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, k);
}

function pickScore(rng: PRNG): { a: number; b: number } {
  const useDeuce = rng() < 0.25;
  if (useDeuce) {
    const winner = Math.floor(rng() * 10) + 22; // 22..31
    const loser = winner - 2;
    return rng() < 0.5 ? { a: winner, b: loser } : { a: loser, b: winner };
  }
  const loser = Math.floor(rng() * 12) + 8; // 8..19
  return rng() < 0.5 ? { a: 21, b: loser } : { a: loser, b: 21 };
}

function main() {
  const force = process.argv.includes("--force");
  const db = createDb(getDatabaseUrl());
  try {
    const existingPlayers = listPlayers(db);
    const existingMatches = listMatchesByDate(db);
    if (!force && (existingPlayers.length > 0 || existingMatches.length > 0)) {
      console.error(
        "Database already contains data. Pass --force to overwrite or run with an empty database."
      );
      process.exit(1);
    }

    if (force) {
      db.prepare(`DELETE FROM matches`).run();
      db.prepare(`DELETE FROM players`).run();
    }

    const rng = mulberry32(SEED);
    const playerIds = NAMES.map((name) => addPlayer(name, db));

    const today = new Date();
    const startOfWeek = new Date(today);
    startOfWeek.setDate(today.getDate() - today.getDay());
    startOfWeek.setHours(0, 0, 0, 0);

    for (let week = 0; week < 8; week++) {
      const weekBase = new Date(startOfWeek);
      weekBase.setDate(startOfWeek.getDate() - (7 - week) * 7);
      for (const dayOffset of [2, 4]) {
        const date = new Date(weekBase);
        date.setDate(weekBase.getDate() + dayOffset);
        if (date > today) continue; // 不生成未来日期的比赛
        // 用本地时区拼 YYYY-MM-DD,避免 toISOString 的 UTC 偏移导致日期错一天
        const dateStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
        const matches = Math.floor(rng() * 2) + 9; // 9..10
        for (let i = 0; i < matches; i++) {
          const [a1, a2, b1, b2] = sample(playerIds, 4, rng);
          const score = pickScore(rng);
          const enteredBy = sample([a1, a2, b1, b2], 1, rng)[0];
          addMatch(
            {
              pa1: a1,
              pa2: a2,
              pb1: b1,
              pb2: b2,
              scoreA: score.a,
              scoreB: score.b,
              playedAt: dateStr,
              enteredBy,
            },
            db
          );
        }
      }
    }

    const count = listMatchesByDate(db).length;
    console.log(`Seeded ${count} mock matches.`);
  } finally {
    db.close();
  }
}

main();
