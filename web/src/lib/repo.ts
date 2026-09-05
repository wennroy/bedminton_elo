import type Database from "better-sqlite3";
import { getDb } from "./db";
import { recomputeElos, INITIAL_RATING, type Match as EloMatch } from "./elo";
import {
  recomputeTrueSkills,
  TS_MU,
  TS_SIGMA,
} from "./trueskill";

export interface Player {
  id: number;
  name: string;
  createdAt: string;
}

export interface Match {
  id: number;
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
  enteredBy: number | null;
  createdAt: string;
}

export interface MatchWithNames extends Match {
  pa1Name: string;
  pa2Name: string;
  pb1Name: string;
  pb2Name: string;
}

export interface PlayerRatings {
  elo: number;
  mu: number;
  sigma: number;
}

function resolveDb(db?: Database.Database): Database.Database {
  return db ?? getDb();
}

export function listPlayers(db?: Database.Database): Player[] {
  const conn = resolveDb(db);
  const rows = conn
    .prepare(
      `SELECT id, name, created_at AS createdAt FROM players ORDER BY name`
    )
    .all() as Player[];
  return rows;
}

export function addPlayer(name: string, db?: Database.Database): number {
  const conn = resolveDb(db);
  const result = conn.prepare(`INSERT INTO players (name) VALUES (?)`).run(name);
  return Number(result.lastInsertRowid);
}

export function renamePlayer(id: number, name: string, db?: Database.Database): void {
  const conn = resolveDb(db);
  conn.prepare(`UPDATE players SET name = ? WHERE id = ?`).run(name, id);
}

export function mergePlayers(fromId: number, toId: number, db?: Database.Database): void {
  const conn = resolveDb(db);
  const tx = conn.transaction(() => {
    for (const col of ["pa1", "pa2", "pb1", "pb2", "entered_by"] as const) {
      conn.prepare(`UPDATE matches SET ${col} = ? WHERE ${col} = ?`).run(
        toId,
        fromId
      );
    }
    conn.prepare(`DELETE FROM players WHERE id = ?`).run(fromId);
  });
  tx();
}

export interface AddMatchInput {
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
  enteredBy?: number | null;
}

export function addMatch(input: AddMatchInput, db?: Database.Database): number {
  const ids = [input.pa1, input.pa2, input.pb1, input.pb2];
  if (new Set(ids).size !== 4) {
    throw new Error("Four players must be distinct");
  }
  if (input.scoreA === input.scoreB) {
    throw new Error("Scores must not be equal");
  }
  const conn = resolveDb(db);
  const result = conn
    .prepare(
      `INSERT INTO matches (pa1, pa2, pb1, pb2, score_a, score_b, played_at, entered_by)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      input.pa1,
      input.pa2,
      input.pb1,
      input.pb2,
      input.scoreA,
      input.scoreB,
      input.playedAt,
      input.enteredBy ?? null
    );
  return Number(result.lastInsertRowid);
}

export function getMatch(id: number, db?: Database.Database): MatchWithNames | undefined {
  const conn = resolveDb(db);
  const row = conn
    .prepare(
      `SELECT
        m.id,
        m.pa1,
        m.pa2,
        m.pb1,
        m.pb2,
        m.score_a AS scoreA,
        m.score_b AS scoreB,
        m.played_at AS playedAt,
        m.entered_by AS enteredBy,
        m.created_at AS createdAt,
        a1.name AS pa1Name,
        a2.name AS pa2Name,
        b1.name AS pb1Name,
        b2.name AS pb2Name
      FROM matches m
      JOIN players a1 ON m.pa1 = a1.id
      JOIN players a2 ON m.pa2 = a2.id
      JOIN players b1 ON m.pb1 = b1.id
      JOIN players b2 ON m.pb2 = b2.id
      WHERE m.id = ?`
    )
    .get(id) as MatchWithNames | undefined;
  return row;
}

export function listMatchesByDate(db?: Database.Database): MatchWithNames[] {
  const conn = resolveDb(db);
  const rows = conn
    .prepare(
      `SELECT
        m.id,
        m.pa1,
        m.pa2,
        m.pb1,
        m.pb2,
        m.score_a AS scoreA,
        m.score_b AS scoreB,
        m.played_at AS playedAt,
        m.entered_by AS enteredBy,
        m.created_at AS createdAt,
        a1.name AS pa1Name,
        a2.name AS pa2Name,
        b1.name AS pb1Name,
        b2.name AS pb2Name
      FROM matches m
      JOIN players a1 ON m.pa1 = a1.id
      JOIN players a2 ON m.pa2 = a2.id
      JOIN players b1 ON m.pb1 = b1.id
      JOIN players b2 ON m.pb2 = b2.id
      ORDER BY m.played_at, m.created_at, m.id`
    )
    .all() as MatchWithNames[];
  return rows;
}

export function deleteMatch(id: number, db?: Database.Database): void {
  const conn = resolveDb(db);
  conn.prepare(`DELETE FROM matches WHERE id = ?`).run(id);
}

export function recomputeAllRatings(db?: Database.Database): Map<number, PlayerRatings> {
  const matches = listMatchesByDate(db);
  const eloMatches: EloMatch[] = matches.map((m) => ({
    date: m.playedAt,
    a1: String(m.pa1),
    a2: String(m.pa2),
    b1: String(m.pb1),
    b2: String(m.pb2),
    scoreA: m.scoreA,
    scoreB: m.scoreB,
  }));

  const { ratings: eloRatings } = recomputeElos(eloMatches);
  const { players: tsPlayers } = recomputeTrueSkills(eloMatches);

  const playerIds = new Set<number>();
  for (const m of matches) {
    playerIds.add(m.pa1);
    playerIds.add(m.pa2);
    playerIds.add(m.pb1);
    playerIds.add(m.pb2);
  }

  const result = new Map<number, PlayerRatings>();
  for (const id of playerIds) {
    const ts = tsPlayers[String(id)];
    result.set(id, {
      elo: eloRatings[String(id)] ?? INITIAL_RATING,
      mu: ts?.mu ?? TS_MU,
      sigma: ts?.sigma ?? TS_SIGMA,
    });
  }
  return result;
}
