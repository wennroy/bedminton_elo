import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("next/cache", () => ({
  revalidatePath: vi.fn(),
}));
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addPlayer } from "@/lib/repo";
import { POST } from "./route";
import { DELETE as deleteById } from "./[id]/route";

function createRequest(body: object): Request {
  return new Request("http://localhost/api/matches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function createDeleteRequest(
  id: number,
  headers?: Record<string, string>
): Request {
  return new Request(`http://localhost/api/matches/${id}`, {
    method: "DELETE",
    headers,
  });
}

function todayString(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

describe.sequential("matches API", () => {
  let dbPath: string;
  let players: number[] = [];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-badminton-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    process.env.ADMIN_PASSWORD = "admin-secret";
    players = [
      addPlayer("A"),
      addPlayer("B"),
      addPlayer("C"),
      addPlayer("D"),
    ];
  });

  afterEach(() => {
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
  });

  it("POST creates a match and returns elo deltas", async () => {
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
        enteredBy: players[0],
      })
    );

    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.id).toBeTypeOf("number");
    expect(json.before).toHaveLength(4);
    expect(json.after).toHaveLength(4);
    for (const item of json.before) {
      expect(item.elo).toBe(1000);
    }
    const winners = json.after.filter(
      (item: { id: number }) => item.id === players[0] || item.id === players[1]
    );
    const losers = json.after.filter(
      (item: { id: number }) => item.id === players[2] || item.id === players[3]
    );
    for (const w of winners) expect(w.elo).toBeGreaterThan(1000);
    for (const l of losers) expect(l.elo).toBeLessThan(1000);
  });

  it("POST rejects equal scores", async () => {
    const response = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 21,
        playedAt: todayString(),
      })
    );

    expect(response.status).toBe(400);
    const json = await response.json();
    expect(json.error).toContain("equal");
  });

  it("DELETE allows retraction within 10 minutes", async () => {
    const postResponse = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
      })
    );
    const { id } = await postResponse.json();

    const deleteResponse = await deleteById(createDeleteRequest(id), {
      params: Promise.resolve({ id: String(id) }),
    });

    expect(deleteResponse.status).toBe(200);
    const json = await deleteResponse.json();
    expect(json.success).toBe(true);
  });

  it("DELETE rejects retraction after 10 minutes", async () => {
    const postResponse = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
      })
    );
    const { id } = await postResponse.json();

    const db = getDb();
    db.prepare(
      "UPDATE matches SET created_at = datetime('now', '-11 minutes') WHERE id = ?"
    ).run(id);

    const deleteResponse = await deleteById(createDeleteRequest(id), {
      params: Promise.resolve({ id: String(id) }),
    });

    expect(deleteResponse.status).toBe(403);
    const json = await deleteResponse.json();
    expect(json.error).toContain("10 minutes");
  });

  it("DELETE bypasses window with admin key", async () => {
    const postResponse = await POST(
      createRequest({
        pa1: players[0],
        pa2: players[1],
        pb1: players[2],
        pb2: players[3],
        scoreA: 21,
        scoreB: 15,
        playedAt: todayString(),
      })
    );
    const { id } = await postResponse.json();

    const db = getDb();
    db.prepare(
      "UPDATE matches SET created_at = datetime('now', '-11 minutes') WHERE id = ?"
    ).run(id);

    const deleteResponse = await deleteById(
      createDeleteRequest(id, { "x-admin-key": "admin-secret" }),
      { params: Promise.resolve({ id: String(id) }) }
    );

    expect(deleteResponse.status).toBe(200);
    const json = await deleteResponse.json();
    expect(json.success).toBe(true);
  });
});
