import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb } from "@/lib/db";
import { GET, POST } from "./route";

function createRequest(body: object): Request {
  return new Request("http://localhost/api/players", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe.sequential("players API", () => {
  let dbPath: string;

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-players-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
  });

  afterEach(() => {
    closeDb();
    try {
      unlinkSync(dbPath);
    } catch {
      // ignore
    }
  });

  it("creates a player and trims the name", async () => {
    const res = await POST(createRequest({ name: " 小明 " }));
    expect(res.status).toBe(201);
    const data = await res.json();
    expect(data.name).toBe("小明");
    expect(typeof data.id).toBe("number");

    const list = await (await GET()).json();
    expect(list.map((p: { name: string }) => p.name)).toContain("小明");
  });

  it("rejects empty or blank names", async () => {
    for (const body of [{ name: "" }, { name: "   " }, {}, { name: 42 }]) {
      const res = await POST(createRequest(body));
      expect(res.status).toBe(400);
    }
  });

  it("rejects names over 20 characters", async () => {
    const res = await POST(createRequest({ name: "很长的名字".repeat(5) }));
    expect(res.status).toBe(400);
  });

  it("rejects duplicate names with 409", async () => {
    const first = await POST(createRequest({ name: "重名" }));
    expect(first.status).toBe(201);
    const dup = await POST(createRequest({ name: "重名" }));
    expect(dup.status).toBe(409);
    const data = await dup.json();
    expect(data.error).toContain("已存在");
  });
});
