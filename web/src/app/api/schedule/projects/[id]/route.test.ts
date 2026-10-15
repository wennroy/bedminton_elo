import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("next/cache", () => ({
  revalidatePath: vi.fn(),
}));

import { randomUUID } from "node:crypto";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import { addPlayer } from "@/lib/repo";
import { createProjectService } from "@/lib/schedule-service";
import { GET, PATCH } from "./route";

function params(id: number) {
  return { params: Promise.resolve({ id: String(id) }) };
}

function patchRequest(id: number, body: object): Request {
  return new Request(`http://localhost/api/schedule/projects/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe.sequential("schedule project [id] API（详情 + 改名）", () => {
  let dbPath: string;
  let players: string[] = [];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-schedule-project-id-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    players = [
      String(addPlayer("A")),
      String(addPlayer("B")),
      String(addPlayer("C")),
      String(addPlayer("D")),
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

  function createProject() {
    return createProjectService({
      name: "详情项目",
      playerIds: players,
      matches: 2,
      seed: 42,
      model: "legacy",
      actorId: players[0],
      requestId: randomUUID(),
      db: getDb(),
    });
  }

  it("GET 200：revision、成员统计、有效/取代场次、最近代次选项", async () => {
    const created = createProject();
    const response = await GET(
      new Request(`http://localhost/api/schedule/projects/${created.project.id}`),
      params(created.project.id)
    );
    expect(response.status).toBe(200);
    const json = await response.json();
    expect(json.project.id).toBe(created.project.id);
    expect(json.project.revision).toBe(0);
    expect(json.project.members).toHaveLength(4);
    expect(json.project.fixtures).toHaveLength(2);
    expect(json.project.fixtures[0]).toMatchObject({
      displayNo: 1,
      status: "pending",
    });
    // 生成时的预测胜率随场次保存（legacy 默认同分 → 0.5）。
    expect(json.project.fixtures[0].generatedWinRate).toBeCloseTo(0.5, 12);
    expect(json.project.latestGeneration.input).toMatchObject({
      matches: 2,
      seed: 42,
    });
    expect(json.names[players[1]]).toBe("B");
  });

  it("GET 未知项目 404；非法 id 400", async () => {
    const missing = await GET(
      new Request("http://localhost/api/schedule/projects/9999"),
      params(9999)
    );
    expect(missing.status).toBe(404);
    expect((await missing.json()).code).toBe("not_found");

    const bad = await GET(
      new Request("http://localhost/api/schedule/projects/abc"),
      params(NaN)
    );
    expect(bad.status).toBe(400);
  });

  it("PATCH 200：改名 + revision bump；可同名", async () => {
    const created = createProject();
    const response = await PATCH(
      patchRequest(created.project.id, {
        name: "  新名字 ",
        expectedRevision: created.project.revision,
        actorId: players[0],
      }),
      params(created.project.id)
    );
    expect(response.status).toBe(200);
    const json = await response.json();
    expect(json).toMatchObject({ id: created.project.id, name: "新名字", revision: 1 });

    const detail = await GET(
      new Request(`http://localhost/api/schedule/projects/${created.project.id}`),
      params(created.project.id)
    );
    expect((await detail.json()).project.name).toBe("新名字");
  });

  it("PATCH revision CAS 冲突 409 携带当前 projectRevision", async () => {
    const created = createProject();
    const response = await PATCH(
      patchRequest(created.project.id, {
        name: "x",
        expectedRevision: created.project.revision + 1,
        actorId: players[0],
      }),
      params(created.project.id)
    );
    expect(response.status).toBe(409);
    const json = await response.json();
    expect(json.code).toBe("revision_conflict");
    expect(json.projectRevision).toBe(created.project.revision);
  });

  it("PATCH 非法 name / 缺 actorId → 400；未知项目 404", async () => {
    const created = createProject();
    const empty = await PATCH(
      patchRequest(created.project.id, {
        name: "   ",
        expectedRevision: created.project.revision,
        actorId: players[0],
      }),
      params(created.project.id)
    );
    expect(empty.status).toBe(400);

    const missing = await PATCH(
      patchRequest(9999, {
        name: "x",
        expectedRevision: 0,
        actorId: players[0],
      }),
      params(9999)
    );
    expect(missing.status).toBe(404);
  });
});
