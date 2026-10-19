import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("next/cache", () => ({
  revalidatePath: vi.fn(),
}));

import { randomUUID } from "node:crypto";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb } from "@/lib/db";
import { addPlayer } from "@/lib/repo";
import { GET, POST } from "./route";

function postRequest(body: object): Request {
  return new Request("http://localhost/api/schedule/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function getRequest(query = ""): Request {
  return new Request(`http://localhost/api/schedule/projects${query}`, {
    method: "GET",
  });
}

describe.sequential("schedule projects API（列表 + 创建）", () => {
  let dbPath: string;
  let players: string[] = [];

  beforeEach(() => {
    closeDb();
    dbPath = join(tmpdir(), `test-schedule-projects-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    players = [
      String(addPlayer("A")),
      String(addPlayer("B")),
      String(addPlayer("C")),
      String(addPlayer("D")),
      String(addPlayer("E")),
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

  it("POST guo 创建八人固定分组，GET 返回模式，省略 seed 的请求可重放", async () => {
    const roster = [...players, ...["F", "G", "H"].map((name) => String(addPlayer(name)))];
    const payload = {mode: "guo", playerIds: roster, actorId: roster[0], requestId: randomUUID(), model: "legacy"};
    const first = await POST(postRequest(payload));
    expect(first.status).toBe(201);
    const json = await first.json();
    expect(json.project.guo.rounds[0].groupA).toHaveLength(4);
    expect(json.project.guo.rounds[0].groupB).toHaveLength(4);
    expect(json.project.fixtures).toHaveLength(6);
    const retry = await POST(postRequest(payload));
    expect((await retry.json()).replayed).toBe(true);
    const list = await GET(getRequest());
    expect((await list.json()).projects[0].mode).toBe("guo");
  });

  it("POST 201：创建项目并返回首批场次与姓名映射", async () => {
    const response = await POST(
      postRequest({
        name: "周五晚场",
        playerIds: players,
        matches: 3,
        seed: 42,
        model: "legacy",
        actorId: players[0],
        requestId: randomUUID(),
      })
    );

    expect(response.status).toBe(201);
    const json = await response.json();
    expect(json.replayed).toBe(false);
    expect(json.project.name).toBe("周五晚场");
    expect(json.project.model).toBe("legacy");
    expect(json.project.fixtures).toHaveLength(3);
    expect(json.project.fixtures.map((f: { displayNo: number }) => f.displayNo)).toEqual([1, 2, 3]);
    expect(Object.keys(json.names)).toHaveLength(5);
    expect(json.names[players[0]]).toBe("A");
  });

  it("POST 201 幂等重放：同 requestId 同 payload 返回已建项目", async () => {
    const payload = {
      name: "幂等",
      playerIds: players.slice(0, 4),
      matches: 2,
      seed: 1,
      model: "legacy",
      actorId: players[0],
      requestId: randomUUID(),
    };
    const first = await POST(postRequest(payload));
    expect(first.status).toBe(201);
    const firstJson = await first.json();

    const replay = await POST(postRequest(payload));
    expect(replay.status).toBe(201);
    const replayJson = await replay.json();
    expect(replayJson.replayed).toBe(true);
    expect(replayJson.project.id).toBe(firstJson.project.id);

    const conflict = await POST(postRequest({ ...payload, matches: 3 }));
    expect(conflict.status).toBe(409);
    const conflictJson = await conflict.json();
    expect(conflictJson.code).toBe("idempotency_conflict");
  });

  it.each([
    [{ playerIds: players.slice(0, 3) }, 400],
    [{ playerIds: [players[0], players[0], players[1], players[2]] }, 400],
    [{ playerIds: [players[0], players[1], players[2], "99999"] }, 400],
    [{ matches: 0 }, 400],
    [{ matches: 100 }, 400],
    [{ seed: -1 }, 400],
    [{ actorId: "not-a-player" }, 400],
    [{ actorId: "99999" }, 400],
    [{ requestId: "bad-uuid" }, 400],
    [{ model: "trueskill" }, 400],
  ])("POST 校验失败 %o → %d", async (overrides, status) => {
    const response = await POST(
      postRequest({
        name: "校验",
        playerIds: players.slice(0, 4),
        matches: 2,
        model: "legacy",
        actorId: players[0],
        requestId: randomUUID(),
        ...overrides,
      })
    );
    expect(response.status).toBe(status);
  });

  it("POST glicko2 无配置 → 409 rating_unavailable", async () => {
    const response = await POST(
      postRequest({
        playerIds: players.slice(0, 4),
        matches: 1,
        model: "glicko2",
        actorId: players[0],
        requestId: randomUUID(),
      })
    );
    expect(response.status).toBe(409);
    const json = await response.json();
    expect(json.code).toBe("rating_unavailable");
    expect(json.state).toBe("unavailable");
  });

  it("GET 列表：分页游标与计数", async () => {
    for (let i = 0; i < 3; i++) {
      const res = await POST(
        postRequest({
          name: `项目${i}`,
          playerIds: players.slice(0, 4),
          matches: 1,
          model: "legacy",
          actorId: players[0],
          requestId: randomUUID(),
        })
      );
      expect(res.status).toBe(201);
    }

    const page1 = await GET(getRequest("?limit=2"));
    expect(page1.status).toBe(200);
    const json1 = await page1.json();
    expect(json1.projects).toHaveLength(2);
    expect(json1.nextCursor).not.toBeNull();
    // 新→旧排序。
    expect(json1.projects[0].name).toBe("项目2");
    expect(json1.projects[0].activePlayers).toBe(4);
    expect(json1.projects[0].pendingCount).toBe(1);

    const page2 = await GET(getRequest(`?limit=2&cursor=${json1.nextCursor}`));
    const json2 = await page2.json();
    expect(json2.projects).toHaveLength(1);
    expect(json2.projects[0].name).toBe("项目0");
    expect(json2.nextCursor).toBeNull();
  });

  it("GET 非法 cursor/limit → 400", async () => {
    const badCursor = await GET(getRequest("?cursor=abc"));
    expect(badCursor.status).toBe(400);
    const badLimit = await GET(getRequest("?limit=0"));
    expect(badLimit.status).toBe(400);
  });
});
