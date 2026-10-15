import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type Database from "better-sqlite3";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { startServer, type RunningServer } from "./helpers/server";
import { ApiClient, type FixtureShape, type ProjectDetailShape } from "./helpers/http";
import { launchBrowser, type BrowserHandle } from "./helpers/browser";
import { runConcurrentPosts } from "./helpers/workers";
import { countIsZero, formAlert, textMatches, visible } from "./helpers/ui";

/**
 * 旅程 2：A/B 同时打开同一 fixture 记分页，A 先提交；B 收到 already_recorded
 * 并看到同一比分；DB 仅增一条 matches、关联唯一。真实争写用两个独立 OS
 * 进程并发打 HTTP；不同 fixture 并发登记均成功。
 */

const PORT = 8912;

let db: TempDb;
let server: RunningServer;
let browser: BrowserHandle;
let api: ApiClient;
let sqlite: Database.Database;
let projectId: number;
let fixtures: FixtureShape[];

async function loadProject(): Promise<ProjectDetailShape> {
  const res = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${projectId}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

function countMatches(): number {
  const row = sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number };
  return row.c;
}

beforeAll(async () => {
  db = makeTempDb("j2");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  browser = await launchBrowser();
  sqlite = db.open();

  const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
    name: "E2E 旅程二项目",
    playerIds: PLAYER_IDS.slice(0, 10).map(String),
    matches: 8,
    actorId: String(PLAYER_IDS[0]),
    requestId: crypto.randomUUID(),
  });
  expect(created.status).toBe(201);
  projectId = created.body.project.id;
  fixtures = created.body.project.fixtures;
  expect(fixtures).toHaveLength(8);
});

afterAll(async () => {
  await browser.close();
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 2：并发登记防重", () => {
  it("两个独立 OS 进程对同一场次并发记分：一个 201、一个 409 already_recorded", async () => {
    const fixture = fixtures[0];
    const revision = fixture.revision;
    const outcomes = await runConcurrentPosts(server.baseUrl, [
      {
        path: `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
        body: { scoreA: 21, scoreB: 19, expectedFixtureRevision: revision, actorId: "1" },
      },
      {
        path: `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
        body: { scoreA: 21, scoreB: 20, expectedFixtureRevision: revision, actorId: "2" },
      },
    ]);

    const statuses = outcomes.map((o) => o.status).sort();
    expect(statuses).toEqual([201, 409]);
    const winner = outcomes.find((o) => o.status === 201)!;
    const loser = outcomes.find((o) => o.status === 409)!;

    // 409 语义精确：already_recorded，且携带胜者已写入的比分。
    const loserBody = loser.body as { code: string; fixture: FixtureShape };
    expect(loserBody.code).toBe("already_recorded");
    expect(loserBody.fixture.status).toBe("recorded");
    const winBody = winner.body as { fixture: FixtureShape };
    expect(loserBody.fixture.scoreA).toBe(winBody.fixture.scoreA);
    expect(loserBody.fixture.scoreB).toBe(winBody.fixture.scoreB);
    expect(loserBody.fixture.matchId).toBe(winBody.fixture.matchId);

    // DB 事实：恰好一条 matches、关联唯一、比分是胜者的。
    expect(countMatches()).toBe(1);
    const row = sqlite
      .prepare(
        `SELECT id, score_a AS scoreA, score_b AS scoreB FROM matches`
      )
      .get() as { id: number; scoreA: number; scoreB: number };
    const fx = sqlite
      .prepare(`SELECT match_id AS matchId FROM schedule_fixtures WHERE id = ?`)
      .get(fixture.id) as { matchId: number };
    expect(fx.matchId).toBe(row.id);
    expect(row.scoreA).toBe(winBody.fixture.scoreA);
    expect(row.scoreB).toBe(winBody.fixture.scoreB);

    const project = await loadProject();
    const recorded = project.fixtures.find((f) => f.id === fixture.id)!;
    expect(recorded.status).toBe("recorded");
    expect(recorded.matchId).toBe(row.id);
  });

  it("浏览器双上下文：A 先提交成功，B 同页提交收到 already_recorded 并看到同一比分", async () => {
    const fixture = fixtures[1];
    const recordUrl = `${server.baseUrl}/schedule/${projectId}/matches/${fixture.id}/record`;

    const ctxB = await browser.newIdentityContext(PLAYER_IDS[1]);
    const pageB = await ctxB.newPage();
    await pageB.goto(recordUrl);
    await visible(pageB.getByRole("button", { name: /确认登记比分/ }));

    const ctxA = await browser.newIdentityContext(PLAYER_IDS[0]);
    const pageA = await ctxA.newPage();
    await pageA.goto(recordUrl);
    await pageA.getByLabel("A队比分").fill("21");
    await pageA.getByLabel("B队比分").fill("15");
    await pageA.getByRole("button", { name: /确认登记比分/ }).click();
    await visible(pageA.getByText("比赛已登记"));
    await visible(pageA.getByText("21 : 15"));
    await ctxA.close();

    // B 的页面停留在旧状态（pending），现在提交：409 already_recorded 并刷新为只读。
    await pageB.getByLabel("B队比分").fill("18");
    await pageB.getByRole("button", { name: /确认登记比分/ }).click();
    await textMatches(formAlert(pageB), "已由其他人登记比分");
    // 只读视图展示 A 的同一份比分（不是 B 的 21:18）。
    await visible(pageB.getByText("21 : 15"));
    await countIsZero(pageB.getByText("21 : 18"));
    await ctxB.close();

    expect(countMatches()).toBe(2);
  });

  it("不同 fixture 并发登记：两个独立进程均 201", async () => {
    const f2 = fixtures[2];
    const f3 = fixtures[3];
    const project = await loadProject();
    const r2 = project.fixtures.find((f) => f.id === f2.id)!.revision;
    const r3 = project.fixtures.find((f) => f.id === f3.id)!.revision;

    const outcomes = await runConcurrentPosts(server.baseUrl, [
      {
        path: `/api/schedule/projects/${projectId}/matches/${f2.id}/score`,
        body: { scoreA: 22, scoreB: 20, expectedFixtureRevision: r2, actorId: "3" },
      },
      {
        path: `/api/schedule/projects/${projectId}/matches/${f3.id}/score`,
        body: { scoreA: 21, scoreB: 17, expectedFixtureRevision: r3, actorId: "4" },
      },
    ]);

    expect(outcomes.map((o) => o.status).sort()).toEqual([201, 201]);
    expect(countMatches()).toBe(4);

    const after = await loadProject();
    const rec2 = after.fixtures.find((f) => f.id === f2.id)!;
    const rec3 = after.fixtures.find((f) => f.id === f3.id)!;
    expect(rec2.status).toBe("recorded");
    expect(rec2.scoreA).toBe(22);
    expect(rec2.scoreB).toBe(20);
    expect(rec3.status).toBe("recorded");
    expect(rec3.scoreA).toBe(21);
    expect(rec3.scoreB).toBe(17);
    // 两场各关联不同比赛。
    expect(rec2.matchId).not.toBe(rec3.matchId);
  });
});
