import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type Database from "better-sqlite3";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { startServer, type RunningServer } from "./helpers/server";
import {
  ApiClient,
  type ProjectDetailShape,
} from "./helpers/http";
import { launchBrowser, type BrowserHandle } from "./helpers/browser";
import { textMatches, formAlert, visible } from "./helpers/ui";

/**
 * 旅程 5：B 持旧 project revision 准备续排，A 先记分/标已打或改名；
 * B 提交 409 且不覆盖新事实。反向：旧记分页对 superseded 场次记分被拒
 * （fixture_superseded）。
 */

const PORT = 8915;

let db: TempDb;
let server: RunningServer;
let browser: BrowserHandle;
let api: ApiClient;
let sqlite: Database.Database;
let projectId: number;

async function loadProject(): Promise<ProjectDetailShape> {
  const res = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${projectId}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

beforeAll(async () => {
  db = makeTempDb("j5");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  browser = await launchBrowser();
  sqlite = db.open();

  const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
    name: "E2E 旅程五项目",
    playerIds: PLAYER_IDS.slice(0, 10).map(String),
    matches: 8,
    actorId: "1",
    requestId: crypto.randomUUID(),
  });
  expect(created.status).toBe(201);
  projectId = created.body.project.id;
});

afterAll(async () => {
  await browser.close();
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 5：revision 冲突与 superseded 拒绝", () => {
  it("B 持旧 revision 续排，A 先标已打+改名：B 收 409 revision_conflict 且不覆盖", async () => {
    // B 视角的旧快照。
    const stale = await loadProject();
    const staleRevision = stale.revision;
    const pendingIds = stale.fixtures.filter((f) => f.status === "pending").map((f) => f.id);

    // A 先标记一场已打（revision++），再改名（revision++）。
    const target = stale.fixtures[0];
    const played = await api.post(
      `/api/schedule/projects/${projectId}/matches/${target.id}/played`,
      { expectedFixtureRevision: target.revision, actorId: "1" }
    );
    expect(played.status).toBe(200);
    const renamed = await api.patch(`/api/schedule/projects/${projectId}`, {
      name: "E2E 旅程五改名后",
      expectedRevision: staleRevision + 1,
      actorId: "1",
    });
    expect(renamed.status).toBe(200);

    // B 用旧 revision 提交续排：409 revision_conflict，携带当前 revision。
    const res = await api.post(`/api/schedule/projects/${projectId}/generate`, {
      playerIds: PLAYER_IDS.slice(0, 10).map(String),
      matches: 4,
      expectedRevision: staleRevision,
      actorId: "2",
      requestId: crypto.randomUUID(),
    });
    expect(res.status).toBe(409);
    const body = res.body as { code: string; projectRevision: number };
    expect(body.code).toBe("revision_conflict");
    expect(body.projectRevision).toBe(staleRevision + 2);

    // 新事实未被覆盖：改名保留、目标场次已打、旧待打未被替换。
    const after = await loadProject();
    expect(after.revision).toBe(staleRevision + 2);
    expect(after.name).toBe("E2E 旅程五改名后");
    expect(after.fixtures.find((f) => f.id === target.id)!.status).toBe("played_unscored");
    const stillPending = after.fixtures.filter((f) => f.status === "pending").map((f) => f.id);
    expect(stillPending.sort()).toEqual(pendingIds.filter((id) => id !== target.id).sort());
    expect(after.superseded).toHaveLength(0);
  });

  it("反向：旧记分页对 superseded 场次提交记分 → 409 fixture_superseded，浏览器转只读", async () => {
    // 选一个 pending 场次，B（浏览器）先打开其记分页。
    const project = await loadProject();
    const fixture = project.fixtures.find((f) => f.status === "pending")!;
    const recordUrl = `${server.baseUrl}/schedule/${projectId}/matches/${fixture.id}/record`;

    const ctxB = await browser.newIdentityContext(PLAYER_IDS[1]);
    const pageB = await ctxB.newPage();
    await pageB.goto(recordUrl);
    await visible(pageB.getByRole("button", { name: /确认登记比分/ }));

    // A 触发续排：该场次被 superseded。
    const res = await api.post<{ supersededCount: number }>(
      `/api/schedule/projects/${projectId}/generate`,
      {
        playerIds: PLAYER_IDS.slice(0, 10).map(String),
        matches: 3,
        expectedRevision: project.revision,
        actorId: "1",
        requestId: crypto.randomUUID(),
      }
    );
    expect(res.status).toBe(201);
    expect(res.body.supersededCount).toBeGreaterThanOrEqual(1);

    // B 在旧页面提交：409 fixture_superseded → 页面刷新为只读并提示。
    const matchesBefore = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }
    ).c;
    await pageB.getByLabel("A队比分").fill("21");
    await pageB.getByLabel("B队比分").fill("11");
    await pageB.getByRole("button", { name: /确认登记比分/ }).click();
    await textMatches(formAlert(pageB), "已被新一批对阵替换");
    await visible(pageB.getByText("已被替换"));
    await ctxB.close();

    // 无新比赛落库；场次确为 superseded。
    const matchesAfter = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }
    ).c;
    expect(matchesAfter).toBe(matchesBefore);
    const after = await loadProject();
    const fx = [...after.fixtures, ...after.superseded].find((f) => f.id === fixture.id)!;
    expect(fx.status).toBe("superseded");
  });
});
