import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type Database from "better-sqlite3";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { startServer, type RunningServer } from "./helpers/server";
import {
  ApiClient,
  type FixtureShape,
  type ProjectDetailShape,
} from "./helpers/http";
import { launchBrowser, type BrowserHandle } from "./helpers/browser";
import {
  attrIs,
  disabled,
  inputValueIs,
  urlIs,
  visible,
} from "./helpers/ui";

/**
 * 旅程 8：表单与异常。
 * 无效比分、未选身份操作被拒、未知/重复 ID、<4 人不能生成但 matches=0
 * 可保存名单、旧 localStorage 缓存不自动上传、旧/陌生项目 URL 可读或
 * 404 语义正确；窄屏（390px）快速记分流程 + 原手动 /record 回归。
 */

const PORT = 8918;

let db: TempDb;
let server: RunningServer;
let browser: BrowserHandle;
let api: ApiClient;
let sqlite: Database.Database;
let projectId: number;
let fixture: FixtureShape;

async function loadProject(): Promise<ProjectDetailShape> {
  const res = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${projectId}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

beforeAll(async () => {
  db = makeTempDb("j8");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  browser = await launchBrowser();
  sqlite = db.open();

  const created = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
    name: "E2E 旅程八项目",
    playerIds: PLAYER_IDS.slice(0, 6).map(String),
    matches: 4,
    actorId: "1",
    requestId: crypto.randomUUID(),
  });
  expect(created.status).toBe(201);
  projectId = created.body.project.id;
  fixture = created.body.project.fixtures[0];
});

afterAll(async () => {
  await browser.close();
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 8：表单与异常", () => {
  it("无效比分与非整数入参一律 400，不落库", async () => {
    const before = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }
    ).c;
    const base = {
      expectedFixtureRevision: fixture.revision,
      actorId: "1",
    };
    // 平局。
    const tie = await api.post(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      { ...base, scoreA: 21, scoreB: 21 }
    );
    expect(tie.status).toBe(400);
    expect((tie.body as { code: string }).code).toBe("validation_error");
    // 负数。
    const negative = await api.post(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      { ...base, scoreA: -1, scoreB: 21 }
    );
    expect(negative.status).toBe(400);
    // 非整数。
    const nonInteger = await api.post(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      { ...base, scoreA: 21.5, scoreB: 10 }
    );
    expect(nonInteger.status).toBe(400);
    // 字符串比分。
    const stringScore = await api.post(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      { ...base, scoreA: "21", scoreB: 10 }
    );
    expect(stringScore.status).toBe(400);

    const after = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }
    ).c;
    expect(after).toBe(before);
    const project = await loadProject();
    expect(project.fixtures.find((f) => f.id === fixture.id)!.status).toBe("pending");
  });

  it("未选身份：API 缺 actorId 400；浏览器按钮禁用", async () => {
    const noActor = await api.post(
      `/api/schedule/projects/${projectId}/matches/${fixture.id}/score`,
      { scoreA: 21, scoreB: 8, expectedFixtureRevision: fixture.revision }
    );
    expect(noActor.status).toBe(400);
    expect((noActor.body as { code: string }).code).toBe("validation_error");

    const ctx = await browser.newIdentityContext(null);
    const page = await ctx.newPage();

    await page.goto(`${server.baseUrl}/schedule`);
    await disabled(page.getByRole("button", { name: "新建配对项目" }));

    await page.goto(`${server.baseUrl}/schedule/${projectId}`);
    await disabled(page.getByRole("button", { name: "改名" }));
    await disabled(page.getByRole("button", { name: "标记已打" }).first());
    await ctx.close();
  });

  it("未知/重复/畸形 ID：404 或 400，不泄露存在性", async () => {
    // 未知项目与未知场次：统一 404 not_found。
    const unknownProject = await api.get("/api/schedule/projects/999999");
    expect(unknownProject.status).toBe(404);
    expect((unknownProject.body as { code: string }).code).toBe("not_found");
    const unknownFixture = await api.post(
      `/api/schedule/projects/${projectId}/matches/999999/score`,
      { scoreA: 21, scoreB: 8, expectedFixtureRevision: 0, actorId: "1" }
    );
    expect(unknownFixture.status).toBe(404);
    expect((unknownFixture.body as { code: string }).code).toBe("not_found");

    // 跨项目场次访问：场次属于另一项目 → 404（不泄露存在性）。
    const other = await api.post<{ project: ProjectDetailShape }>("/api/schedule/projects", {
      name: "E2E 旅程八第二项目",
      playerIds: PLAYER_IDS.slice(6, 10).map(String),
      matches: 2,
      actorId: "1",
      requestId: crypto.randomUUID(),
    });
    expect(other.status).toBe(201);
    const cross = await api.post(
      `/api/schedule/projects/${other.body.project.id}/matches/${fixture.id}/score`,
      { scoreA: 21, scoreB: 8, expectedFixtureRevision: fixture.revision, actorId: "1" }
    );
    expect(cross.status).toBe(404);

    // 重复球员 ID。
    const dup = await api.post("/api/schedule/projects", {
      name: "E2E 重复",
      playerIds: ["1", "2", "3", "3"],
      matches: 1,
      actorId: "1",
      requestId: crypto.randomUUID(),
    });
    expect(dup.status).toBe(400);
    // 不存在的球员。
    const unknownPlayer = await api.post("/api/schedule/projects", {
      name: "E2E 幽灵",
      playerIds: ["1", "2", "3", "999999"],
      matches: 1,
      actorId: "1",
      requestId: crypto.randomUUID(),
    });
    expect(unknownPlayer.status).toBe(400);
    expect((unknownPlayer.body as { code: string }).code).toBe("player_not_found");
    // 畸形 ID。
    const badId = await api.post("/api/schedule/projects", {
      name: "E2E 畸形",
      playerIds: ["1", "2", "3", "abc"],
      matches: 1,
      actorId: "1",
      requestId: crypto.randomUUID(),
    });
    expect(badId.status).toBe(400);
  });

  it("<4 人不能生成，但 matches=0 可保存名单并清空未打", async () => {
    const project = await loadProject();

    // 3 人 + matches=1 → 400。
    const tooFew = await api.post(`/api/schedule/projects/${projectId}/generate`, {
      playerIds: ["1", "2", "3"],
      matches: 1,
      expectedRevision: project.revision,
      actorId: "1",
      requestId: crypto.randomUUID(),
    });
    expect(tooFew.status).toBe(400);

    // 3 人 + matches=0 → 201：名单保存、未打清空。
    const saved = await api.post<{ project: ProjectDetailShape }>(
      `/api/schedule/projects/${projectId}/generate`,
      {
        playerIds: ["1", "2", "3"],
        matches: 0,
        expectedRevision: project.revision,
        actorId: "1",
        requestId: crypto.randomUUID(),
      }
    );
    expect(saved.status).toBe(201);
    expect(saved.body.project.members.map((m) => m.playerId).sort()).toEqual([
      "1",
      "2",
      "3",
    ]);
    expect(saved.body.project.fixtures.filter((f) => f.status === "pending")).toHaveLength(0);
    expect(saved.body.project.superseded).toHaveLength(4);

    // 回到 4 人即可再生成。
    const regen = await api.post<{ fixtures: unknown[] }>(
      `/api/schedule/projects/${projectId}/generate`,
      {
        playerIds: ["1", "2", "3", "4"],
        matches: 2,
        expectedRevision: saved.body.project.revision,
        actorId: "1",
        requestId: crypto.randomUUID(),
      }
    );
    expect(regen.status).toBe(201);
    expect(regen.body.fixtures).toHaveLength(2);
  });

  it("旧 localStorage 配对草稿只作提示与勾选参考，不上传", async () => {
    const legacyDraft = {
      playerIds: [1, 2, 3, 4],
      matches: 5,
      seed: 1234,
      lambda: 0.5,
      savedAt: "2026-09-01T12:00:00.000Z",
      result: {
        schedule: [{ a1: "1", a2: "2", b1: "3", b2: "4", winRate: 0.5 }],
        metrics: {
          alphaVar: 0,
          bestLoss: 0,
          meanCloseness: 0,
          maxCloseness: 0,
          entropy: 0,
        },
        names: { "1": "E2E-01", "2": "E2E-02", "3": "E2E-03", "4": "E2E-04" },
      },
    };
    const context = await browser.newIdentityContext(PLAYER_IDS[0]);
    await context.addInitScript((draft: object) => {
      window.localStorage.setItem("badminton:schedule:1", JSON.stringify(draft));
    }, legacyDraft);
    const page = await context.newPage();
    await page.goto(`${server.baseUrl}/schedule/new`);

    // 提示出现 + 名单按草稿勾选 + 场次数按草稿带过来。
    await visible(page.getByText(/检测到本机旧版配对草稿/));
    await visible(page.getByText(/已按当时的名单勾选/));
    await inputValueIs(page.getByLabel("场次数", { exact: true }), "5");
    for (const name of ["E2E-01", "E2E-02", "E2E-03", "E2E-04"]) {
      await attrIs(page.getByRole("button", { name, exact: true }), "aria-pressed", "true");
    }

    // 服务器侧没有被自动创建任何项目（本文件此前用例已建 2 个，数量不变）。
    const list = await api.get<{ projects: Array<{ name: string }> }>(
      "/api/schedule/projects"
    );
    expect(list.body.projects).toHaveLength(2);
    expect(
      list.body.projects.every((p) => p.name.startsWith("E2E 旅程八"))
    ).toBe(true);
    await context.close();
  });

  it("旧/陌生项目 URL：可读的项目 200，未知/畸形 404", async () => {
    const ctx = await browser.newIdentityContext(PLAYER_IDS[0]);
    const page = await ctx.newPage();
    const ok = await page.goto(`${server.baseUrl}/schedule/${projectId}`);
    expect(ok?.status()).toBe(200);
    await visible(page.getByRole("heading", { name: "E2E 旅程八项目" }));

    const unknown = await page.goto(`${server.baseUrl}/schedule/999999`);
    expect(unknown?.status()).toBe(404);
    const malformed = await page.goto(`${server.baseUrl}/schedule/abc`);
    expect(malformed?.status()).toBe(404);
    const unknownFixture = await page.goto(
      `${server.baseUrl}/schedule/${projectId}/matches/999999/record`
    );
    expect(unknownFixture?.status()).toBe(404);
    await ctx.close();
  });

  it("窄屏 390px 快速记分全流程，无横向滚动", async () => {
    const context = await browser.newIdentityContext(PLAYER_IDS[0]);
    const page = await context.newPage();
    await page.setViewportSize({ width: 390, height: 844 });

    await page.goto(`${server.baseUrl}/schedule/${projectId}`);
    // 手机宽度无横向滚动。
    const detailScroll = await page.evaluate(
      () => document.scrollingElement!.scrollWidth
    );
    expect(detailScroll).toBeLessThanOrEqual(391);

    await page
      .getByRole("link", { name: "记分", exact: true })
      .first()
      .click();
    await urlIs(page, /\/record$/);
    const recordScroll = await page.evaluate(
      () => document.scrollingElement!.scrollWidth
    );
    expect(recordScroll).toBeLessThanOrEqual(391);

    await page.getByLabel("A队比分").fill("21");
    await page.getByLabel("B队比分").fill("12");
    await page.getByRole("button", { name: /确认登记比分/ }).click();
    await visible(page.getByText("比赛已登记"));
    await visible(page.getByText("21 : 12"));
    await context.close();
  });

  it("原手动 /record 回归：选人 → 比分 → 确认 → 成功，且写入 matches", async () => {
    const before = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }
    ).c;

    const context = await browser.newIdentityContext(PLAYER_IDS[1]);
    const page = await context.newPage();
    await page.goto(`${server.baseUrl}/record`);

    const picks: Array<[string, string]> = [
      ["选择A队1号球员", "E2E-05"],
      ["选择A队2号球员", "E2E-06"],
      ["选择B队1号球员", "E2E-07"],
      ["选择B队2号球员", "E2E-08"],
    ];
    for (const [slotLabel, playerName] of picks) {
      await page.getByRole("button", { name: slotLabel }).click();
      await page.getByRole("button", { name: playerName, exact: true }).click();
    }
    await page.getByLabel("A队比分").fill("21");
    await page.getByLabel("B队比分").fill("18");
    await page.getByRole("button", { name: "确认比分" }).click();
    await visible(page.getByText("确认这场比赛"));
    await page.getByRole("button", { name: "确认记分" }).click();
    await visible(page.getByText("比赛已记录"));
    await context.close();

    // API 侧确认这场比赛已写入。
    const after = (
      sqlite.prepare(`SELECT COUNT(*) AS c FROM matches`).get() as { c: number }
    ).c;
    expect(after).toBe(before + 1);
    const matches = await api.get<
      Array<{
        pa1: number;
        pa2: number;
        pb1: number;
        pb2: number;
        scoreA: number;
        scoreB: number;
      }>
    >("/api/matches");
    const found = matches.body.find(
      (m) =>
        [m.pa1, m.pa2].sort().join(",") === "5,6" &&
        [m.pb1, m.pb2].sort().join(",") === "7,8"
    );
    expect(found).toBeTruthy();
    expect(found!.scoreA).toBe(21);
    expect(found!.scoreB).toBe(18);
  });
});
