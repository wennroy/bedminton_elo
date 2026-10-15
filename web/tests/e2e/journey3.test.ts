import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { startServer, type RunningServer } from "./helpers/server";
import {
  ApiClient,
  type FixtureShape,
  type ProjectDetailShape,
} from "./helpers/http";
import { launchBrowser, type BrowserHandle } from "./helpers/browser";
import { visible } from "./helpers/ui";

/**
 * 旅程 3：任意顺序完成 3 场（其中一场仅标记已打待录）；移除一名成员、
 * 加入一名新人后续排：已打 ID/阵容/日期/比分保留，离场者不进入新场次，
 * 新人无补场负担（真实已打=0，配额口径从加入起算）。
 */

const PORT = 8913;

let db: TempDb;
let server: RunningServer;
let browser: BrowserHandle;
let api: ApiClient;
let projectId: number;
let names: Record<string, string>;
let playedBefore: Array<Pick<FixtureShape, "id" | "pa1" | "pa2" | "pb1" | "pb2" | "status" | "playedAt" | "scoreA" | "scoreB">>;

async function loadProject(): Promise<ProjectDetailShape> {
  const res = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${projectId}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

beforeAll(async () => {
  db = makeTempDb("j3");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  browser = await launchBrowser();

  const created = await api.post<{ project: ProjectDetailShape; names: Record<string, string> }>(
    "/api/schedule/projects",
    {
      name: "E2E 旅程三项目",
      playerIds: PLAYER_IDS.slice(0, 10).map(String),
      matches: 8,
      actorId: "1",
      requestId: crypto.randomUUID(),
    }
  );
  expect(created.status).toBe(201);
  projectId = created.body.project.id;
  names = created.body.names;
});

afterAll(async () => {
  await browser.close();
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 3：已打保留 + 人员变动续排", () => {
  it("任意顺序完成 3 场：一场直接记分、一场仅标记已打、一场再记分", async () => {
    const project = await loadProject();
    const [f1, f2, f3] = project.fixtures;

    // 顺序 1：f2 仅标记已打（显式日期 2026-10-01）。
    const played = await api.post<{ fixture: FixtureShape }>(
      `/api/schedule/projects/${projectId}/matches/${f2.id}/played`,
      {
        expectedFixtureRevision: f2.revision,
        actorId: "1",
        playedAt: "2026-10-01",
      }
    );
    expect(played.status).toBe(200);
    expect(played.body.fixture.status).toBe("played_unscored");
    expect(played.body.fixture.playedAt).toBe("2026-10-01");

    // 顺序 2：f3 直接记分（显式日期 2026-09-30）。
    const s3 = await api.post(
      `/api/schedule/projects/${projectId}/matches/${f3.id}/score`,
      {
        scoreA: 21,
        scoreB: 12,
        expectedFixtureRevision: f3.revision,
        actorId: "2",
        playedAt: "2026-09-30",
      }
    );
    expect(s3.status).toBe(201);

    // 顺序 3：f1 记分（显式日期 2026-09-28）。
    const s1 = await api.post(
      `/api/schedule/projects/${projectId}/matches/${f1.id}/score`,
      {
        scoreA: 15,
        scoreB: 21,
        expectedFixtureRevision: f1.revision,
        actorId: "3",
        playedAt: "2026-09-28",
      }
    );
    expect(s1.status).toBe(201);

    const after = await loadProject();
    const snap = (id: number) => {
      const f = after.fixtures.find((x) => x.id === id)!;
      const { pa1, pa2, pb1, pb2, status, playedAt, scoreA, scoreB } = f;
      return { id, pa1, pa2, pb1, pb2, status, playedAt, scoreA, scoreB };
    };
    playedBefore = [snap(f1.id), snap(f2.id), snap(f3.id)];
    expect(playedBefore.map((f) => f.status)).toEqual([
      "recorded",
      "played_unscored",
      "recorded",
    ]);
    expect(playedBefore[0].scoreA).toBe(15);
    expect(playedBefore[0].scoreB).toBe(21);
    expect(playedBefore[1].playedAt).toBe("2026-10-01");
    expect(playedBefore[2].playedAt).toBe("2026-09-30");
  });

  it("浏览器移除一名成员、加入新人后续排：已打三场原样保留", async () => {
    const leaverId = playedBefore[0].pa1; // 已打场次的出场者
    const leaverName = names[leaverId];
    expect(leaverName).toBeTruthy();
    const newcomerName = "E2E-11";

    const context = await browser.newIdentityContext(PLAYER_IDS[0]);
    const page = await context.newPage();
    await page.goto(`${server.baseUrl}/schedule/${projectId}`);

    await page.getByRole("button", { name: "调整人员并续排" }).click();
    await page.getByRole("button", { name: `移除 ${leaverName}` }).click();
    await page.getByRole("button", { name: /添加球员/ }).click();
    await page.getByRole("button", { name: newcomerName }).click();
    await page.getByLabel("续排场次数", { exact: true }).fill("4");
    await visible(page.getByText("保留 3 场已打比赛，替换 5 场待打比赛"));
    await page.getByRole("button", { name: "重新生成待打比赛" }).click();
    await visible(page.getByText("已重新生成 4 场待打比赛"));
    await context.close();

    // API 断言：已打三场 ID/阵容/状态/日期/比分原样保留。
    // 注意用续排后服务端返回的最新 names（包含新人 E2E-11）。
    const detail = await api.get<{
      project: ProjectDetailShape;
      names: Record<string, string>;
    }>(`/api/schedule/projects/${projectId}`);
    const project = detail.body.project;
    const latestNames = detail.body.names;
    for (const before of playedBefore) {
      const now = project.fixtures.find((f) => f.id === before.id);
      expect(now, `fixture ${before.id} 必须仍在有效场次中`).toBeTruthy();
      expect({
        id: now!.id,
        pa1: now!.pa1,
        pa2: now!.pa2,
        pb1: now!.pb1,
        pb2: now!.pb2,
        status: now!.status,
        playedAt: now!.playedAt,
        scoreA: now!.scoreA,
        scoreB: now!.scoreB,
      }).toEqual(before);
    }

    // 旧待打 5 场全部 superseded；新出 4 场 pending。
    expect(project.superseded).toHaveLength(5);
    const pending = project.fixtures.filter((f) => f.status === "pending");
    expect(pending).toHaveLength(4);
    expect(project.fixtures.filter((f) => f.status === "recorded")).toHaveLength(2);
    expect(project.fixtures.filter((f) => f.status === "played_unscored")).toHaveLength(1);

    // 离场者不进入新场次，也不再是 active 成员。
    for (const f of pending) {
      expect([f.pa1, f.pa2, f.pb1, f.pb2]).not.toContain(leaverId);
    }
    expect(project.members.map((m) => m.playerId)).not.toContain(leaverId);

    // 新人无补场负担：真实已打 0、加入时点配额基数 0——不背负补场债务；
    // fairness_base 与「继续在场者最低 effective」对齐（effective =
    // fairness_base + actualPlayed - played_at_join，本项目原成员后两项起点为 0）。
    const newcomer = project.members.find((m) => latestNames[m.playerId] === newcomerName)!;
    expect(newcomer.actualPlayed).toBe(0);
    expect(newcomer.playedAtJoin).toBe(0);
    const continuingEffective = project.members
      .filter((m) => m.playerId !== newcomer.playerId)
      .map((m) => m.fairnessBase + m.actualPlayed - m.playedAtJoin);
    expect(newcomer.fairnessBase).toBe(Math.min(...continuingEffective));

    // 继续在场者保留其已打计数（不重复计算）。
    for (const m of project.members) {
      if (m.playerId === newcomer.playerId) continue;
      const before = playedBefore;
      const expectedPlayed = before.filter(
        (f) => [f.pa1, f.pa2, f.pb1, f.pb2].includes(m.playerId)
      ).length;
      expect(m.actualPlayed).toBe(expectedPlayed);
    }
  });
});
