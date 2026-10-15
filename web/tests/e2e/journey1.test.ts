import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { makeTempDb, PLAYER_IDS, type TempDb } from "./helpers/db";
import { startServer, type RunningServer } from "./helpers/server";
import { ApiClient, type ProjectDetailShape } from "./helpers/http";
import { launchBrowser, type BrowserHandle } from "./helpers/browser";
import { textMatches, urlIs, visible } from "./helpers/ui";

/**
 * 旅程 1：A 创建 10 人 8 场项目并改名；B（另一浏览器上下文/另一身份）
 * 列表看到并打开；重启应用进程后项目与名称仍在。
 */

const PORT = 8911;

let db: TempDb;
let server: RunningServer;
let browser: BrowserHandle;
let api: ApiClient;

beforeAll(async () => {
  db = makeTempDb("j1");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  browser = await launchBrowser();
});

afterAll(async () => {
  await browser.close();
  await server.stop();
  db.close();
  db.cleanup();
});

describe("旅程 1：创建 / 共享可见 / 改名 / 重启持久", () => {
  it("A 通过浏览器创建 10 人 8 场项目并改名", async () => {
    const contextA = await browser.newIdentityContext(PLAYER_IDS[0]);
    const pageA = await contextA.newPage();
    await pageA.goto(`${server.baseUrl}/schedule/new`);

    // 勾选 10 名球员（E2E-01 .. E2E-10），留下 11/12 不选。
    for (let i = 1; i <= 10; i += 1) {
      const name = `E2E-${String(i).padStart(2, "0")}`;
      await pageA.getByRole("button", { name, exact: true }).click();
    }
    await visible(pageA.getByText("已选 10 人"));

    // 8 场 + 命名 + 默认勾选「减少已打对阵重复」。
    await pageA.getByLabel("场次数", { exact: true }).fill("8");
    await pageA.getByLabel("项目名称", { exact: true }).fill("E2E 旅程一项目");
    await pageA.getByRole("button", { name: "创建配对项目" }).click();

    // 落详情页：名单 10 人、待打 8 场。
    await urlIs(pageA, /\/schedule\/\d+$/);
    await visible(pageA.getByRole("heading", { name: "E2E 旅程一项目" }));
    await visible(pageA.getByText("当前名单 · 10 人"));
    await visible(pageA.getByText("待打 · 8"));

    // 改名。
    await pageA.getByRole("button", { name: "改名" }).click();
    await pageA.getByLabel("项目名称", { exact: true }).fill("E2E 旅程一改名后");
    await pageA.getByRole("button", { name: "保存", exact: true }).click();
    await visible(pageA.getByRole("heading", { name: "E2E 旅程一改名后" }));
    await contextA.close();

    // API 侧核对：8 个 pending 场次、revision 已随改名递增。
    const list = await api.get<{ projects: Array<{ id: number; name: string }> }>(
      "/api/schedule/projects"
    );
    expect(list.status).toBe(200);
    expect(list.body.projects).toHaveLength(1);
    const pid = list.body.projects[0].id;
    const detail = await api.get<{ project: ProjectDetailShape }>(`/api/schedule/projects/${pid}`);
    expect(detail.body.project.name).toBe("E2E 旅程一改名后");
    expect(detail.body.project.fixtures).toHaveLength(8);
    expect(detail.body.project.fixtures.every((f) => f.status === "pending")).toBe(true);
    expect(detail.body.project.members).toHaveLength(10);
  });

  it("B（另一上下文/身份）列表看到并打开详情", async () => {
    const contextB = await browser.newIdentityContext(PLAYER_IDS[1]);
    const pageB = await contextB.newPage();
    await pageB.goto(`${server.baseUrl}/schedule`);

    await visible(pageB.getByText("共 1 个项目"));
    const link = pageB.getByRole("link", { name: /E2E 旅程一改名后/ });
    await visible(link);
    await textMatches(link, "10 人在名单");
    await textMatches(link, "待打 8");

    await link.click();
    await urlIs(pageB, /\/schedule\/\d+$/);
    await visible(pageB.getByRole("heading", { name: "E2E 旅程一改名后" }));
    await visible(pageB.getByText("待打 · 8"));
    await contextB.close();
  });

  it("重启应用进程后项目与名称仍在", async () => {
    const before = await api.get<{ projects: Array<{ id: number; name: string }> }>(
      "/api/schedule/projects"
    );
    expect(before.body.projects).toHaveLength(1);

    await server.stop();
    server = await startServer(db.dbPath, PORT);
    api = new ApiClient(server.baseUrl);

    // API 直接核对持久化。
    const after = await api.get<{ projects: Array<{ id: number; name: string }> }>(
      "/api/schedule/projects"
    );
    expect(after.status).toBe(200);
    expect(after.body.projects).toHaveLength(1);
    expect(after.body.projects[0].name).toBe("E2E 旅程一改名后");

    // 真实浏览器复核（B 身份重开）。
    const contextB = await browser.newIdentityContext(PLAYER_IDS[1]);
    const pageB = await contextB.newPage();
    await pageB.goto(`${server.baseUrl}/schedule`);
    const link = pageB.getByRole("link", { name: /E2E 旅程一改名后/ });
    await visible(link);
    await link.click();
    await visible(pageB.getByRole("heading", { name: "E2E 旅程一改名后" }));
    await visible(pageB.getByText("待打 · 8"));
    await contextB.close();
  });
});
