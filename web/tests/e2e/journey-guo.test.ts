import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { BrowserContext, Page } from "playwright";
import type { Fixture, ProjectDetail } from "@/app/schedule/types";
import { makeTempDb, type TempDb } from "./helpers/db";
import { ADMIN_PASSWORD, startServer, type RunningServer } from "./helpers/server";
import { ApiClient } from "./helpers/http";
import { launchBrowser, type BrowserHandle } from "./helpers/browser";
import { disabled, urlIs, visible } from "./helpers/ui";

/**
 * Independent acceptance journey for the eight-player Guo format. Expectations
 * come from raw fixture results and tournament rules, never Guo engine helpers.
 * Run after next build: pnpm test:e2e -- tests/e2e/journey-guo.test.ts
 */
const PORT = 8920;
const NAME = "E2E 郭老师四轮实战";
const ids = Array.from({ length: 8 }, (_, i) => String(i + 1));
const pairKey = (a: string, b: string) => [a, b].sort().join(":");
const playerName = (id: string) => `E2E-${id.padStart(2, "0")}`;

let db: TempDb;
let server: RunningServer;
let browser: BrowserHandle;
let api: ApiClient;
let context: BrowserContext;
let page: Page;
let projectId: number;
let drawOrder: Record<string, number>;
const browserErrors: string[] = [];
const artifacts = mkdtempSync(join(tmpdir(), "bedminton-guo-browser-"));

type Round = NonNullable<ProjectDetail["guo"]>["rounds"][number];

async function loadProject(): Promise<ProjectDetail> {
  const res = await api.get<{ project: ProjectDetail }>(`/api/schedule/projects/${projectId}`);
  expect(res.status).toBe(200);
  return res.body.project;
}

function watchErrors(target: Page): void {
  target.on("pageerror", (error) => browserErrors.push(error.message));
  target.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
}

function fixturesFor(project: ProjectDetail, round: Round): Fixture[] {
  return project.fixtures.filter((f) => f.generationId === round.generationId);
}

function assertFullRound(project: ProjectDetail, round: Round): void {
  const fixtures = fixturesFor(project, round);
  expect(fixtures).toHaveLength(6);
  expect([...round.groupA, ...round.groupB].sort()).toEqual(ids);
  for (const group of [round.groupA, round.groupB]) {
    expect(group).toHaveLength(4);
    const matches = fixtures.filter((f) => group.includes(f.pa1));
    expect(matches).toHaveLength(3);
    const partners: string[] = [];
    const opponents: Record<string, number> = {};
    for (const f of matches) {
      expect([f.pa1, f.pa2, f.pb1, f.pb2].sort()).toEqual([...group].sort());
      partners.push(pairKey(f.pa1, f.pa2), pairKey(f.pb1, f.pb2));
      for (const a of [f.pa1, f.pa2]) for (const b of [f.pb1, f.pb2]) {
        const key = pairKey(a, b);
        opponents[key] = (opponents[key] ?? 0) + 1;
      }
    }
    const allPairs = group.flatMap((a, i) => group.slice(i + 1).map((b) => pairKey(a, b)));
    expect(partners.sort()).toEqual(allPairs.sort());
    expect(Object.keys(opponents).sort()).toEqual(allPairs.sort());
    expect(Object.values(opponents)).toEqual(Array(6).fill(2));
  }
}

function independentStandings(project: ProjectDetail, round: Round, group: string[]) {
  const matches = fixturesFor(project, round).filter((f) => f.status === "recorded");
  return group.map((playerId) => {
    let played = 0, wins = 0, pointsFor = 0, pointsAgainst = 0, expectedWins = 0;
    for (const f of matches) {
      const left = f.pa1 === playerId || f.pa2 === playerId;
      const right = f.pb1 === playerId || f.pb2 === playerId;
      if (!left && !right) continue;
      expect(f.scoreA).not.toBeNull();
      expect(f.scoreB).not.toBeNull();
      expect(f.generatedWinRate).toBeGreaterThan(0);
      expect(f.generatedWinRate).toBeLessThan(1);
      const own = left ? f.scoreA! : f.scoreB!;
      const other = left ? f.scoreB! : f.scoreA!;
      played += 1;
      if (own > other) wins += 1;
      pointsFor += own;
      pointsAgainst += other;
      expectedWins += left ? f.generatedWinRate! : 1 - f.generatedWinRate!;
    }
    return {
      playerId, played, wins, pointsFor, pointsAgainst,
      netPoints: pointsFor - pointsAgainst,
      expectedWins, overperformance: wins - expectedWins, drawOrder: drawOrder[playerId],
    };
  }).sort((a, b) => {
    if (a.wins !== b.wins) return b.wins - a.wins;
    if (a.netPoints !== b.netPoints) return b.netPoints - a.netPoints;
    if (Math.abs(a.overperformance - b.overperformance) > 1e-12) return b.overperformance - a.overperformance;
    return a.drawOrder - b.drawOrder;
  });
}

function assertStandings(project: ProjectDetail, round: Round): void {
  for (const [group, actual] of [[round.groupA, round.standingsA], [round.groupB, round.standingsB]] as const) {
    const expected = independentStandings(project, round, group);
    expect(actual.map((s) => s.playerId)).toEqual(expected.map((s) => s.playerId));
    for (let i = 0; i < expected.length; i += 1) {
      const { expectedWins, overperformance, ...exact } = expected[i];
      expect(actual[i]).toMatchObject(exact);
      expect(actual[i].expectedWins).toBeCloseTo(expectedWins, 10);
      expect(actual[i].overperformance).toBeCloseTo(overperformance, 10);
    }
  }
}

async function assertVisibleRound(project: ProjectDetail, round: Round): Promise<void> {
  await visible(page.getByRole("heading", { level: 2, name: new RegExp(`第 ${round.roundNo} 轮`) }));
  for (const [label, group] of [["A 组 · 场地 1", round.groupA], ["B 组 · 场地 2", round.groupB]] as const) {
    const region = page.getByRole("region", { name: label, exact: true });
    await visible(region);
    const rows = region.getByRole("table").locator("tbody tr");
    expect(await rows.count()).toBe(4);
    const expected = independentStandings(project, round, group);
    for (let i = 0; i < expected.length; i += 1) {
      const cells = await rows.nth(i).locator("th, td").allTextContents();
      expect(cells[0]).toContain(playerName(expected[i].playerId));
      expect(cells[1]).toMatch(new RegExp(`^${expected[i].wins}\\s*/\\s*${expected[i].played ? Math.round(expected[i].wins / expected[i].played * 100) + "%" : "—"}$`));
      expect(Number(cells[2])).toBe(expected[i].pointsFor);
      expect(Number(cells[3])).toBe(expected[i].netPoints);
      expect(Number(cells[4])).toBeCloseTo(expected[i].overperformance, 2);
    }
    expect(await region.locator("ol > li").count()).toBe(3);
  }
}

async function scoreFixture(fixture: Fixture, scoreA = 21, scoreB = 13): Promise<void> {
  const result = await api.post<{ fixture: Fixture }>(`/api/schedule/projects/${projectId}/matches/${fixture.id}/score`, {
    expectedFixtureRevision: fixture.revision, actorId: "1", scoreA, scoreB,
  });
  expect(result.status).toBe(201);
  expect(result.body.fixture.status).toBe("recorded");
}

async function completeRound(roundNo: number): Promise<ProjectDetail> {
  const before = await loadProject();
  const round = before.guo!.rounds.find((r) => r.roundNo === roundNo)!;
  for (const [index, fixture] of fixturesFor(before, round).entries()) {
    if (fixture.status === "recorded") continue;
    // R2 deliberately gives the three one-win players equal net points, so
    // frozen forecast and persistent draw order resolve real ranking ties.
    const scoreB = roundNo === 2 ? 13 : [13, 15, 18, 10, 11, 17][index];
    await scoreFixture(fixture, 21, scoreB);
  }
  const after = await loadProject();
  const finished = after.guo!.rounds.find((r) => r.roundNo === roundNo)!;
  expect(finished.complete).toBe(true);
  assertFullRound(after, finished);
  assertStandings(after, finished);
  await page.goto(`${server.baseUrl}/schedule/${projectId}`);
  await assertVisibleRound(after, finished);
  return after;
}

async function advanceViaBrowser(previous: ProjectDetail, roundNo: number): Promise<ProjectDetail> {
  const prior = previous.guo!.rounds.at(-1)!;
  const a = independentStandings(previous, prior, prior.groupA).map((s) => s.playerId);
  const b = independentStandings(previous, prior, prior.groupB).map((s) => s.playerId);
  const expectedA = [...a.slice(0, 2), ...(roundNo === 3 ? b.slice(2) : b.slice(0, 2))];
  const expectedB = [...a.slice(2), ...(roundNo === 3 ? b.slice(0, 2) : b.slice(2))];
  await page.getByRole("button", { name: roundNo === 4 ? "再打第 4 轮" : `开启第 ${roundNo} 轮`, exact: true }).click();
  await visible(page.getByRole("heading", { level: 2, name: new RegExp(`第 ${roundNo} 轮`) }));
  const next = await loadProject();
  expect(next.guo!.rounds).toHaveLength(roundNo);
  const round = next.guo!.rounds.at(-1)!;
  expect([...round.groupA].sort()).toEqual(expectedA.sort());
  expect([...round.groupB].sort()).toEqual(expectedB.sort());
  expect(round.kind).toBe(roundNo === 3 ? "cross" : "promotion");
  expect(next.fixtures).toHaveLength(roundNo * 6);
  expect(next.superseded).toHaveLength(0);
  const snapshot = next.latestGeneration!.input as { drawOrder: Record<string, number> };
  expect(snapshot.drawOrder).toEqual(drawOrder);
  assertFullRound(next, round);
  assertStandings(next, round);
  await assertVisibleRound(next, round);
  return next;
}

beforeAll(async () => {
  db = makeTempDb("guo");
  server = await startServer(db.dbPath, PORT);
  api = new ApiClient(server.baseUrl);
  browser = await launchBrowser();
  context = await browser.newIdentityContext(1);
  page = await context.newPage();
  await page.setViewportSize({ width: 390, height: 844 });
  watchErrors(page);
  // Known stronger four: seed actual wins before creating the project.
  for (const [pa1, pa2, pb1, pb2] of [[1, 2, 5, 6], [3, 4, 7, 8]]) {
    const seeded = await api.post("/api/matches", { pa1, pa2, pb1, pb2, scoreA: 21, scoreB: 10, playedAt: "2026-10-01", enteredBy: 1 });
    expect(seeded.status).toBe(201);
  }
});

afterAll(async () => {
  await context?.close();
  await browser?.close();
  await server?.stop();
  db?.close();
  db?.cleanup();
  console.info(`Guo browser evidence: ${artifacts}`);
});

describe("郭老师方法：真实浏览器从八人开局到四轮收官", () => {
  it("7/9 人禁止创建，8 人可创建并按实力分组；手机完成第一场记分", async () => {
    for (const count of [7, 9]) {
      const invalid = await api.post("/api/schedule/projects", { name: `E2E 无效 ${count}`, mode: "guo", playerIds: Array.from({ length: count }, (_, i) => String(i + 1)), actorId: "1", requestId: crypto.randomUUID() });
      expect(invalid.status).toBe(400);
    }
    await page.goto(`${server.baseUrl}/schedule/new`);
    await page.getByRole("radio", { name: /郭老师分配方法/ }).check();
    for (let i = 1; i <= 7; i += 1) await page.getByRole("button", { name: playerName(String(i)), exact: true }).click();
    const create = page.getByRole("button", { name: "创建配对项目", exact: true });
    await disabled(create);
    await page.getByRole("button", { name: "E2E-08", exact: true }).click();
    expect(await create.isEnabled()).toBe(true);
    await page.getByRole("button", { name: "E2E-09", exact: true }).click();
    await disabled(create);
    await page.getByRole("button", { name: "E2E-09", exact: true }).click();
    expect(await page.getByLabel("场次数", { exact: true }).count()).toBe(0);
    await page.getByLabel("项目名称", { exact: true }).fill(NAME);
    await page.getByRole("button", { name: "高级选项", exact: true }).click();
    await page.getByLabel("随机种子", { exact: true }).fill("20261019");
    await create.click();
    await urlIs(page, /\/schedule\/\d+$/);
    projectId = Number(page.url().split("/").at(-1));
    await visible(page.getByRole("heading", { name: NAME, exact: true }));
    const project = await loadProject();
    drawOrder = (project.latestGeneration!.input as { drawOrder: Record<string, number> }).drawOrder;
    expect(project.members.map((m) => m.playerId).sort()).toEqual(ids);
    expect(project.guo!.rounds).toHaveLength(1);
    expect([...project.guo!.rounds[0].groupA].sort()).toEqual(["1", "2", "3", "4"]);
    expect([...project.guo!.rounds[0].groupB].sort()).toEqual(["5", "6", "7", "8"]);
    assertFullRound(project, project.guo!.rounds[0]);
    await assertVisibleRound(project, project.guo!.rounds[0]);
    await visible(page.getByText(/还差 6 场比分/));
    await disabled(page.getByRole("button", { name: "开启第 2 轮", exact: true }));
    expect(await page.evaluate(() => document.scrollingElement!.scrollWidth)).toBeLessThanOrEqual(391);

    const first = project.fixtures[0];
    await page.getByRole("region", { name: "A 组 · 场地 1", exact: true }).getByRole("link", { name: "记分", exact: true }).first().click();
    await urlIs(page, new RegExp(`/matches/${first.id}/record$`));
    expect(await page.evaluate(() => document.scrollingElement!.scrollWidth)).toBeLessThanOrEqual(391);
    await page.getByLabel("A队比分").fill("21");
    await page.getByLabel("B队比分").fill("13");
    await page.getByRole("button", { name: /确认登记比分/ }).click();
    await visible(page.getByText("比赛已登记", { exact: true }));
    const scored = await loadProject();
    expect(scored.fixtures.find((f) => f.id === first.id)).toMatchObject({ status: "recorded", scoreA: 21, scoreB: 13 });
    assertStandings(scored, scored.guo!.rounds[0]);
  });

  it("已打但未记分仍不能进轮；补齐六场后按战绩升降到第二轮", async () => {
    let project = await loadProject();
    for (const [index, fixture] of project.fixtures.slice(1, 5).entries()) await scoreFixture(fixture, 21, [15, 18, 10, 11][index]);
    const last = project.fixtures[5];
    const marked = await api.post(`/api/schedule/projects/${projectId}/matches/${last.id}/played`, { expectedFixtureRevision: last.revision, actorId: "1", playedAt: "2026-10-02" });
    expect(marked.status).toBe(200);
    project = await loadProject();
    expect(project.guo!.canAdvance).toBe(false);
    const blocked = await api.post(`/api/schedule/projects/${projectId}/generate`, { actorId: "1", expectedRevision: project.revision, requestId: crypto.randomUUID() });
    expect(blocked.status).toBe(409);
    await page.goto(`${server.baseUrl}/schedule/${projectId}`);
    await visible(page.getByText(/还差 1 场比分/));
    await visible(page.getByRole("link", { name: "补记分", exact: true }));
    await disabled(page.getByRole("button", { name: "开启第 2 轮", exact: true }));
    await scoreFixture(project.fixtures.find((f) => f.id === last.id)!, 21, 17);
    project = await loadProject();
    expect(project.guo!.canAdvance).toBe(true);
    assertStandings(project, project.guo!.rounds[0]);
    await page.reload();
    await advanceViaBrowser(project, 2);
  });

  it("第二轮重新计分，第三轮按 A 前二与 B 后二交叉，手机与桌面无横向溢出", async () => {
    const completed = await completeRound(2);
    const next = await advanceViaBrowser(completed, 3);
    await visible(page.getByRole("heading", { level: 2, name: /第 3 轮.*交叉交流/ }));
    expect(await page.evaluate(() => document.scrollingElement!.scrollWidth)).toBeLessThanOrEqual(391);
    await page.screenshot({ path: join(artifacts, "guo-round3-mobile.png"), fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await assertVisibleRound(next, next.guo!.rounds.at(-1)!);
    expect(await page.evaluate(() => document.scrollingElement!.scrollWidth)).toBeLessThanOrEqual(1441);
    await page.screenshot({ path: join(artifacts, "guo-round3-desktop.png"), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
  });

  it("三轮完成后第四轮可选，恢复升降；四轮收官且不能开第五轮", async () => {
    const completed = await completeRound(3);
    await visible(page.getByText("三轮已打，可以收拍，也可以再来一轮", { exact: true }));
    await advanceViaBrowser(completed, 4);
    const finished = await completeRound(4);
    expect(finished.guo).toMatchObject({ maxRounds: 4, canAdvance: false, nextRoundKind: null });
    expect(finished.guo!.rounds.every((r) => r.complete)).toBe(true);
    expect(finished.fixtures).toHaveLength(24);
    await visible(page.getByText("四轮已完成，今天辛苦了！", { exact: true }));
    expect(await page.getByRole("button", { name: /开启第 5 轮|再打第 5 轮/ }).count()).toBe(0);
    const blocked = await api.post(`/api/schedule/projects/${projectId}/generate`, { actorId: "1", expectedRevision: finished.revision, requestId: crypto.randomUUID() });
    expect(blocked.status).toBe(409);
    expect((await loadProject()).fixtures).toHaveLength(24);
  });

  it("历史比分更正更新本轮战绩但不重排后轮；重启后另一身份读到相同四轮", async () => {
    const before = await loadProject();
    const teams = before.fixtures.map(({ id, generationId, pa1, pa2, pb1, pb2 }) => ({ id, generationId, pa1, pa2, pb1, pb2 }));
    const groups = before.guo!.rounds.map(({ groupA, groupB }) => ({ groupA, groupB }));
    const first = before.fixtures[0];
    const corrected = await api.patch(`/api/admin/matches/${first.matchId}`, { scoreA: 13, scoreB: 21 }, { "x-admin-key": ADMIN_PASSWORD });
    expect(corrected.status).toBe(200);
    const after = await loadProject();
    expect(after.guo!.rounds.map(({ groupA, groupB }) => ({ groupA, groupB }))).toEqual(groups);
    expect(after.fixtures.map(({ id, generationId, pa1, pa2, pb1, pb2 }) => ({ id, generationId, pa1, pa2, pb1, pb2 }))).toEqual(teams);
    expect(after.guo!.rounds[0].standingsA).not.toEqual(before.guo!.rounds[0].standingsA);
    for (const round of after.guo!.rounds) assertStandings(after, round);
    await server.stop();
    server = await startServer(db.dbPath, PORT);
    api = new ApiClient(server.baseUrl);
    const persisted = await loadProject();
    expect(persisted.guo).toEqual(after.guo);
    expect(persisted.fixtures).toEqual(after.fixtures);
    const otherContext = await browser.newIdentityContext(2);
    const otherPage = await otherContext.newPage();
    watchErrors(otherPage);
    await otherPage.goto(`${server.baseUrl}/schedule`);
    await otherPage.getByRole("link", { name: new RegExp(NAME) }).click();
    await visible(otherPage.getByRole("heading", { name: NAME, exact: true }));
    await visible(otherPage.getByRole("heading", { level: 2, name: /第 4 轮/ }));
    await otherPage.getByRole("button", { name: "第 1 轮", exact: true }).click();
    await visible(otherPage.getByRole("heading", { level: 2, name: /第 1 轮/ }));
    await visible(otherPage.getByText(/^13\s*:\s*21$/));
    for (const [label, group] of [["A 组 · 场地 1", groups[0].groupA], ["B 组 · 场地 2", groups[0].groupB]] as const) {
      const region = otherPage.getByRole("region", { name: label, exact: true });
      const names = await region.locator("tbody th").allTextContents();
      expect(names.map((name) => name.match(/E2E-\d+/)?.[0]).sort()).toEqual(group.map(playerName).sort());
    }
    await otherContext.close();
  });

  it("普通配对仍可用八人以外人数和自定义场次；全部浏览器旅程无运行错误", async () => {
    await page.goto(`${server.baseUrl}/schedule/new`);
    expect(await page.getByRole("radio", { name: /普通配对/ }).isChecked()).toBe(true);
    for (let i = 1; i <= 6; i += 1) await page.getByRole("button", { name: playerName(String(i)), exact: true }).click();
    await page.getByLabel("场次数", { exact: true }).fill("4");
    await page.getByLabel("项目名称", { exact: true }).fill("E2E 普通配对回归");
    await page.getByRole("button", { name: "创建配对项目", exact: true }).click();
    await urlIs(page, /\/schedule\/\d+$/);
    await visible(page.getByRole("heading", { name: "E2E 普通配对回归", exact: true }));
    const normalId = Number(page.url().split("/").at(-1));
    const normal = await api.get<{ project: ProjectDetail }>(`/api/schedule/projects/${normalId}`);
    expect(normal.status).toBe(200);
    expect(normal.body.project.guo).toBeUndefined();
    expect(normal.body.project.members).toHaveLength(6);
    expect(normal.body.project.fixtures).toHaveLength(4);
    await visible(page.getByText("待打 · 4", { exact: true }));
    expect(browserErrors).toEqual([]);
  });
});
