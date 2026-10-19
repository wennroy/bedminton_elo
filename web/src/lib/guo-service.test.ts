import { afterEach, describe, expect, it, vi } from "vitest";
vi.mock("next/cache", () => ({ revalidatePath: vi.fn() }));
import { randomUUID } from "node:crypto";
import Database from "better-sqlite3";
import { createDb } from "./db";
import { addPlayer } from "./repo";
import { createProjectService, regenerateProjectService, recordFixtureScoreService, markPlayedService, mergePlayerWithScheduleRefs, toScheduleHttp, noteMatchScoreEdited } from "./schedule-service";
import { detachScore, getProject, listProjects, regenerate, type ProjectDetail } from "./schedule-projects";
import { createGuoSchedule, nextGuoGroups, type GuoSnapshot } from "./scheduling/guo";
import { initializeRatingConfig, setActiveModel } from "./rating-config";
import * as ratingService from "./rating-service";

const opened: Database.Database[] = [];
afterEach(() => { vi.restoreAllMocks(); for (const db of opened.splice(0)) db.close(); });
function setup() {
  const db = createDb(":memory:"); opened.push(db);
  const ids = Array.from({length: 9}, (_, i) => String(addPlayer(`P${i}`, db)));
  const payload = { mode: "guo", playerIds: ids.slice(0, 8), actorId: ids[0], model: "legacy", requestId: randomUUID() };
  return { db, ids, payload, project: createProjectService({ ...payload, db }).project };
}
function complete(db: Database.Database, project: ProjectDetail) {
  for (const f of project.fixtures.filter((f) => f.status === "pending")) {
    recordFixtureScoreService({ db, projectId: project.id, fixtureId: f.id, expectedFixtureRevision: f.revision, scoreA: 21, scoreB: 15 + f.displayNo % 4, actorId: project.createdBy, playedAt: "2026-10-19" });
  }
  return getProject(project.id, db);
}
function advance(db: Database.Database, project: ProjectDetail, overrides = {}) {
  return regenerateProjectService({ db, projectId: project.id, actorId: project.createdBy, expectedRevision: project.revision, requestId: randomUUID(), ...overrides });
}
function errorCode(fn: () => unknown, status: number, code: string) {
  try { fn(); throw new Error("expected failure"); } catch (e) { const http = toScheduleHttp(e); expect(http.status).toBe(status); expect(http.body.code).toBe(code); }
}

describe("郭老师方法的项目生命周期", () => {
  it("八人固定名单、六场、持久抽签、可幂等重试（省略 seed）", () => {
    const {db, payload, project} = setup();
    expect(project.guo?.rounds).toHaveLength(1);
    expect(project.fixtures).toHaveLength(6);
    expect(project.guo?.canAdvance).toBe(false);
    const input = project.latestGeneration!.input as GuoSnapshot;
    expect(input.mode).toBe("guo"); expect(input.schedule).toHaveLength(6);
    expect(listProjects({}, db).projects[0].mode).toBe("guo");
    const retry = createProjectService({...payload, db});
    expect(retry.replayed).toBe(true); expect(retry.project.id).toBe(project.id);
    expect(retry.project.latestGeneration!.input).toEqual(input);
    errorCode(() => createProjectService({...payload, name: "different", db}), 409, "idempotency_conflict");
    errorCode(() => createProjectService({...payload, playerIds: payload.playerIds.slice(1), requestId: randomUUID(), db}), 400, "validation_error");
    errorCode(() => createProjectService({...payload, matches: 7, requestId: randomUUID(), db}), 400, "validation_error");
    errorCode(() => createProjectService({...payload, mode: "typo", requestId: randomUUID(), db}), 400, "validation_error");
  });

  it("按站点当前 Glicko 评分分组，保留有限真实队伍胜率，评分服务失效不阻止旧请求重放", () => {
    const {db, ids} = setup();
    initializeRatingConfig({firstSeasonStart: "2026-07-01"}, db);
    setActiveModel("glicko2", db);
    const actual = ratingService.loadGlickoSnapshot(db, new Date().toISOString());
    if (actual.state !== "ready") throw new Error("expected ready rating snapshot");
    const current = Object.fromEntries(ids.map((id, index) => [id, {r: 1000 + index * 100, rd: 100, volatility: 0.06}]));
    const spy = vi.spyOn(ratingService,"loadGlickoSnapshot").mockReturnValue({...actual, replay: {...actual.replay, current}});
    const payload = {mode:"guo", playerIds:ids.slice(0,8), actorId:ids[0], requestId:randomUUID(), db};
    const created = createProjectService(payload).project;
    expect(created.model).toBe("glicko2");
    expect(created.guo!.rounds[0].groupA).toEqual(ids.slice(4,8).reverse());
    expect(created.fixtures.every((f)=> Number.isFinite(f.generatedWinRate) && f.generatedWinRate! > 0 && f.generatedWinRate! < 1)).toBe(true);
    const snapshot = created.latestGeneration!.input as GuoSnapshot & {rating: {model: string; playerRatings: unknown}};
    expect(snapshot.rating.model).toBe("glicko2");
    expect(snapshot.rating.playerRatings).toEqual(Object.fromEntries(ids.slice(0,8).map((id)=>[id,current[id]])));
    spy.mockImplementation(()=> {throw new Error("rating temporarily unavailable");});
    expect(createProjectService(payload).replayed).toBe(true);
    errorCode(()=>createProjectService({...payload,requestId:randomUUID()}),409,"rating_unavailable");
    const finished = complete(db,created);
    spy.mockReturnValue({...actual, replay: {...actual.replay, current}});
    const requestId = randomUUID();
    const result = advance(db,finished,{requestId});
    spy.mockImplementation(()=> {throw new Error("rating temporarily unavailable");});
    expect(advance(db,finished,{requestId}).replayed).toBe(true);
    expect(result.project.guo!.rounds).toHaveLength(2);
  });

  it("所有场次录分才可续轮，标记已打无比分仍不能续轮，拒绝替换名单", () => {
    const {db, project, ids} = setup();
    errorCode(() => advance(db, project), 409, "state_conflict");
    errorCode(() => advance(db, project, {playerIds: ids.slice(1), matches: 0}), 400, "validation_error");
    const f = project.fixtures[0];
    markPlayedService({db, projectId: project.id, fixtureId:f.id, expectedFixtureRevision:0, actorId:ids[0]});
    errorCode(() => advance(db, getProject(project.id, db)), 409, "state_conflict");
  });

  it("四轮依次正常升降、交叉、正常升降；排名只取本轮；CAS 与重试安全", () => {
    const {db, project: initial} = setup();
    let project = complete(db, initial);
    for (let round = 2; round <= 4; round++) {
      const previous = project.guo!.rounds.at(-1)!;
      const kind = round === 3 ? "cross" : "promotion";
      const groups = nextGuoGroups(previous.standingsA, previous.standingsB, kind);
      const requestId = randomUUID();
      const revision = project.revision;
      errorCode(() => advance(db, project, {expectedRevision: revision - 1}), 409, "revision_conflict");
      const result = advance(db, project, {requestId});
      project = result.project;
      const next = project.guo!.rounds.at(-1)!;
      expect(next.kind).toBe(kind); expect(next.groupA).toEqual(groups.groupA); expect(next.groupB).toEqual(groups.groupB);
      expect(next.standingsA.every((s) => s.played === 0 && s.wins === 0)).toBe(true);
      expect(result.supersededCount).toBe(0);
      const retry = advance(db, project, {requestId, expectedRevision: revision});
      expect(retry.replayed).toBe(true); expect(retry.generationId).toBe(result.generationId);
      project = complete(db, project);
    }
    expect(project.fixtures).toHaveLength(24); expect(project.guo!.canAdvance).toBe(false);
    expect(project.guo!.nextRoundKind).toBeNull();
    errorCode(() => advance(db, project), 409, "state_conflict");
  });

  it("冻结场次胜率用于超预期；旧轮改分或撤回不改写已生成的下一轮", () => {
    const {db, project: initial} = setup();
    const finished = complete(db, initial);
    for (const row of finished.guo!.rounds[0].standingsA) {
      const matches = finished.fixtures.filter((f) => [f.pa1,f.pa2,f.pb1,f.pb2].includes(row.playerId));
      const expected = matches.reduce((sum,f) => sum + ([f.pa1,f.pa2].includes(row.playerId) ? f.generatedWinRate! : 1-f.generatedWinRate!),0);
      expect(row.expectedWins).toBeCloseTo(expected); expect(row.overperformance).toBeCloseTo(row.wins - expected);
    }
    const second = advance(db, finished).project;
    const snapshot = structuredClone(second.latestGeneration!.input);
    const first = second.fixtures[0];
    db.prepare("UPDATE matches SET score_a = 1, score_b = 21 WHERE id = ?").run(first.matchId);
    noteMatchScoreEdited(first.matchId!, db);
    expect(getProject(second.id,db).latestGeneration!.input).toEqual(snapshot);
    detachScore({projectId:second.id,fixtureId:first.id,matchId:first.matchId!},db);
    expect(getProject(second.id,db).latestGeneration!.input).toEqual(snapshot);
    expect(getProject(second.id,db).guo!.rounds[0].complete).toBe(false);
  });

  it("存储层也拒绝以普通续排绕过八人及轮次条件", () => {
    const {db,project,ids} = setup();
    errorCode(() => regenerate({projectId:project.id,expectedRevision:0,requestId:randomUUID(),payloadHash:"bypass",newPlayerIds:ids.slice(1),engineInput:{},engineResult:null,actorId:ids[0]},db),409,"state_conflict");
    expect(getProject(project.id,db).fixtures).toHaveLength(6);
    const finished = complete(db,project);
    const previous = finished.guo!.rounds[0];
    const original = finished.latestGeneration!.input as GuoSnapshot;
    const groups = nextGuoGroups(previous.standingsA,previous.standingsB,"promotion");
    const schedule = createGuoSchedule(groups.groupA,groups.groupB);
    const snapshot = {...original,...groups,schedule,kind:"promotion" as const,roundNo:2,sourceStandings:{groupA:previous.standingsA,groupB:previous.standingsB}};
    // Correct teams alone must not permit changing the persistent draw or omitting forecasts.
    errorCode(() => regenerate({projectId:project.id,expectedRevision:finished.revision,requestId:randomUUID(),payloadHash:"tampered-draw",newPlayerIds:ids.slice(0,8),engineInput:{...snapshot,drawOrder:{}},engineResult:{schedule,metrics:{}},fixtureWinRates:Array(6).fill(0.5),actorId:ids[0]},db),409,"state_conflict");
    errorCode(() => regenerate({projectId:project.id,expectedRevision:finished.revision,requestId:randomUUID(),payloadHash:"missing-forecast",newPlayerIds:ids.slice(0,8),engineInput:snapshot,engineResult:{schedule,metrics:{}},actorId:ids[0]},db),409,"state_conflict");
  });

  it("合并球员同时迁移郭老师快照标识，保留排名与抽签、胜率数值", () => {
    const {db,project:initial,ids} = setup();
    const finished = complete(db,initial);
    const project = advance(db,finished).project;
    const old = project.latestGeneration!.input as GuoSnapshot;
    mergePlayerWithScheduleRefs(Number(ids[0]),Number(ids[8]),db);
    const updated = getProject(project.id,db);
    const snapshot = updated.latestGeneration!.input as GuoSnapshot;
    expect([...snapshot.groupA,...snapshot.groupB]).toContain(ids[8]);
    expect([...snapshot.groupA,...snapshot.groupB]).not.toContain(ids[0]);
    expect(snapshot.drawOrder[ids[8]]).toBe(old.drawOrder[ids[0]]);
    expect(snapshot.sourceStandings!.groupA.concat(snapshot.sourceStandings!.groupB).find((r)=>r.playerId===ids[8])?.wins).toBe(old.sourceStandings!.groupA.concat(old.sourceStandings!.groupB).find((r)=>r.playerId===ids[0])?.wins);
    expect(updated.guo!.rounds[0].standingsA.concat(updated.guo!.rounds[0].standingsB).find((r)=>r.playerId===ids[8])?.played).toBe(3);
  });
});
