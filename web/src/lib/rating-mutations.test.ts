import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("next/cache", () => ({
  revalidatePath: vi.fn(),
}));
import { revalidatePath } from "next/cache";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync } from "fs";
import { closeDb, getDb } from "@/lib/db";
import {
  addMatch,
  addPlayer,
  listMatchesByDate,
  listPlayers,
  listRawMatches,
  mergePlayers,
  renamePlayer,
} from "@/lib/repo";
import { initializeRatingConfig, setActiveModel } from "@/lib/rating-config";
import { loadGlickoSnapshot } from "@/lib/rating-service";
import { loadRatingView, loadPredictionView } from "@/lib/rating-view";
import { projectRatingView } from "@/lib/ratings/projections";
import {
  shanghaiLocalDateFromInstant,
  weekStart,
} from "@/lib/ratings/calendar";
import {
  buildWeeklyStats,
  weeklyDataVersion,
  weeklyDataVersionContext,
} from "@/lib/weekly";
import {
  RATING_REVALIDATE_PATHS,
} from "@/lib/rating-revalidation";
import { POST as matchesPOST, DELETE as matchesDELETE } from "@/app/api/matches/route";
import { DELETE as matchDeleteById } from "@/app/api/matches/[id]/route";
import { PATCH as adminMatchPatch } from "@/app/api/admin/matches/[id]/route";
import { POST as adminMergePOST } from "@/app/api/admin/players/[id]/merge/route";
import { POST as adminRenamePOST } from "@/app/api/admin/players/[id]/rename/route";
import { DELETE as adminPlayerDELETE } from "@/app/api/admin/players/[id]/route";

const revalidatePathMock = vi.mocked(revalidatePath);

/** 每个写入口成功后必须完整调用的 revalidate 序列（与导出集合一一对应）。 */
const EXPECTED_REVALIDATE_CALLS = RATING_REVALIDATE_PATHS.map((entry) =>
  entry.type === undefined ? [entry.path] : [entry.path, entry.type]
);

function shanghaiToday(): string {
  return shanghaiLocalDateFromInstant(new Date().toISOString());
}

/** 上一周的周三（必定早于当前周，用于产生历史周 Final）。 */
function previousWeekWednesday(): string {
  const monday = weekStart(shanghaiToday());
  const date = new Date(`${monday}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() - 7 + 2);
  return date.toISOString().slice(0, 10);
}

function createPostRequest(body: object): Request {
  return new Request("http://localhost/api/matches", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function adminHeaders(): Record<string, string> {
  return { "x-admin-key": "admin-secret", "Content-Type": "application/json" };
}

/** 初始化配置：首赛季起点固定在过去季度，当前周必在某赛季内。 */
function initConfig() {
  initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, getDb());
}

describe.sequential("rating mutations 全链路一致性", () => {
  let dbPath: string;
  let players: number[] = [];

  beforeEach(() => {
    vi.clearAllMocks();
    closeDb();
    dbPath = join(tmpdir(), `test-badminton-mutations-${Date.now()}.db`);
    process.env.DATABASE_URL = dbPath;
    process.env.ADMIN_PASSWORD = "admin-secret";
    players = [
      addPlayer("A"),
      addPlayer("B"),
      addPlayer("C"),
      addPlayer("D"),
      addPlayer("E"),
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

  /** 同一 asOf 下的完整读取束：服务快照、展示投影、预测、周报与指纹。 */
  function readBundle(asOf: string) {
    const db = getDb();
    const service = loadGlickoSnapshot(db, asOf);
    if (service.state !== "ready") {
      throw new Error(`expected ready snapshot, got ${service.state}`);
    }
    const view = loadRatingView({ rating: "glicko2", asOf, db });
    if (view.model !== "glicko2") {
      throw new Error("expected glicko2 view");
    }
    const prediction = loadPredictionView({
      pa1: players[0],
      pa2: players[1],
      pb1: players[2],
      pb2: players[3],
      rating: "glicko2",
      asOf,
      db,
    });
    if (prediction.model !== "glicko2") {
      throw new Error("expected glicko2 prediction");
    }
    const weekly = buildWeeklyStats(weekStart(previousWeekWednesday()), {
      rating: "glicko2",
      asOf,
      db,
    });
    const version = weeklyDataVersion(weekly, weeklyDataVersionContext(weekly));
    return { inputHash: service.inputHash, replay: service.replay, view, prediction, weekly, version };
  }

  it("录入→改分→撤回→合并→改名：当前分、历史 Final、赛季点、周报指纹逐步一致", async () => {
    initConfig();
    const db = getDb();
    const asOf = new Date().toISOString();
    const w1 = previousWeekWednesday();
    const [A, B, C, D, E] = players;

    // 步骤 1：录入两场（都在历史周 W1）。
    const m1 = addMatch({ pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: w1 }, db);
    const m2 = addMatch({ pa1: A, pa2: C, pb1: B, pb2: D, scoreA: 21, scoreB: 10, playedAt: w1 }, db);

    let bundle = readBundle(asOf);
    let current = bundle.replay.current;
    // A 两场全胜、D 两场全负；等效对手下单场胜利 r 恰为 1000（RD 收缩），
    // 故只断言相对关系，不断言单场胜者的绝对值。
    expect(current[String(A)].r).toBeGreaterThan(1000);
    expect(current[String(A)].r).toBeGreaterThan(current[String(B)].r);
    expect(current[String(B)].r).toBeGreaterThan(current[String(D)].r);
    expect(current[String(D)].r).toBeLessThan(1000);
    expect(Object.keys(bundle.replay.matchEstimates)).toHaveLength(2);
    expect(bundle.replay.issues).toEqual([]);
    // 历史周 Final 已产生，且投影/周报都带赛季点。
    const finals = bundle.view.view.points.filter((p) => p.kind === "weekly_final");
    expect(finals.length).toBeGreaterThan(0);
    expect(finals.every((p) => p.season !== null)).toBe(true);
    expect(bundle.weekly.ratingReport).toBeDefined();
    // 同一输入 + 同一 asOf 两次读取：结果完全相同。
    expect(readBundle(asOf)).toEqual(bundle);
    let lastHash = bundle.inputHash;
    let lastVersion = bundle.version;

    // 步骤 2：改分并翻转胜负（21:15 → 15:21）。
    const patchResponse = await adminMatchPatch(
      new Request(`http://localhost/api/admin/matches/${m1}`, {
        method: "PATCH",
        headers: adminHeaders(),
        body: JSON.stringify({ scoreA: 15, scoreB: 21 }),
      }),
      { params: Promise.resolve({ id: String(m1) }) }
    );
    expect(patchResponse.status).toBe(200);

    bundle = readBundle(asOf);
    expect(bundle.inputHash).not.toBe(lastHash); // 旧指纹不复用
    expect(bundle.version).not.toBe(lastVersion);
    current = bundle.replay.current;
    // 翻转后：C 两场全胜、B 两场全负、A 与 D 一胜一负（净变化恰为 0）。
    expect(current[String(C)].r).toBeGreaterThan(1000);
    expect(current[String(C)].r).toBeGreaterThan(current[String(A)].r);
    expect(current[String(C)].r).toBeGreaterThan(current[String(B)].r);
    expect(current[String(B)].r).toBeLessThan(1000);
    expect(current[String(A)].r).toBe(1000);
    expect(current[String(D)].r).toBe(1000);
    expect(readBundle(asOf)).toEqual(bundle);
    lastHash = bundle.inputHash;
    lastVersion = bundle.version;

    // 步骤 3：撤回 M2。
    const deleteResponse = await matchDeleteById(
      new Request(`http://localhost/api/matches/${m2}`, { method: "DELETE" }),
      { params: Promise.resolve({ id: String(m2) }) }
    );
    expect(deleteResponse.status).toBe(200);

    bundle = readBundle(asOf);
    expect(bundle.inputHash).not.toBe(lastHash);
    expect(bundle.version).not.toBe(lastVersion);
    expect(bundle.replay.matchEstimates[String(m2)]).toBeUndefined();
    expect(listRawMatches(db)).toHaveLength(1);
    expect(readBundle(asOf)).toEqual(bundle);
    lastHash = bundle.inputHash;
    lastVersion = bundle.version;

    // 步骤 4：合并 D → E（E 未参赛、不产生重复参赛）。
    mergePlayers(D, E, db);

    bundle = readBundle(asOf);
    expect(bundle.inputHash).not.toBe(lastHash);
    expect(bundle.version).not.toBe(lastVersion);
    expect(bundle.replay.current[String(E)]).toBeDefined();
    expect(bundle.replay.current[String(D)]).toBeUndefined();
    expect(
      bundle.view.view.players.find((p) => p.playerId === E)?.name
    ).toBe("E");
    expect(readBundle(asOf)).toEqual(bundle);
    lastHash = bundle.inputHash;
    lastVersion = bundle.version;

    // 步骤 5：改名 E。
    renamePlayer(E, "E-star", db);

    bundle = readBundle(asOf);
    expect(bundle.inputHash).not.toBe(lastHash);
    expect(bundle.version).not.toBe(lastVersion);
    expect(
      bundle.view.view.players.find((p) => p.playerId === E)?.name
    ).toBe("E-star");
    // 改名后当前分不变（指纹只因姓名变化）。
    expect(bundle.replay.current[String(E)]).toEqual(
      readBundle(asOf).replay.current[String(E)]
    );
    expect(readBundle(asOf)).toEqual(bundle);
    void lastHash;
    void lastVersion;
  });

  it("合并制造同场重复参赛 ID：旧事实保留、引擎诊断 duplicate_player、该场不计分", () => {
    initConfig();
    const db = getDb();
    const asOf = new Date().toISOString();
    const w1 = previousWeekWednesday();
    const [A, B, C, D, E] = players;

    const m1 = addMatch({ pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: w1 }, db);
    const m2 = addMatch({ pa1: A, pa2: C, pb1: B, pb2: E, scoreA: 21, scoreB: 10, playedAt: w1 }, db);

    // C 并入 A：M1（A+B vs C+D）与 M2（A+C vs B+E）都出现重复 A。
    mergePlayers(C, A, db);

    // 旧事实不丢：原始比赛仍在（合并只重写参与者指向，比分/日期不变）。
    const raws = listRawMatches(db);
    expect(raws).toHaveLength(2);
    expect(raws.find((row) => row.id === m1)).toMatchObject({
      pa1: A,
      pa2: B,
      pb1: A,
      pb2: D,
      scoreA: 21,
      scoreB: 15,
      playedAt: w1,
    });
    // 事实浏览页不隐藏异常记录。
    expect(listMatchesByDate(db)).toHaveLength(2);

    const service = loadGlickoSnapshot(db, asOf);
    if (service.state !== "ready") throw new Error("expected ready");
    // 引擎如实诊断两场 duplicate_player；被诊断的比赛不计分。
    expect(service.replay.issues).toEqual([
      { matchId: m1, reason: "duplicate_player" },
      { matchId: m2, reason: "duplicate_player" },
    ]);
    expect(Object.keys(service.replay.matchEstimates)).toHaveLength(0);
    expect(service.replay.current).toEqual({});
  });

  it("读取无副作用：连续读取/投影/预测/跨连接读取均不产生重复结算", () => {
    initConfig();
    const db = getDb();
    const w1 = previousWeekWednesday();
    const [A, B, C, D] = players;
    addMatch({ pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: w1 }, db);
    addMatch({ pa1: A, pa2: C, pb1: B, pb2: D, scoreA: 18, scoreB: 21, playedAt: w1 }, db);
    const asOf = new Date().toISOString();

    const first = loadGlickoSnapshot(db, asOf);
    if (first.state !== "ready") throw new Error("expected ready");
    // 连续读取：同连接 memo 命中，深度相等。
    expect(loadGlickoSnapshot(db, asOf)).toEqual(first);

    // 连续投影：不重复结算（Final/Estimated 点数量稳定）。
    const playersNow = listPlayers(db);
    const view1 = projectRatingView(first.replay, playersNow);
    const view2 = projectRatingView(first.replay, playersNow);
    expect(view2).toEqual(view1);
    expect(view1.points.filter((p) => p.kind === "weekly_final")).toHaveLength(4);
    expect(view1.points.filter((p) => p.kind === "match_estimated")).toHaveLength(8);

    // 连续预测：不修改原状态。
    const predict = () =>
      loadPredictionView({ pa1: A, pa2: B, pb1: C, pb2: D, rating: "glicko2", asOf, db });
    const prediction1 = predict();
    expect(predict()).toEqual(prediction1);
    expect(loadGlickoSnapshot(db, asOf)).toEqual(first);

    // 跨连接读取（模拟重启）：走 meta 最近成功快照，同样无重复结算。
    closeDb();
    const reopened = getDb();
    const again = loadGlickoSnapshot(reopened, asOf);
    if (again.state !== "ready") throw new Error("expected ready after reopen");
    expect(again.replay).toEqual(first.replay);
    expect(again.replay.asOf).toBe(first.replay.asOf);
  });

  it("activeModel=legacy 时老接口可运行且 rating 字段如实；切 glicko2 后同 asOf 不混模型", async () => {
    const db = getDb();
    const asOf = new Date().toISOString();
    const [A, B, C, D] = players;
    const matchBody = { pa1: A, pa2: B, pb1: C, pb2: D, scoreA: 21, scoreB: 15, playedAt: shanghaiToday() };

    // 未初始化配置：POST rating 如实 not_effective，Legacy 字段照常。
    const post1 = await matchesPOST(createPostRequest(matchBody));
    expect(post1.status).toBe(201);
    const json1 = await post1.json();
    expect(json1.rating).toEqual({
      state: "not_effective",
      model: "glicko2",
      reason: "rating config not initialized",
    });
    expect(json1.before).toHaveLength(4);
    expect(json1.after).toHaveLength(4);

    // 撤回（10 分钟窗口内）。
    const deleteResponse = await matchDeleteById(
      new Request(`http://localhost/api/matches/${json1.id}`, { method: "DELETE" }),
      { params: Promise.resolve({ id: String(json1.id) }) }
    );
    expect(deleteResponse.status).toBe(200);

    // 重新录入并保留，供改分。
    const post2 = await matchesPOST(createPostRequest(matchBody));
    expect(post2.status).toBe(201);
    const json2 = await post2.json();

    // admin 改分（复用整数比分规则）。
    const patchResponse = await adminMatchPatch(
      new Request(`http://localhost/api/admin/matches/${json2.id}`, {
        method: "PATCH",
        headers: adminHeaders(),
        body: JSON.stringify({ scoreA: 21, scoreB: 9 }),
      }),
      { params: Promise.resolve({ id: String(json2.id) }) }
    );
    expect(patchResponse.status).toBe(200);
    const badPatch = await adminMatchPatch(
      new Request(`http://localhost/api/admin/matches/${json2.id}`, {
        method: "PATCH",
        headers: adminHeaders(),
        body: JSON.stringify({ scoreA: 21.5, scoreB: 9 }),
      }),
      { params: Promise.resolve({ id: String(json2.id) }) }
    );
    expect(badPatch.status).toBe(400);

    // 合并与改名（无比赛的新球员）。
    const spare1 = addPlayer("S1");
    const spare2 = addPlayer("S2");
    const mergeResponse = await adminMergePOST(
      new Request(`http://localhost/api/admin/players/${spare1}/merge`, {
        method: "POST",
        headers: adminHeaders(),
        body: JSON.stringify({ toId: spare2 }),
      }),
      { params: Promise.resolve({ id: String(spare1) }) }
    );
    expect(mergeResponse.status).toBe(200);
    const renameResponse = await adminRenamePOST(
      new Request(`http://localhost/api/admin/players/${spare2}/rename`, {
        method: "POST",
        headers: adminHeaders(),
        body: JSON.stringify({ name: "S2-renamed" }),
      }),
      { params: Promise.resolve({ id: String(spare2) }) }
    );
    expect(renameResponse.status).toBe(200);

    // 默认视图仍是 Legacy（无配置）。
    expect(loadRatingView({ asOf, db }).model).toBe("legacy");

    // 初始化配置（默认 activeModel=legacy）：默认消费者仍 Legacy，POST rating 按配置状态 ready。
    initConfig();
    expect(loadRatingView({ asOf, db }).model).toBe("legacy");
    const legacyPrediction = loadPredictionView({ ...matchBody, rating: undefined, asOf, db });
    expect(legacyPrediction.model).toBe("legacy");
    const legacyWeekly = buildWeeklyStats(weekStart(shanghaiToday()), { asOf, db });
    expect(legacyWeekly.ratingReport).toBeUndefined();

    const post3 = await matchesPOST(createPostRequest(matchBody));
    expect(post3.status).toBe(201);
    const json3 = await post3.json();
    expect(json3.rating.state).toBe("ready");
    expect(json3.rating.model).toBe("glicko2");

    // 切 glicko2：同一 asOf 下 view/预测/周报 model 字段一致，不混模型。
    setActiveModel("glicko2", db);
    const glickoView = loadRatingView({ asOf, db });
    expect(glickoView.model).toBe("glicko2");
    const glickoPrediction = loadPredictionView({ ...matchBody, asOf, db });
    expect(glickoPrediction.model).toBe("glicko2");
    const glickoWeekly = buildWeeklyStats(weekStart(shanghaiToday()), { asOf, db });
    expect(glickoWeekly.ratingReport).toBeDefined();
    expect(glickoWeekly.ratingReport?.asOf).toBe(glickoView.asOf);
    // 显式 legacy 查看选项仍保留。
    expect(loadRatingView({ rating: "legacy", asOf, db }).model).toBe("legacy");
  });

  it.each([
    [
      "POST /api/matches",
      async () =>
        matchesPOST(
          createPostRequest({
            pa1: players[0],
            pa2: players[1],
            pb1: players[2],
            pb2: players[3],
            scoreA: 21,
            scoreB: 15,
            playedAt: shanghaiToday(),
          })
        ),
    ],
    [
      "DELETE /api/matches?id=",
      async () => {
        const id = addMatch(
          { pa1: players[0], pa2: players[1], pb1: players[2], pb2: players[3], scoreA: 21, scoreB: 15, playedAt: shanghaiToday() },
          getDb()
        );
        return matchesDELETE(
          new Request(`http://localhost/api/matches?id=${id}`, { method: "DELETE" })
        );
      },
    ],
    [
      "DELETE /api/matches/[id]",
      async () => {
        const id = addMatch(
          { pa1: players[0], pa2: players[1], pb1: players[2], pb2: players[3], scoreA: 21, scoreB: 15, playedAt: shanghaiToday() },
          getDb()
        );
        return matchDeleteById(
          new Request(`http://localhost/api/matches/${id}`, { method: "DELETE" }),
          { params: Promise.resolve({ id: String(id) }) }
        );
      },
    ],
    [
      "PATCH /api/admin/matches/[id]",
      async () => {
        const id = addMatch(
          { pa1: players[0], pa2: players[1], pb1: players[2], pb2: players[3], scoreA: 21, scoreB: 15, playedAt: shanghaiToday() },
          getDb()
        );
        return adminMatchPatch(
          new Request(`http://localhost/api/admin/matches/${id}`, {
            method: "PATCH",
            headers: adminHeaders(),
            body: JSON.stringify({ scoreA: 21, scoreB: 9 }),
          }),
          { params: Promise.resolve({ id: String(id) }) }
        );
      },
    ],
    [
      "POST /api/admin/players/[id]/merge",
      () =>
        adminMergePOST(
          new Request(`http://localhost/api/admin/players/${players[3]}/merge`, {
            method: "POST",
            headers: adminHeaders(),
            body: JSON.stringify({ toId: players[4] }),
          }),
          { params: Promise.resolve({ id: String(players[3]) }) }
        ),
    ],
    [
      "POST /api/admin/players/[id]/rename",
      () =>
        adminRenamePOST(
          new Request(`http://localhost/api/admin/players/${players[3]}/rename`, {
            method: "POST",
            headers: adminHeaders(),
            body: JSON.stringify({ name: "Renamed" }),
          }),
          { params: Promise.resolve({ id: String(players[3]) }) }
        ),
    ],
    [
      "DELETE /api/admin/players/[id]",
      async () => {
        const spare = addPlayer("Spare", getDb());
        return adminPlayerDELETE(
          new Request(`http://localhost/api/admin/players/${spare}`, {
            method: "DELETE",
            headers: adminHeaders(),
          }),
          { params: Promise.resolve({ id: String(spare) }) }
        );
      },
    ],
  ])("写入口 %s 成功后调用全部评分 revalidate 路径", async (_name, invoke) => {
    revalidatePathMock.mockClear();
    const response = await invoke();
    expect(response.status).toBeLessThan(300);
    expect(revalidatePathMock.mock.calls).toEqual(EXPECTED_REVALIDATE_CALLS);
    expect(revalidatePathMock).toHaveBeenCalledTimes(RATING_REVALIDATE_PATHS.length);
  });
});
