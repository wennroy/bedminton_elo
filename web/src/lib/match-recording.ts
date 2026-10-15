import type Database from "better-sqlite3";
import { getDb } from "./db";
import * as repo from "./repo";
import {
  loadGlickoSnapshot,
  RATING_CONFIG_MISSING_REASON,
} from "./rating-service";
import { revalidateRatingPages } from "./rating-revalidation";
import { attachScore, detachScore, type FixtureDto } from "./schedule-projects";
import { shanghaiLocalDateFromInstant } from "./ratings/calendar";
import type { MatchEstimate } from "./ratings/types";

/**
 * 公共比赛录分服务：手动录入（/api/matches）与项目记分（fixture score）
 * 共用同一条「写入 + 评分结果构建 + revalidation」链路，不复制三套评分代码。
 *
 * 核心约定（与旧 POST 行为一致）：
 * - 校验/写入失败原样抛出，由调用方映射 HTTP；
 * - 事务提交后构建 legacy before/after 与 Glicko MatchRatingField，
 *   这两步失败绝不把已保存的比赛伪装成失败响应（降级为空数组 / pending）；
 * - revalidation 失败同样不回滚已提交事实。
 */

/** 响应中的新版评分判别字段（共享契约：ready/pending/not_effective）。 */
export type MatchRatingField =
  | {
      state: "ready";
      model: "glicko2";
      configVersion: string;
      /** 本次请求捕获的评分时点，贯穿整场评分。 */
      asOf: string;
      segmentId: string;
      /** 该 matchId 自身的单场变化（补录导致的下游当前分变化不冒充该场变化）。 */
      estimate: MatchEstimate;
      /** 该场落在历史区段、触发下游重算（estimate.segmentId !== 当前区段）。 */
      historyRecomputed: boolean;
    }
  | { state: "pending"; model: "glicko2"; reason: string }
  | { state: "not_effective"; model: "glicko2"; reason: string };

/** legacy before/after 的单人条目（旧 ELO 展示口径）。 */
export interface LegacyRatingEntry {
  id: number;
  name: string;
  elo: number;
}

/**
 * 写入成功后的新版评分判别：
 * - ready：服务 ready 且重放含该 matchId 的单场事件；
 * - not_effective：ready 但无该场事件（未来日期未生效），或新版配置未初始化；
 * - pending：服务失败（unavailable 计算失败 / stale），比赛已保存、积分稍后更新。
 */
export function buildMatchRatingField(
  db: Database.Database,
  asOf: string,
  matchId: number,
  playedAt: string
): MatchRatingField {
  let result;
  try {
    result = loadGlickoSnapshot(db, asOf);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return {
      state: "pending",
      model: "glicko2",
      reason: `rating service failed: ${message}`,
    };
  }

  if (result.state === "ready") {
    const estimate = result.replay.matchEstimates[String(matchId)];
    if (estimate === undefined) {
      const asOfDate = shanghaiLocalDateFromInstant(asOf);
      return {
        state: "not_effective",
        model: "glicko2",
        reason:
          playedAt > asOfDate
            ? `match date ${playedAt} is after as-of date ${asOfDate}; not yet effective`
            : "match did not produce a glicko2 rating event",
      };
    }
    return {
      state: "ready",
      model: "glicko2",
      configVersion: result.replay.configVersion,
      asOf,
      segmentId: estimate.segmentId,
      estimate,
      historyRecomputed: estimate.segmentId !== result.replay.currentSegment.id,
    };
  }

  if (result.state === "stale") {
    return {
      state: "pending",
      model: "glicko2",
      reason: `rating service stale: ${result.reason}`,
    };
  }

  if (result.reason === RATING_CONFIG_MISSING_REASON) {
    return {
      state: "not_effective",
      model: "glicko2",
      reason: RATING_CONFIG_MISSING_REASON,
    };
  }
  return { state: "pending", model: "glicko2", reason: result.reason };
}

function ratingsForPlayers(
  ratings: Map<number, repo.PlayerRatings>,
  names: Map<number, string>,
  ids: number[]
): LegacyRatingEntry[] {
  return ids.map((id) => ({
    id,
    name: names.get(id) ?? "?",
    elo: ratings.get(id)?.elo ?? 1000,
  }));
}

/** 项目记分：recordMatch 同事务内 attachScore 所需的关联参数。 */
export interface FixtureLinkInput {
  projectId: number;
  fixtureId: number;
  expectedFixtureRevision: number;
  actorId: string;
}

export interface RecordMatchInput {
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
  enteredBy?: number | null;
  db?: Database.Database;
  /** 项目记分：插入 matches 与 attachScore 在同一事务内完成。 */
  fixtureLink?: FixtureLinkInput;
}

export interface RecordMatchResult {
  id: number;
  before: LegacyRatingEntry[];
  after: LegacyRatingEntry[];
  legacyRatingsAvailable: boolean;
  rating: MatchRatingField;
  /** fixtureLink 时返回 attachScore 之后的场次状态（recorded）。 */
  fixture?: FixtureDto;
}

function resolveDb(db?: Database.Database): Database.Database {
  return db ?? getDb();
}

/**
 * 录入一场比赛并构建评分结果。
 *
 * 顺序：legacy before（可失败降级）→ 插入 matches（可选同事务关联 fixture）→
 * legacy after（可失败降级）→ Glicko MatchRatingField（可失败降级）→ revalidation。
 * 只有校验与写入失败才向上抛；评分构建失败只降级字段。
 */
export function recordMatch(input: RecordMatchInput): RecordMatchResult {
  const conn = resolveDb(input.db);
  const ids = [input.pa1, input.pa2, input.pb1, input.pb2];

  // asOf 在写入前一次捕获，贯穿本次评分。
  const asOf = new Date().toISOString();

  // Legacy before/after 是可失败的附加计算：不参加写入事务、不阻止写入。
  let names: Map<number, string> = new Map();
  let before: LegacyRatingEntry[] = [];
  let legacyRatingsAvailable = true;
  try {
    names = new Map(
      repo.listPlayers(conn).map((p) => [p.id, p.name] as const)
    );
  } catch {
    names = new Map();
  }
  try {
    before = ratingsForPlayers(repo.recomputeAllRatings(conn), names, ids);
  } catch {
    before = [];
    legacyRatingsAvailable = false;
  }

  let id: number;
  let fixture: FixtureDto | undefined;
  if (input.fixtureLink) {
    const link = input.fixtureLink;
    // 插入 matches 与 attachScore 必须同事务：attach 失败（状态/revision 竞态、
    // 阵容不一致）时比赛行一并回滚，绝不留孤儿比赛。
    const tx = conn.transaction(() => {
      const matchId = repo.addMatch(
        {
          pa1: input.pa1,
          pa2: input.pa2,
          pb1: input.pb1,
          pb2: input.pb2,
          scoreA: input.scoreA,
          scoreB: input.scoreB,
          playedAt: input.playedAt,
          enteredBy: input.enteredBy,
        },
        conn
      );
      const fx = attachScore(
        {
          projectId: link.projectId,
          fixtureId: link.fixtureId,
          expectedFixtureRevision: link.expectedFixtureRevision,
          matchId,
          actorId: link.actorId,
          playedAt: input.playedAt,
        },
        conn
      );
      return { matchId, fx };
    });
    const committed = tx.immediate();
    id = committed.matchId;
    fixture = committed.fx;
  } else {
    id = repo.addMatch(
      {
        pa1: input.pa1,
        pa2: input.pa2,
        pb1: input.pb1,
        pb2: input.pb2,
        scoreA: input.scoreA,
        scoreB: input.scoreB,
        playedAt: input.playedAt,
        enteredBy: input.enteredBy,
      },
      conn
    );
  }

  let after: LegacyRatingEntry[] = [];
  try {
    after = ratingsForPlayers(repo.recomputeAllRatings(conn), names, ids);
  } catch {
    after = [];
    legacyRatingsAvailable = false;
  }

  const rating = buildMatchRatingField(conn, asOf, id, input.playedAt);
  try {
    revalidateRatingPages();
  } catch {
    // revalidation 失败不影响已保存的比赛事实。
  }

  return { id, before, after, legacyRatingsAvailable, rating, fixture };
}

export interface WithdrawMatchResult {
  /** 该比赛曾被 fixture 引用时，返回解除关联后的场次（played_unscored）。 */
  fixture: FixtureDto | null;
}

/**
 * 撤回比赛：若 matches 行被 fixture 引用，先 detachScore（recorded →
 * played_unscored、解除 match_id、保留 played_at、双 revision++）再删比赛；
 * 全程一个事务，回滚不留半解除状态。未被引用时等同普通删除。
 *
 * /api/matches 两条撤回路由与 repo.deleteMatch 统一走这里。
 */
export function withdrawMatch(
  matchId: number,
  db?: Database.Database
): WithdrawMatchResult {
  const conn = resolveDb(db);
  const link = conn
    .prepare(
      `SELECT id AS fixtureId, project_id AS projectId
       FROM schedule_fixtures WHERE match_id = ?`
    )
    .get(matchId) as { fixtureId: number; projectId: number } | undefined;

  if (!link) {
    conn.prepare(`DELETE FROM matches WHERE id = ?`).run(matchId);
    return { fixture: null };
  }

  let fixture: FixtureDto | null = null;
  const tx = conn.transaction(() => {
    fixture = detachScore(
      {
        projectId: link.projectId,
        fixtureId: link.fixtureId,
        matchId,
      },
      conn
    );
    conn.prepare(`DELETE FROM matches WHERE id = ?`).run(matchId);
  });
  tx.immediate();
  return { fixture };
}
