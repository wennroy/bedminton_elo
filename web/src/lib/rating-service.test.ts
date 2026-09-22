import { describe, it, expect, beforeEach, afterEach, vi, type MockInstance } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { createDb } from "./db";
import { addMatch, addPlayer, deleteMatch, mergePlayers, renamePlayer } from "./repo";
import { initializeRatingConfig } from "./rating-config";
import {
  loadGlickoSnapshot,
  ratingLastGoodKey,
  type RatingServiceResult,
} from "./rating-service";
import { ratingConfigVersion } from "./ratings/config";
import * as replayModule from "./ratings/replay";
import type { RatingReplay } from "./ratings/types";

const FIRST_SEASON_START = "2026-10-01";
const AS_OF_WEEK_1 = "2026-10-06T20:00:00+08:00"; // 周二，区段 2026-10-05 起
const AS_OF_WEEK_2 = "2026-10-12T20:00:00+08:00"; // 下周一，跨周界
const AS_OF_SEASON_1 = "2026-12-29T20:00:00+08:00"; // 季界前
const AS_OF_SEASON_2 = "2027-01-02T20:00:00+08:00"; // 跨 2027-01-01 季界

let dir: string;
let db: Database.Database;
let players: number[];
let replaySpy: MockInstance<typeof replayModule.replayRatings>;

function ready(result: RatingServiceResult): RatingReplay {
  if (result.state !== "ready") {
    throw new Error(`expected ready, got ${result.state}: ${"reason" in result ? result.reason : ""}`);
  }
  return result.replay;
}

function configVersionOf(db: Database.Database): string {
  const row = db
    .prepare(`SELECT value FROM meta WHERE key = 'ratings.config.v1'`)
    .get() as { value: string };
  return ratingConfigVersion(JSON.parse(row.value).config);
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "rating-service-test-"));
  db = createDb(join(dir, "test.db"));
  initializeRatingConfig({ firstSeasonStart: FIRST_SEASON_START }, db);
  players = [
    addPlayer("A", db),
    addPlayer("B", db),
    addPlayer("C", db),
    addPlayer("D", db),
  ];
  replaySpy = vi.spyOn(replayModule, "replayRatings");
});

afterEach(() => {
  replaySpy.mockRestore();
  try {
    db.close();
  } catch {
    // 个别用例中途已关闭连接
  }
  rmSync(dir, { recursive: true, force: true });
});

function seedWeekMatches(playedAt: string): number[] {
  return [
    addMatch(
      { pa1: players[0], pa2: players[1], pb1: players[2], pb2: players[3], scoreA: 21, scoreB: 15, playedAt },
      db
    ),
    addMatch(
      { pa1: players[2], pa2: players[3], pb1: players[0], pb2: players[1], scoreA: 21, scoreB: 19, playedAt },
      db
    ),
  ];
}

describe("loadGlickoSnapshot 基础语义", () => {
  it("无配置（未初始化）→ unavailable，不冒充 Legacy", () => {
    const fresh = createDb(join(dir, "fresh.db"));
    const result = loadGlickoSnapshot(fresh, AS_OF_WEEK_1);
    expect(result).toEqual({
      state: "unavailable",
      model: "glicko2",
      reason: "rating config not initialized",
    });
    fresh.close();
  });

  it("有配置但无比赛历史 → ready（空历史可正常重放）", () => {
    const result = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(result.state).toBe("ready");
    if (result.state === "ready") {
      expect(result.replay.current).toEqual({});
      expect(result.replay.events).toEqual([]);
      expect(result.replay.currentSegment.id).toBe("2026-10-05:2026-10-05");
    }
  });

  it("引用未知球员的比赛行不被原始查询丢掉，引擎按 unknown_player 诊断", () => {
    const matchId = addMatch(
      { pa1: players[0], pa2: players[1], pb1: players[2], pb2: players[3], scoreA: 21, scoreB: 15, playedAt: "2026-10-05" },
      db
    );
    // 绕过外键校验直接删除球员，制造悬空引用（模拟历史脏数据）。
    db.pragma("foreign_keys = OFF");
    db.prepare(`DELETE FROM players WHERE id = ?`).run(players[3]);
    db.pragma("foreign_keys = ON");

    const result = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(result.state).toBe("ready");
    if (result.state === "ready") {
      expect(result.replay.issues).toEqual([{ matchId, reason: "unknown_player" }]);
      expect(result.replay.matchEstimates).toEqual({});
    }
  });
});

describe("同请求 memo 与确定性", () => {
  it("同输入同 asOf 两次读取只计算一次，数值完全一致", () => {
    seedWeekMatches("2026-10-05");
    const first = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    const second = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(replaySpy).toHaveBeenCalledTimes(1);
    expect(second.state).toBe("ready");
    if (first.state === "ready" && second.state === "ready") {
      expect(second.replay).toEqual(first.replay);
      expect(second.inputHash).toBe(first.inputHash);
    }
  });

  it("memo 只缓存成功结果：失败后再读取会重试计算", () => {
    seedWeekMatches("2026-10-05");
    replaySpy.mockImplementationOnce(() => {
      throw new RangeError("transient boom");
    });
    const failed = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(failed.state).toBe("unavailable");
    const retried = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(retried.state).toBe("ready");
    expect(replaySpy).toHaveBeenCalledTimes(2);
  });
});

describe("时钟边界：不写 DB 推进 asOf 也必须重算", () => {
  it("跨周界：同一 DB 无写入，asOf 进入下一段后结果变化", () => {
    seedWeekMatches("2026-10-05");
    const week1 = ready(loadGlickoSnapshot(db, AS_OF_WEEK_1));
    expect(week1.currentSegment.id).toBe("2026-10-05:2026-10-05");
    // 未过周界：本周段尚未 Final（lastFinal 只含段初新人初值基准）
    expect(week1.events.every((event) => event.kind !== "weekly_final")).toBe(true);
    const initial = { r: 1000, rd: 180, volatility: 0.06 };
    expect(week1.lastFinal).toEqual({
      1: initial,
      2: initial,
      3: initial,
      4: initial,
    });

    const week2 = ready(loadGlickoSnapshot(db, AS_OF_WEEK_2));
    expect(week2.currentSegment.id).toBe("2026-10-12:2026-10-12");
    expect(week2.events.some((event) => event.kind === "weekly_final")).toBe(true);
    // 上周已结算，Final 值不再是初值，本周 current 与周中 Estimated 不同
    expect(week2.lastFinal).not.toEqual(week1.lastFinal);
    expect(week2.current).not.toEqual(week1.current);
  });

  it("跨季界：出现 season_reset，当前值按季软重置规则变化", () => {
    seedWeekMatches("2026-12-28");
    const before = ready(loadGlickoSnapshot(db, AS_OF_SEASON_1));
    expect(before.events.some((event) => event.kind === "season_reset")).toBe(false);

    const after = ready(loadGlickoSnapshot(db, AS_OF_SEASON_2));
    expect(after.currentSegment.id).toBe("2026-12-28:2027-01-01");
    const reset = after.events.find((event) => event.kind === "season_reset");
    expect(reset).toBeDefined();
    if (reset && reset.kind === "season_reset") {
      expect(reset.seasonId).toBe("2027-01-01");
    }
    expect(after.current).not.toEqual(before.current);
  });
});

describe("数据修正使旧缓存失效", () => {
  it("改名后指纹变化并重算，但重放数值与改名前一致（纯函数）", () => {
    seedWeekMatches("2026-10-05");
    const first = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(first.state).toBe("ready");

    renamePlayer(players[0], "A-renamed", db);
    const second = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(replaySpy).toHaveBeenCalledTimes(2);
    expect(second.state).toBe("ready");
    if (first.state === "ready" && second.state === "ready") {
      expect(second.inputHash).not.toBe(first.inputHash);
      expect(second.replay.current).toEqual(first.replay.current);
    }
  });

  it("改分后旧缓存不复用，结果变化", () => {
    seedWeekMatches("2026-10-05");
    const first = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    db.prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    const second = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(replaySpy).toHaveBeenCalledTimes(2);
    expect(second.state).toBe("ready");
    if (first.state === "ready" && second.state === "ready") {
      expect(second.inputHash).not.toBe(first.inputHash);
      expect(second.replay.current).not.toEqual(first.replay.current);
    }
  });

  it("撤回（删比赛）后重算", () => {
    const ids = seedWeekMatches("2026-10-05");
    const first = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    deleteMatch(ids[1], db);
    const second = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(replaySpy).toHaveBeenCalledTimes(2);
    expect(second.state).toBe("ready");
    if (first.state === "ready" && second.state === "ready") {
      expect(second.inputHash).not.toBe(first.inputHash);
      expect(Object.keys(second.replay.matchEstimates)).toHaveLength(1);
    }
  });

  it("合并球员后重算：目录与比赛事实都进指纹", () => {
    seedWeekMatches("2026-10-05");
    const p5 = addPlayer("E", db);
    addMatch(
      { pa1: p5, pa2: players[1], pb1: players[2], pb2: players[3], scoreA: 21, scoreB: 17, playedAt: "2026-10-06" },
      db
    );
    const first = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    mergePlayers(p5, players[0], db);
    const second = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(replaySpy).toHaveBeenCalledTimes(2);
    expect(second.state).toBe("ready");
    if (first.state === "ready" && second.state === "ready") {
      expect(second.inputHash).not.toBe(first.inputHash);
      // 合并后 E 的出场记到 A 名下，历史重放结果发生变化
      expect(second.replay.current).not.toEqual(first.replay.current);
    }
  });
});

describe("失败快照判别表", () => {
  it("无历史缓存时计算失败 → unavailable", () => {
    seedWeekMatches("2026-10-05");
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const result = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(result).toEqual({
      state: "unavailable",
      model: "glicko2",
      reason: "replay failed: numeric failure",
    });
  });

  it("同输入曾成功、本次重算失败（asOf 跨段）→ 直接复用旧成功结果 ready", () => {
    seedWeekMatches("2026-10-05");
    const first = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(first.state).toBe("ready");
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const result = loadGlickoSnapshot(db, AS_OF_WEEK_2);
    // inputHash 与缓存一致（同输入曾成功），允许直接复用
    expect(result.state).toBe("ready");
    if (result.state === "ready" && first.state === "ready") {
      expect(result.replay).toEqual(first.replay);
      expect(result.replay.asOf).toBe(AS_OF_WEEK_1); // 如实反映旧时点
    }
  });

  it("输入已变（旧输入缓存）+ 计算失败 → stale，lastGood 带旧 asOf，不冒充当前", () => {
    seedWeekMatches("2026-10-05");
    const first = ready(loadGlickoSnapshot(db, AS_OF_WEEK_1));
    db.prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const result = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(result.state).toBe("stale");
    if (result.state === "stale") {
      expect(result.reason).toContain("numeric failure");
      expect(result.lastGood.asOf).toBe(AS_OF_WEEK_1);
      expect(result.lastGood).toEqual(first);
    }
  });

  it("缓存序列化损坏 + 计算失败 → unavailable；计算成功则覆写损坏缓存", () => {
    seedWeekMatches("2026-10-05");
    ready(loadGlickoSnapshot(db, AS_OF_WEEK_1));
    db.prepare(`UPDATE meta SET value = 'garbage{{' WHERE key = ?`).run(
      ratingLastGoodKey(configVersionOf(db))
    );
    // 改变输入以绕过进程内 memo，强制走「读缓存 → 计算」路径。
    db.prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();

    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const failed = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(failed.state).toBe("unavailable");

    replaySpy.mockRestore();
    replaySpy = vi.spyOn(replayModule, "replayRatings");
    const recovered = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(recovered.state).toBe("ready");
    const row = db
      .prepare(`SELECT value FROM meta WHERE key = ?`)
      .get(ratingLastGoodKey(configVersionOf(db))) as { value: string };
    const stored = JSON.parse(row.value) as { inputHash: string };
    expect(() => JSON.parse(row.value)).not.toThrow();
    if (recovered.state === "ready") {
      expect(stored.inputHash).toBe(recovered.inputHash);
    }
  });

  it("重启进程（closeDb+重开连接）后读取旧成功记录仍校验指纹", () => {
    seedWeekMatches("2026-10-05");
    const first = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(first.state).toBe("ready");
    db.close();

    const reopened = createDb(join(dir, "test.db"));
    const callsBefore = replaySpy.mock.calls.length;
    // 同输入同边界：直接复用磁盘快照，不重新计算
    const fromDisk = loadGlickoSnapshot(reopened, AS_OF_WEEK_1);
    expect(fromDisk.state).toBe("ready");
    expect(replaySpy.mock.calls.length).toBe(callsBefore);
    if (fromDisk.state === "ready" && first.state === "ready") {
      expect(fromDisk.replay).toEqual(first.replay);
    }

    // 同输入曾成功而重算失败：复用旧成功结果 ready
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const sameInput = loadGlickoSnapshot(reopened, AS_OF_WEEK_2);
    expect(sameInput.state).toBe("ready");

    // 输入已变 + 失败：磁盘上旧输入缓存只给 stale，不冒充当前
    reopened.prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    const changed = loadGlickoSnapshot(reopened, AS_OF_WEEK_1);
    expect(changed.state).toBe("stale");
    reopened.close();
  });

  it("配置版本不符的 last-good 键不被当作当前", () => {
    seedWeekMatches("2026-10-05");
    const bogusReplay = ready(loadGlickoSnapshot(db, AS_OF_WEEK_1));
    db.prepare(`INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)`).run(
      "ratings.last-good.v1:glicko2-doubles-v1|p0|2020-01-01|Asia%2FShanghai|1000|180|60|250|0.06|0.3|900|1100|0.75|90",
      JSON.stringify({ inputHash: "bogus", asOf: "2020-01-01T00:00:00+08:00", replay: bogusReplay })
    );

    const real = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(real.state).toBe("ready");
    if (real.state === "ready") {
      expect(real.inputHash).not.toBe("bogus");
      expect(real.replay.currentSegment.id).toBe("2026-10-05:2026-10-05");
    }

    // 输入已变 + 计算失败：stale 只能来自同版本真实缓存，不得回退到 bogus 键
    db.prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    replaySpy.mockImplementation(() => {
      throw new RangeError("numeric failure");
    });
    const failed = loadGlickoSnapshot(db, AS_OF_WEEK_1);
    expect(failed.state).toBe("stale");
    if (failed.state === "stale") {
      expect(failed.lastGood.asOf).toBe(AS_OF_WEEK_1);
      expect(failed.lastGood.currentSegment.id).toBe("2026-10-05:2026-10-05");
    }
  });

  it("两个连接先后写缓存：后写覆盖先写且指纹正确", () => {
    seedWeekMatches("2026-10-05");
    const conn1 = db;
    const first = loadGlickoSnapshot(conn1, AS_OF_WEEK_1);
    expect(first.state).toBe("ready");

    // 第二个连接改分并读取，重算后写入新缓存
    const conn2 = createDb(join(dir, "test.db"));
    conn2.prepare(`UPDATE matches SET score_a = 5 WHERE id = 1`).run();
    const second = loadGlickoSnapshot(conn2, AS_OF_WEEK_1);
    expect(second.state).toBe("ready");
    if (first.state === "ready" && second.state === "ready") {
      expect(second.inputHash).not.toBe(first.inputHash);
    }

    const row = conn2
      .prepare(`SELECT value FROM meta WHERE key = ?`)
      .get(ratingLastGoodKey(configVersionOf(conn2))) as { value: string };
    const stored = JSON.parse(row.value) as { inputHash: string; replay: RatingReplay };
    if (second.state === "ready") {
      expect(stored.inputHash).toBe(second.inputHash);
      expect(stored.replay).toEqual(second.replay);
    }

    // 连接 1 再读：输入指纹已变，得到与连接 2 一致的结果
    const reread = loadGlickoSnapshot(conn1, AS_OF_WEEK_1);
    expect(reread.state).toBe("ready");
    if (reread.state === "ready" && second.state === "ready") {
      expect(reread.inputHash).toBe(second.inputHash);
      expect(reread.replay).toEqual(second.replay);
    }
    conn2.close();
  });
});
