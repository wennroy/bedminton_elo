import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createDb } from "./db";
import {
  RATING_CONFIG_META_KEY,
  defaultFirstSeasonStart,
  describeRatingConfigChanges,
  initializeRatingConfig,
  planInitializeRatingConfig,
  readRatingConfig,
  setActiveModel,
} from "./rating-config";

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "rating-config-test-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function dbPath(name = "test.db"): string {
  return join(dir, name);
}

describe("defaultFirstSeasonStart", () => {
  it("取初始化当天上海日期之后的首个季度起点", () => {
    expect(defaultFirstSeasonStart("2026-09-22T13:00:00+08:00")).toBe("2026-10-01");
    expect(defaultFirstSeasonStart("2026-10-05T13:00:00+08:00")).toBe("2027-01-01");
  });

  it("初始化当天本身是季度起点时取下一个季度（严格之后）", () => {
    expect(defaultFirstSeasonStart("2026-10-01T10:00:00+08:00")).toBe("2027-01-01");
  });
});

describe("initializeRatingConfig", () => {
  it("未初始化时写入默认 legacy 配置并固化首赛季起点", () => {
    const db = createDb(dbPath());
    const result = initializeRatingConfig(
      { todayInstant: "2026-09-22T13:00:00+08:00" },
      db
    );
    expect(result.created).toBe(true);
    expect(result.record.activeModel).toBe("legacy");
    expect(result.record.config.firstSeasonStart).toBe("2026-10-01");
    expect(result.record.config.paramsVersion).toBe("p1");
    expect(readRatingConfig(db)).toEqual(result.record);
    db.close();
  });

  it("支持显式 firstSeasonStart，非法季度首日被拒绝", () => {
    const db = createDb(dbPath());
    const ok = initializeRatingConfig({ firstSeasonStart: "2026-07-01" }, db);
    expect(ok.record.config.firstSeasonStart).toBe("2026-07-01");
    expect(() => initializeRatingConfig({ firstSeasonStart: "2026-10-02" }, db)).toThrow(
      /quarter start/
    );
    db.close();
  });

  it("重复初始化不覆写旧配置", () => {
    const path = dbPath();
    const db = createDb(path);
    const first = initializeRatingConfig(
      { todayInstant: "2026-09-22T13:00:00+08:00" },
      db
    );
    db.close();

    const reopened = createDb(path);
    const second = initializeRatingConfig(
      { todayInstant: "2026-11-15T13:00:00+08:00" },
      reopened
    );
    expect(second.created).toBe(false);
    expect(second.replaced).toBe(false);
    expect(second.record).toEqual(first.record);
    // 月份变化不得移动首赛季起点
    expect(second.record.config.firstSeasonStart).toBe("2026-10-01");
    reopened.close();
  });

  it("同 paramsVersion 参数不同则拒绝且不覆写", () => {
    const db = createDb(dbPath());
    initializeRatingConfig({}, db);
    expect(() => initializeRatingConfig({ tau: 0.4 }, db)).toThrow(
      /already initialized with different parameters/
    );
    expect(readRatingConfig(db)!.config.tau).toBe(0.3);
    db.close();
  });

  it("新 paramsVersion 整体替换参数但保留 activeModel 与首赛季起点", () => {
    const db = createDb(dbPath());
    initializeRatingConfig(
      { todayInstant: "2026-09-22T13:00:00+08:00" },
      db
    );
    setActiveModel("glicko2", db);
    const result = initializeRatingConfig({ paramsVersion: "p2", tau: 0.4 }, db);
    expect(result.created).toBe(true);
    expect(result.replaced).toBe(true);
    expect(result.record.config.paramsVersion).toBe("p2");
    expect(result.record.config.tau).toBe(0.4);
    expect(result.record.activeModel).toBe("glicko2");
    expect(result.record.config.firstSeasonStart).toBe("2026-10-01");
    db.close();
  });

  it("新 paramsVersion 显式指定 firstSeasonStart 时采用新值", () => {
    const db = createDb(dbPath());
    initializeRatingConfig({ todayInstant: "2026-09-22T13:00:00+08:00" }, db);
    const result = initializeRatingConfig(
      { paramsVersion: "p2", firstSeasonStart: "2027-01-01" },
      db
    );
    expect(result.record.config.firstSeasonStart).toBe("2027-01-01");
    db.close();
  });
});

describe("readRatingConfig", () => {
  it("无配置时返回 null（不隐式写入）", () => {
    const db = createDb(dbPath());
    expect(readRatingConfig(db)).toBeNull();
    const row = db
      .prepare(`SELECT COUNT(*) AS c FROM meta WHERE key = ?`)
      .get(RATING_CONFIG_META_KEY) as { c: number };
    expect(row.c).toBe(0);
    db.close();
  });

  it("已存在但损坏的记录抛错而不是当成未初始化", () => {
    const db = createDb(dbPath());
    db.prepare(`INSERT INTO meta (key, value) VALUES (?, ?)`).run(
      RATING_CONFIG_META_KEY,
      "not-json{{"
    );
    expect(() => readRatingConfig(db)).toThrow(/not valid JSON/);
    db.close();
  });
});

describe("setActiveModel", () => {
  it("切换模型并持久化；相同模型为幂等 no-op 不写入", () => {
    const db = createDb(dbPath());
    initializeRatingConfig({}, db);
    const switched = setActiveModel("glicko2", db);
    expect(switched.activeModel).toBe("glicko2");
    expect(readRatingConfig(db)!.activeModel).toBe("glicko2");

    const before = (
      db.prepare(`SELECT value FROM meta WHERE key = ?`).get(RATING_CONFIG_META_KEY) as {
        value: string;
      }
    ).value;
    const same = setActiveModel("glicko2", db);
    expect(same.activeModel).toBe("glicko2");
    const after = (
      db.prepare(`SELECT value FROM meta WHERE key = ?`).get(RATING_CONFIG_META_KEY) as {
        value: string;
      }
    ).value;
    expect(after).toBe(before);
    db.close();
  });

  it("未初始化时抛错；非法模型名抛错", () => {
    const db = createDb(dbPath());
    expect(() => setActiveModel("glicko2", db)).toThrow(/not initialized/);
    initializeRatingConfig({}, db);
    expect(() => setActiveModel("trueskill" as never, db)).toThrow(/glicko2 or legacy/);
    db.close();
  });
});

describe("planInitializeRatingConfig / describeRatingConfigChanges", () => {
  it("dry-run 推演与实际 init 一致但不写入", () => {
    const db = createDb(dbPath());
    const planned = planInitializeRatingConfig(
      { todayInstant: "2026-09-22T13:00:00+08:00", tau: 0.4, paramsVersion: "p2" },
      db
    );
    expect(planned.created).toBe(true);
    expect(readRatingConfig(db)).toBeNull();
    const executed = initializeRatingConfig(
      { todayInstant: "2026-09-22T13:00:00+08:00", tau: 0.4, paramsVersion: "p2" },
      db
    );
    expect(executed.record).toEqual(planned.record);
    db.close();
  });

  it("输出具体差异行，无差异时为空", () => {
    const db = createDb(dbPath());
    const first = initializeRatingConfig({}, db);
    expect(describeRatingConfigChanges(null, first.record).length).toBeGreaterThan(0);
    expect(describeRatingConfigChanges(first.record, first.record)).toEqual([]);
    const next = planInitializeRatingConfig({ paramsVersion: "p2", tau: 0.4 }, db);
    expect(describeRatingConfigChanges(next.previous, next.record)).toEqual([
      "paramsVersion: p1 -> p2",
      "tau: 0.3 -> 0.4",
    ]);
    db.close();
  });
});

describe("临时 DB 独立性", () => {
  it("两个临时库互不影响", () => {
    const dbA = createDb(dbPath("a.db"));
    const dbB = createDb(dbPath("b.db"));
    initializeRatingConfig({}, dbA);
    expect(readRatingConfig(dbA)).not.toBeNull();
    expect(readRatingConfig(dbB)).toBeNull();
    dbA.close();
    dbB.close();
  });
});
