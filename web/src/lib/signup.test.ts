import { describe, it, expect } from "vitest";
import type Database from "better-sqlite3";
import { tmpdir } from "os";
import { join } from "path";
import { createDb } from "./db";
import { addPlayer } from "./repo";
import {
  getActiveSessionDate,
  formatSessionDate,
  listSignups,
  upsertSignup,
  removeSignup,
  signupSummary,
} from "./signup";

function mkDb(): Database.Database {
  const path = join(tmpdir(), `signup-test-${Date.now()}-${Math.random()}.db`);
  return createDb(path);
}

// 2026-09-02 是周三；以下时间均为 UTC，注释里给出对应上海时间
describe("getActiveSessionDate", () => {
  it("周三 20:00（上海）前返回当天", () => {
    // 上海 2026-09-02 19:59 = UTC 11:59
    expect(getActiveSessionDate(new Date("2026-09-02T11:59:00Z"))).toBe(
      "2026-09-02"
    );
  });

  it("周三 20:00（上海）整点起返回下周三", () => {
    // 上海 2026-09-02 20:00 = UTC 12:00
    expect(getActiveSessionDate(new Date("2026-09-02T12:00:00Z"))).toBe(
      "2026-09-09"
    );
  });

  it("周二返回第二天（当周周三）", () => {
    // 上海 2026-09-01 23:30 = UTC 15:30
    expect(getActiveSessionDate(new Date("2026-09-01T15:30:00Z"))).toBe(
      "2026-09-02"
    );
  });

  it("周四返回下周三", () => {
    // 上海 2026-09-03 10:00 = UTC 02:00
    expect(getActiveSessionDate(new Date("2026-09-03T02:00:00Z"))).toBe(
      "2026-09-09"
    );
  });

  it("周日返回 3 天后的周三", () => {
    // 上海 2026-09-06 08:00 = UTC 00:00
    expect(getActiveSessionDate(new Date("2026-09-06T00:00:00Z"))).toBe(
      "2026-09-09"
    );
  });
});

describe("formatSessionDate", () => {
  it("格式化为中文日期", () => {
    expect(formatSessionDate("2026-09-09")).toBe("9月9日（周三）");
  });
});

describe("signups 存储", () => {
  it("upsert → list → summary → remove 全流程", () => {
    const db = mkDb();
    const a = addPlayer("alice", db);
    const b = addPlayer("bob", db);
    const session = "2026-09-09";

    upsertSignup(session, a, 1, db);
    upsertSignup(session, b, 2, db);

    const rows = listSignups(session, db);
    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({ playerId: a, name: "alice", partySize: 1 });
    expect(rows[1]).toMatchObject({ playerId: b, name: "bob", partySize: 2 });

    expect(signupSummary(session, db)).toEqual({ count: 2, totalPeople: 3 });

    removeSignup(session, a, db);
    expect(listSignups(session, db)).toHaveLength(1);
    expect(signupSummary(session, db)).toEqual({ count: 1, totalPeople: 2 });
  });

  it("重复报名 = 更新人数（唯一约束）", () => {
    const db = mkDb();
    const a = addPlayer("alice", db);
    const session = "2026-09-09";

    upsertSignup(session, a, 1, db);
    upsertSignup(session, a, 2, db);

    const rows = listSignups(session, db);
    expect(rows).toHaveLength(1);
    expect(rows[0].partySize).toBe(2);
  });

  it("不同场次互不影响", () => {
    const db = mkDb();
    const a = addPlayer("alice", db);

    upsertSignup("2026-09-02", a, 2, db);
    upsertSignup("2026-09-09", a, 1, db);

    expect(signupSummary("2026-09-02", db)).toEqual({
      count: 1,
      totalPeople: 2,
    });
    expect(signupSummary("2026-09-09", db)).toEqual({
      count: 1,
      totalPeople: 1,
    });
  });

  it("空场次 summary 为 0", () => {
    const db = mkDb();
    expect(signupSummary("2026-09-09", db)).toEqual({
      count: 0,
      totalPeople: 0,
    });
  });
});
