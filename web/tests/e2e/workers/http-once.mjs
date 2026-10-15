// E2E 并发 HTTP worker：独立 OS 进程，参与文件屏障后并发开火一次请求。
// 用法（env）：
//   E2E_URL            服务器根地址
//   E2E_PATH           请求路径（如 /api/schedule/projects/1/matches/2/score）
//   E2E_BODY           JSON 字符串（POST 时）
//   E2E_BARRIER_DIR    屏障目录；E2E_BARRIER_NAME 本进程名；E2E_BARRIER_COUNT 总人数
// 结果以一行 `RESULT: <json>` 打印到 stdout。

const {
  E2E_URL,
  E2E_PATH,
  E2E_BODY,
  E2E_BARRIER_DIR,
  E2E_BARRIER_NAME,
  E2E_BARRIER_COUNT,
} = process.env;

if (!E2E_URL || !E2E_PATH || !E2E_BARRIER_DIR || !E2E_BARRIER_NAME || !E2E_BARRIER_COUNT) {
  console.error("missing env");
  process.exit(2);
}

import fs from "node:fs";
import path from "node:path";

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function barrier() {
  fs.writeFileSync(path.join(E2E_BARRIER_DIR, `${E2E_BARRIER_NAME}.ready`), "1");
  const total = Number(E2E_BARRIER_COUNT);
  const started = Date.now();
  while (Date.now() - started < 30000) {
    const files = fs.readdirSync(E2E_BARRIER_DIR).filter((f) => f.endsWith(".ready"));
    if (files.length >= total) return;
    await sleep(20);
  }
  throw new Error("barrier timeout");
}

(async () => {
  await barrier();
  // 屏障齐后再对齐一次节奏，尽最大可能真同时。
  await sleep(50);
  const startedAt = Date.now();
  let result;
  try {
    const res = await fetch(`${E2E_URL}${E2E_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: E2E_BODY ?? "{}",
    });
    let body = null;
    try {
      body = await res.json();
    } catch {
      body = null;
    }
    result = { status: res.status, body };
  } catch (error) {
    result = { status: 0, error: String(error) };
  }
  console.log(`RESULT: ${JSON.stringify({ ...result, startedAt, finishedAt: Date.now() })}`);
  process.exit(0);
})();
