import { spawn } from "child_process";
import { mkdirSync, rmSync } from "fs";
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

/**
 * 真实并发：每个 worker 是独立 node OS 进程，各自打一次 HTTP。
 * 文件屏障保证所有进程就绪后同时开火，而不是用同一连接顺序冒充并发。
 */

export interface WorkerOutcome {
  status: number;
  body: unknown;
  startedAt: number;
  finishedAt: number;
}

const WORKER = join(__dirname, "..", "workers", "http-once.mjs");

export async function runConcurrentPosts(
  baseUrl: string,
  requests: Array<{ path: string; body: unknown }>
): Promise<WorkerOutcome[]> {
  const barrierDir = mkdtempSync(join(tmpdir(), "bedminton-e2e-barrier-"));
  mkdirSync(barrierDir, { recursive: true });
  try {
    const children = requests.map((req, i) => {
      const child = spawn(process.execPath, [WORKER], {
        env: {
          ...process.env,
          E2E_URL: baseUrl,
          E2E_PATH: req.path,
          E2E_BODY: JSON.stringify(req.body ?? {}),
          E2E_BARRIER_DIR: barrierDir,
          E2E_BARRIER_NAME: `w${i}`,
          E2E_BARRIER_COUNT: String(requests.length),
        },
        stdio: ["ignore", "pipe", "pipe"],
      });
      return { child, stdout: "" as string, stderr: "" as string };
    });

    const outcomes = await Promise.all(
      children.map(
        ({ child, stdout, stderr }) =>
          new Promise<WorkerOutcome>((resolve, reject) => {
            child.stdout.on("data", (d: Buffer) => {
              stdout += d.toString();
            });
            child.stderr.on("data", (d: Buffer) => {
              stderr += d.toString();
            });
            child.on("error", reject);
            child.on("exit", (code) => {
              const line = stdout.split("\n").find((l) => l.startsWith("RESULT: "));
              if (!line) {
                reject(new Error(`worker exited ${code} without RESULT: ${stderr}`));
                return;
              }
              resolve(JSON.parse(line.slice("RESULT: ".length)) as WorkerOutcome);
            });
          })
      )
    );
    return outcomes;
  } finally {
    try {
      rmSync(barrierDir, { recursive: true, force: true });
    } catch {
      // 屏障目录清理失败不影响结果。
    }
  }
}
