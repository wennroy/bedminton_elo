import { spawn, type ChildProcess } from "child_process";
import { join } from "path";

/**
 * 用 production 模式（next build && next start）起真实服务器：
 * 比 next dev 稳定——没有按需编译的秒级首请求、没有 HMR websocket 噪音、
 * React 不跑 strict-mode 双 effect，轮询/刷新断言更确定。
 *
 * 数据库通过 DATABASE_URL 指向测试临时文件；ADMIN_PASSWORD 固定，
 * admin API 走 x-admin-key 头。
 */

export const ADMIN_PASSWORD = "e2e-admin-password";

const WEB_ROOT = join(__dirname, "..", "..", "..");

export interface RunningServer {
  baseUrl: string;
  stop(): Promise<void>;
}

async function waitReady(baseUrl: string, proc: ChildProcess, timeoutMs = 60000): Promise<void> {
  const started = Date.now();
  let lastError: unknown = null;
  while (Date.now() - started < timeoutMs) {
    if (proc.exitCode !== null) {
      throw new Error(`next start exited early with code ${proc.exitCode}`);
    }
    try {
      const res = await fetch(`${baseUrl}/api/players`);
      if (res.ok) return;
    } catch (error) {
      lastError = error;
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`server not ready at ${baseUrl}: ${String(lastError)}`);
}

export async function startServer(dbPath: string, port: number): Promise<RunningServer> {
  const baseUrl = `http://127.0.0.1:${port}`;
  const proc = spawn(
    process.execPath,
    [join(WEB_ROOT, "node_modules", "next", "dist", "bin", "next"), "start", "-p", String(port)],
    {
      cwd: WEB_ROOT,
      env: {
        ...process.env,
        DATABASE_URL: dbPath,
        ADMIN_PASSWORD,
        HOSTNAME: "127.0.0.1",
        PORT: String(port),
      },
      stdio: ["ignore", "pipe", "pipe"],
    }
  );
  // 静音服务器日志噪音；诊断时再打开。
  proc.stderr?.on("data", () => {});
  proc.stdout?.on("data", () => {});
  await waitReady(baseUrl, proc);
  return {
    baseUrl,
    stop: () =>
      new Promise((resolve) => {
        if (proc.exitCode !== null) {
          resolve();
          return;
        }
        proc.once("exit", () => resolve());
        proc.kill("SIGTERM");
        setTimeout(() => {
          if (proc.exitCode === null) proc.kill("SIGKILL");
        }, 5000).unref();
      }),
  };
}
