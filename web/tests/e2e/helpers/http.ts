/** 极小 HTTP JSON 客户端，所有旅程断言都打在真实服务器上。 */

export interface HttpResult<T = unknown> {
  status: number;
  body: T;
}

export class ApiClient {
  constructor(private readonly baseUrl: string) {}

  async get<T = unknown>(path: string, headers?: Record<string, string>): Promise<HttpResult<T>> {
    const res = await fetch(`${this.baseUrl}${path}`, { headers });
    return { status: res.status, body: (await res.json()) as T };
  }

  async post<T = unknown>(
    path: string,
    payload?: unknown,
    headers?: Record<string, string>
  ): Promise<HttpResult<T>> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
    return { status: res.status, body: (await res.json()) as T };
  }

  async patch<T = unknown>(
    path: string,
    payload?: unknown,
    headers?: Record<string, string>
  ): Promise<HttpResult<T>> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", ...headers },
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
    return { status: res.status, body: (await res.json()) as T };
  }

  async delete<T = unknown>(
    path: string,
    headers?: Record<string, string>
  ): Promise<HttpResult<T>> {
    const res = await fetch(`${this.baseUrl}${path}`, { method: "DELETE", headers });
    return { status: res.status, body: (await res.json()) as T };
  }
}

/** 简化接口形状（与 src/app/schedule/types.ts 对齐）。 */
export interface FixtureShape {
  id: number;
  displayNo: number;
  pa1: string;
  pa2: string;
  pb1: string;
  pb2: string;
  status: "pending" | "played_unscored" | "recorded" | "superseded";
  revision: number;
  playedAt: string | null;
  matchId: number | null;
  scoreA: number | null;
  scoreB: number | null;
}

export interface ProjectDetailShape {
  id: number;
  name: string;
  revision: number;
  members: Array<{
    playerId: string;
    fairnessBase: number;
    playedAtJoin: number;
    actualPlayed: number;
    pendingCount: number;
  }>;
  fixtures: FixtureShape[];
  superseded: FixtureShape[];
}
