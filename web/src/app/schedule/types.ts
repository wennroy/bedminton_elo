/**
 * 排阵项目的客户端 DTO 形状：与 /api/schedule/projects* 响应一致。
 * 客户端组件不 import lib 层（避免把 better-sqlite3 服务端代码带进 bundle），
 * 类型在此按 API 合同镜像维护（与 lib/schedule-projects.ts 结构对齐）。
 */

export type FixtureStatus = "pending" | "played_unscored" | "recorded" | "superseded";

export interface Fixture {
  id: number;
  projectId: number;
  generationId: number;
  displayNo: number;
  pa1: string;
  pa2: string;
  pb1: string;
  pb2: string;
  /** 生成时随场次保存的预测胜率（原快照，不冒充最新预测）。 */
  generatedWinRate: number | null;
  status: FixtureStatus;
  revision: number;
  playedAt: string | null;
  markedAt: string | null;
  markedBy: string | null;
  matchId: number | null;
  /** recorded 时联 matches 读取；其余状态为 null。 */
  scoreA: number | null;
  scoreB: number | null;
}

export interface ProjectMember {
  playerId: string;
  fairnessBase: number;
  playedAtJoin: number;
  joinedAt: string | null;
  leftAt: string | null;
  /** 本项目内已打（played_unscored + recorded）且该人出场的场次数。 */
  actualPlayed: number;
  /** 本项目内 pending 且该人出场的场次数。 */
  pendingCount: number;
}

export interface ProjectSummary {
  id: number;
  name: string;
  model: string;
  revision: number;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  activePlayers: number;
  pendingCount: number;
  playedUnscoredCount: number;
  recordedCount: number;
}

export interface ProjectDetail {
  id: number;
  name: string;
  model: string;
  revision: number;
  nextFixtureNo: number;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  /** 当前名单（active = 1），每人带真实已打/待打统计。 */
  members: ProjectMember[];
  /** 有效场次：pending + played_unscored + recorded，按 displayNo 升序。 */
  fixtures: Fixture[];
  /** 已被取代的场次（旧链接只读展示）。 */
  superseded: Fixture[];
  latestGeneration: {
    id: number;
    requestId: string;
    sourceRevision: number;
    /** 引擎输入快照（JSON）；客户端只读取 matches / avoidPlayedRematches 做回显。 */
    input: unknown;
    metrics: unknown;
    createdAt: string;
  } | null;
}

/** latestGeneration.input 里客户端关心的续排选项回显字段。 */
export interface LatestGenerationOptions {
  matches?: unknown;
  avoidPlayedRematches?: unknown;
}

/** 从 latestGeneration.input 解析续排表单回显值（缺省用安全默认）。 */
export function readGenerationOptions(input: unknown): {
  matches: number | null;
  avoidPlayedRematches: boolean;
} {
  const record =
    input && typeof input === "object" ? (input as LatestGenerationOptions) : {};
  const matches =
    typeof record.matches === "number" && Number.isInteger(record.matches)
      ? record.matches
      : null;
  const avoidPlayedRematches =
    typeof record.avoidPlayedRematches === "boolean"
      ? record.avoidPlayedRematches
      : true;
  return { matches, avoidPlayedRematches };
}
