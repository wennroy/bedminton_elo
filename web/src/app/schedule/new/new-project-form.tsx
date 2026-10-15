"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { ChevronDown, ChevronUp, Sparkles, UserPlus } from "lucide-react";
import { PlayerAvatar } from "@/components/player-avatar";
import { Button } from "@/components/ui/button";
import { getMyPlayerId } from "@/lib/identity";
import { loadSchedule } from "@/lib/schedule-storage";
import { shanghaiLocalDateFromInstant } from "@/lib/ratings/calendar";
import { useMyPlayerId } from "../use-my-player-id";

interface Player {
  id: number;
  name: string;
}

const MAX_NAME_LENGTH = 60;
const MAX_SEED = 4294967295;

export function NewProjectForm({ players }: { players: Player[] }) {
  const router = useRouter();
  const myId = useMyPlayerId();

  const [directory, setDirectory] = React.useState<Player[]>(players);
  const [selected, setSelected] = React.useState<Set<number>>(new Set());
  const [matches, setMatches] = React.useState(4);
  const [avoidRematch, setAvoidRematch] = React.useState(true);
  const [name, setName] = React.useState("");
  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  /** 留空 = 不传 seed，由服务器取新随机值（普通操作）。 */
  const [seed, setSeed] = React.useState("");
  const [adding, setAdding] = React.useState(false);
  const [addName, setAddName] = React.useState("");
  const [addError, setAddError] = React.useState<string | null>(null);
  const [addBusy, setAddBusy] = React.useState(false);
  const [legacyHint, setLegacyHint] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  /** 409 rating_unavailable：评分服务不可用时单独提示，不伪造胜率。 */
  const [ratingDown, setRatingDown] = React.useState<string | null>(null);
  /**
   * 幂等重试：同一表单版本共用一个 requestId，网络失败重试安全；
   * 任何字段变化后作废重发（避免同 requestId 不同 payload 的幂等冲突）。
   */
  const [requestId, setRequestId] = React.useState<string | null>(null);

  // 兼容旧 localStorage 配对草稿：仅作表单提示与名单勾选参考，不上传。
  React.useEffect(() => {
    const stored = loadSchedule(getMyPlayerId());
    if (!stored) return;
    const valid = new Set(players.map((p) => p.id));
    const ids = stored.playerIds.filter((id) => valid.has(id));
    if (ids.length >= 4) {
      setSelected(new Set(ids));
      if (stored.matches >= 1 && stored.matches <= 99) {
        setMatches(stored.matches);
      }
      setLegacyHint(
        `检测到本机旧版配对草稿（${stored.playerIds.length} 人 · ${stored.matches} 场），已按当时的名单勾选。旧数据不会上传，公共项目以服务器为准。`
      );
    } else {
      setLegacyHint(
        `检测到本机旧版配对草稿（${stored.playerIds.length} 人 · ${stored.matches} 场），仅供参考，不会上传；公共项目以服务器为准。`
      );
    }
    // 仅在挂载时读取一次；players 为服务端注入的初始名单。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  React.useEffect(() => {
    setRequestId(null);
  }, [selected, matches, avoidRematch, name, seed]);

  const defaultName = `${shanghaiLocalDateFromInstant(
    new Date().toISOString()
  )} 配对`;

  function toggle(id: number) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleAddPlayer(event: React.FormEvent) {
    event.preventDefault();
    const trimmed = addName.trim();
    if (!trimmed || addBusy) return;
    setAddBusy(true);
    setAddError(null);
    try {
      const res = await fetch("/api/players", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: trimmed }),
      });
      const data = await res.json();
      if (!res.ok) {
        setAddError(data.error || "添加失败");
        return;
      }
      const id = Number(data.id);
      if (Number.isInteger(id)) {
        // 保持操作者身份不变，仅把新球员加入目录并勾选，留在表单继续编辑。
        setDirectory((prev) =>
          prev.some((p) => p.id === id)
            ? prev
            : [...prev, { id, name: data.name ?? trimmed }].sort(
                (a, b) => a.id - b.id
              )
        );
        setSelected((prev) => new Set(prev).add(id));
      }
      setAdding(false);
      setAddName("");
    } catch {
      setAddError("网络错误，请重试");
    } finally {
      setAddBusy(false);
    }
  }

  const trimmedName = name.trim();
  const canSubmit = selected.size >= 4 && myId !== null && !loading;

  async function handleSubmit() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    setRatingDown(null);

    if (trimmedName.length > MAX_NAME_LENGTH) {
      setError(`名称最多 ${MAX_NAME_LENGTH} 个字符`);
      setLoading(false);
      return;
    }
    const trimmedSeed = seed.trim();
    let seedValue: number | null = null;
    if (trimmedSeed !== "") {
      const n = Number(trimmedSeed);
      if (
        !Number.isSafeInteger(n) ||
        n < 0 ||
        n > MAX_SEED
      ) {
        setError(`随机种子必须是 0 到 ${MAX_SEED} 的整数`);
        setLoading(false);
        return;
      }
      seedValue = n;
    }

    const rid = requestId ?? crypto.randomUUID();
    setRequestId(rid);
    try {
      const response = await fetch("/api/schedule/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: trimmedName || undefined,
          playerIds: Array.from(selected).map(String),
          matches,
          avoidPlayedRematches: avoidRematch,
          actorId: String(myId),
          requestId: rid,
          ...(seedValue !== null ? { seed: seedValue } : {}),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        if (response.status === 409 && data?.code === "rating_unavailable") {
          setRatingDown(
            typeof data.reason === "string" ? data.reason : "评分服务暂不可用"
          );
        } else {
          setError(data?.error || "创建失败，请重试");
        }
        return;
      }
      setRequestId(null);
      router.push(`/schedule/${data.project.id}`);
    } catch {
      // 网络失败：未收到响应，不知是否已创建；同 requestId 重试安全。
      setError("网络错误，未确认是否创建成功；可直接重试，不会重复建项目");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {legacyHint && (
        <div className="rounded-xl border border-dashed border-border bg-muted/30 p-3 text-xs leading-relaxed text-muted-foreground">
          {legacyHint}
        </div>
      )}

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-semibold text-card-foreground">在场球员</h2>
          {selected.size > 0 && (
            <button
              type="button"
              onClick={() => setSelected(new Set())}
              className="text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              清空
            </button>
          )}
        </div>
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-5">
          {directory.map((player) => {
            const active = selected.has(player.id);
            return (
              <button
                key={player.id}
                type="button"
                onClick={() => toggle(player.id)}
                aria-pressed={active}
                className={`flex flex-col items-center gap-1 rounded-xl border p-2 transition-all ${
                  active
                    ? "border-primary bg-primary/5"
                    : "border-border bg-background hover:bg-muted"
                }`}
              >
                <PlayerAvatar name={player.name} size="sm" />
                <span className="max-w-full truncate text-xs font-medium text-foreground">
                  {player.name}
                </span>
              </button>
            );
          })}
          {adding ? (
            <form
              onSubmit={handleAddPlayer}
              className="col-span-full flex flex-col gap-2 rounded-xl border border-dashed border-border p-3"
            >
              <input
                autoFocus
                value={addName}
                onChange={(e) => {
                  setAddName(e.target.value);
                  setAddError(null);
                }}
                maxLength={20}
                placeholder="输入名字"
                aria-label="新球员名字"
                className="h-11 w-full rounded-xl border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
              {addError && <p className="text-xs text-destructive">{addError}</p>}
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  className="flex-1"
                  onClick={() => {
                    setAdding(false);
                    setAddName("");
                    setAddError(null);
                  }}
                >
                  返回
                </Button>
                <Button
                  type="submit"
                  className="flex-1"
                  disabled={addBusy || !addName.trim()}
                >
                  {addBusy ? "添加中…" : "确认添加"}
                </Button>
              </div>
            </form>
          ) : (
            <button
              type="button"
              onClick={() => setAdding(true)}
              className="flex flex-col items-center gap-1 rounded-xl border border-dashed border-border p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <span className="flex size-10 items-center justify-center rounded-full bg-muted">
                <UserPlus className="size-5" />
              </span>
              <span className="text-xs font-medium">新球员</span>
            </button>
          )}
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          已选 {selected.size} 人{selected.size < 4 && "（至少 4 人）"}
        </p>
      </div>

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <h2 className="mb-4 font-semibold text-card-foreground">接下来安排几场</h2>
        <div className="flex items-center gap-4">
          <button
            type="button"
            onClick={() => setMatches((m) => Math.max(1, m - 1))}
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
            aria-label="减少场次"
          >
            −
          </button>
          <input
            type="number"
            min={1}
            max={99}
            value={matches}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n)) setMatches(Math.max(1, Math.min(99, n)));
            }}
            aria-label="场次数"
            className="h-12 w-20 rounded-xl border border-border bg-background text-center text-xl font-bold outline-none focus-visible:ring-2 focus-visible:ring-ring"
          />
          <button
            type="button"
            onClick={() => setMatches((m) => Math.min(99, m + 1))}
            className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted text-foreground shadow-sm active:scale-95"
            aria-label="增加场次"
          >
            +
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card p-4 shadow-sm">
        <h2 className="mb-3 font-semibold text-card-foreground">项目名称（可选）</h2>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          maxLength={MAX_NAME_LENGTH}
          placeholder={`留空则命名为「${defaultName}」`}
          aria-label="项目名称"
          className="h-12 w-full rounded-xl border border-border bg-background px-4 text-base outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
      </div>

      <label className="flex items-start gap-3 rounded-2xl border border-border bg-card p-4 shadow-sm">
        <input
          type="checkbox"
          checked={avoidRematch}
          onChange={(e) => setAvoidRematch(e.target.checked)}
          className="mt-0.5 size-4 accent-primary"
        />
        <span>
          <span className="block text-sm font-medium text-card-foreground">
            减少已打对阵重复
          </span>
          <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">
            优先安排本项目里没打过的对阵，必要时仍可重复
          </span>
        </span>
      </label>

      <div className="rounded-2xl border border-border bg-card shadow-sm">
        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          aria-expanded={advancedOpen}
          className="flex w-full items-center justify-between p-4 text-left"
        >
          <span className="font-semibold text-card-foreground">高级选项</span>
          {advancedOpen ? (
            <ChevronUp className="size-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="size-4 text-muted-foreground" />
          )}
        </button>
        {advancedOpen && (
          <div className="space-y-2 border-t border-border px-4 pb-4 pt-4">
            <label className="block text-xs font-medium text-muted-foreground">
              随机种子（留空则由服务器随机取新值）
            </label>
            <input
              type="number"
              min={0}
              max={MAX_SEED}
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="仅复现排阵结果时填写"
              aria-label="随机种子"
              className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>
        )}
      </div>

      {error && (
        <div
          role="alert"
          className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-center text-sm text-destructive"
        >
          {error}
        </div>
      )}

      {ratingDown && (
        <div className="rounded-xl border border-dashed border-loss/50 bg-loss-bg p-3 text-center text-xs leading-relaxed text-loss">
          评分服务暂不可用，无法生成带预测的配对：
          {ratingDown}。恢复后请重试。
        </div>
      )}

      {myId === null && (
        <p className="text-center text-xs text-muted-foreground">
          请先在右上角选择球员身份，再创建项目。
        </p>
      )}

      <Button
        size="lg"
        disabled={!canSubmit}
        onClick={handleSubmit}
        className="h-14 w-full text-lg"
      >
        {loading ? "创建中…" : "创建配对项目"}
        {!loading && <Sparkles className="ml-2 size-4" />}
      </Button>
    </div>
  );
}
