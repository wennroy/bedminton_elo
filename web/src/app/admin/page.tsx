"use client";

import * as React from "react";
import Link from "next/link";
import {
  getAdminKey,
  setAdminKey,
  clearAdminKey,
} from "@/lib/admin";
import { Button } from "@/components/ui/button";
import { ChevronLeft, Lock, Trash2, Pencil, UserX } from "lucide-react";

interface Player {
  id: number;
  name: string;
}

interface Match {
  id: number;
  pa1: number;
  pa2: number;
  pb1: number;
  pb2: number;
  scoreA: number;
  scoreB: number;
  playedAt: string;
  createdAt: string;
  pa1Name: string;
  pa2Name: string;
  pb1Name: string;
  pb2Name: string;
}

const PAGE_SIZE = 20;

export default function AdminPage() {
  const [adminKey, setLocalAdminKey] = React.useState<string | null>(null);
  const [passwordInput, setPasswordInput] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    setLocalAdminKey(getAdminKey());
  }, []);

  function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setAdminKey(passwordInput);
    setLocalAdminKey(passwordInput);
    setPasswordInput("");
  }

  function handleLogout() {
    clearAdminKey();
    setLocalAdminKey(null);
  }

  if (adminKey === null) {
    return (
      <main className="min-h-full bg-background px-4 pb-28 pt-4">
        <div className="mb-6 flex items-center gap-2">
          <Button variant="ghost" size="icon-sm" asChild>
            <Link href="/" aria-label="返回">
              <ChevronLeft className="size-5" />
            </Link>
          </Button>
          <h1 className="text-xl font-bold text-foreground">管理员登录</h1>
        </div>

        <form
          onSubmit={handleLogin}
          className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6 shadow-sm"
        >
          <div className="flex flex-col gap-2">
            <label htmlFor="admin-password" className="text-sm font-medium text-card-foreground">
              管理口令
            </label>
            <input
              id="admin-password"
              type="password"
              value={passwordInput}
              onChange={(e) => setPasswordInput(e.target.value)}
              placeholder="输入 ADMIN_PASSWORD"
              className="h-10 rounded-xl border border-border bg-background px-3 text-sm text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
          </div>
          {error && (
            <div className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-center text-sm text-destructive">
              {error}
            </div>
          )}
          <Button type="submit" disabled={!passwordInput}>
            解锁
          </Button>
        </form>
      </main>
    );
  }

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon-sm" asChild>
            <Link href="/" aria-label="返回">
              <ChevronLeft className="size-5" />
            </Link>
          </Button>
          <h1 className="text-xl font-bold text-foreground">管理后台</h1>
        </div>
        <Button variant="ghost" size="sm" onClick={handleLogout}>
          <Lock className="mr-1 size-4" />
          退出
        </Button>
      </div>

      <AdminDashboard adminKey={adminKey} />
    </main>
  );
}

function AdminDashboard({ adminKey }: { adminKey: string }) {
  const [players, setPlayers] = React.useState<Player[]>([]);
  const [matches, setMatches] = React.useState<Match[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [page, setPage] = React.useState(1);

  const fetchData = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [playersRes, matchesRes] = await Promise.all([
        fetch("/api/players", { headers: { "x-admin-key": adminKey } }),
        fetch("/api/matches", { headers: { "x-admin-key": adminKey } }),
      ]);
      if (!playersRes.ok) {
        const data = await playersRes.json();
        throw new Error(data.error || "加载球员失败");
      }
      if (!matchesRes.ok) {
        const data = await matchesRes.json();
        throw new Error(data.error || "加载比赛失败");
      }
      const playersData = (await playersRes.json()) as Player[];
      const matchesData = (await matchesRes.json()) as Match[];
      setPlayers(playersData);
      setMatches(
        matchesData.sort(
          (a, b) =>
            new Date(b.playedAt).getTime() - new Date(a.playedAt).getTime() ||
            b.id - a.id
        )
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [adminKey]);

  React.useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (loading) {
    return (
      <div className="py-12 text-center text-sm text-muted-foreground">
        加载中…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
        {error}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">比赛管理</h2>
        <MatchList
          adminKey={adminKey}
          matches={matches}
          players={players}
          page={page}
          setPage={setPage}
          onMutate={fetchData}
        />
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-bold text-foreground">球员管理</h2>
        <PlayerList
          adminKey={adminKey}
          players={players}
          matches={matches}
          onMutate={fetchData}
        />
      </section>
    </div>
  );
}

function MatchList({
  adminKey,
  matches,
  players,
  page,
  setPage,
  onMutate,
}: {
  adminKey: string;
  matches: Match[];
  players: Player[];
  page: number;
  setPage: (page: number) => void;
  onMutate: () => void;
}) {
  const [editingId, setEditingId] = React.useState<number | null>(null);
  const [scoreA, setScoreA] = React.useState("");
  const [scoreB, setScoreB] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const visibleMatches = matches.slice(0, page * PAGE_SIZE);
  const hasMore = visibleMatches.length < matches.length;

  function startEdit(match: Match) {
    setEditingId(match.id);
    setScoreA(String(match.scoreA));
    setScoreB(String(match.scoreB));
    setError(null);
  }

  function cancelEdit() {
    setEditingId(null);
    setScoreA("");
    setScoreB("");
    setError(null);
  }

  async function saveEdit(match: Match) {
    const sa = Number(scoreA);
    const sb = Number(scoreB);
    if (![sa, sb].every((n) => Number.isFinite(n) && n >= 0)) {
      setError("比分必须是非负数");
      return;
    }
    if (sa === sb) {
      setError("比分不能相同");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/admin/matches/${match.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          "x-admin-key": adminKey,
        },
        body: JSON.stringify({ scoreA: sa, scoreB: sb }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "保存失败");
        return;
      }
      setEditingId(null);
      onMutate();
    } catch (e) {
      setError(e instanceof Error ? e.message : "保存失败");
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(match: Match) {
    if (!confirm(`确定删除这场比赛？\n${formatMatch(match)}`)) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/matches/${match.id}`, {
        method: "DELETE",
        headers: { "x-admin-key": adminKey },
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "删除失败");
        return;
      }
      onMutate();
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {error && (
        <div className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}
      {visibleMatches.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          暂无比赛
        </div>
      ) : (
        <>
          <div className="space-y-2">
            {visibleMatches.map((match) => (
              <div
                key={match.id}
                className="rounded-2xl border border-border bg-card p-3 shadow-sm"
              >
                <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                  <span>{match.playedAt}</span>
                  <span>#{match.id}</span>
                </div>
                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-1 flex-col gap-1 text-sm">
                    <div className="text-card-foreground">
                      {match.pa1Name} / {match.pa2Name}
                    </div>
                    <div className="text-muted-foreground">
                      VS {match.pb1Name} / {match.pb2Name}
                    </div>
                  </div>
                  {editingId === match.id ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min={0}
                        max={99}
                        value={scoreA}
                        onChange={(e) => setScoreA(e.target.value)}
                        className="h-9 w-14 rounded-lg border border-border bg-background px-2 text-center text-sm font-bold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      />
                      <span className="text-muted-foreground">:</span>
                      <input
                        type="number"
                        min={0}
                        max={99}
                        value={scoreB}
                        onChange={(e) => setScoreB(e.target.value)}
                        className="h-9 w-14 rounded-lg border border-border bg-background px-2 text-center text-sm font-bold text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      />
                    </div>
                  ) : (
                    <div className="text-lg font-bold tabular-nums text-card-foreground">
                      {match.scoreA} : {match.scoreB}
                    </div>
                  )}
                </div>
                <div className="mt-3 flex justify-end gap-2">
                  {editingId === match.id ? (
                    <>
                      <Button
                        size="xs"
                        variant="outline"
                        onClick={cancelEdit}
                        disabled={busy}
                      >
                        取消
                      </Button>
                      <Button
                        size="xs"
                        onClick={() => saveEdit(match)}
                        disabled={busy}
                      >
                        保存
                      </Button>
                    </>
                  ) : (
                    <>
                      <Button
                        size="xs"
                        variant="outline"
                        onClick={() => startEdit(match)}
                      >
                        <Pencil className="mr-1 size-3" />
                        改比分
                      </Button>
                      <Button
                        size="xs"
                        variant="destructive"
                        onClick={() => handleDelete(match)}
                        disabled={busy}
                      >
                        <Trash2 className="mr-1 size-3" />
                        删除
                      </Button>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
          {hasMore && (
            <Button
              variant="outline"
              onClick={() => setPage(page + 1)}
              className="w-full"
            >
              加载更多
            </Button>
          )}
        </>
      )}
    </div>
  );
}

function PlayerList({
  adminKey,
  players,
  matches,
  onMutate,
}: {
  adminKey: string;
  players: Player[];
  matches: Match[];
  onMutate: () => void;
}) {
  const [editingId, setEditingId] = React.useState<number | null>(null);
  const [editName, setEditName] = React.useState("");
  const [mergeTargets, setMergeTargets] = React.useState<Map<number, string>>(
    new Map()
  );
  const [busyId, setBusyId] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  function getMergeTarget(playerId: number): string {
    return mergeTargets.get(playerId) ?? "";
  }

  function setMergeTarget(playerId: number, targetId: string): void {
    setMergeTargets((prev) => {
      const next = new Map(prev);
      next.set(playerId, targetId);
      return next;
    });
  }

  const matchCounts = React.useMemo(() => {
    const counts = new Map<number, number>();
    for (const p of players) counts.set(p.id, 0);
    for (const m of matches) {
      for (const id of [m.pa1, m.pa2, m.pb1, m.pb2]) {
        counts.set(id, (counts.get(id) || 0) + 1);
      }
    }
    return counts;
  }, [players, matches]);

  function startEdit(player: Player) {
    setEditingId(player.id);
    setEditName(player.name);
    setError(null);
  }

  function cancelEdit() {
    setEditingId(null);
    setEditName("");
    setError(null);
  }

  async function saveRename(player: Player) {
    const name = editName.trim();
    if (!name) {
      setError("姓名不能为空");
      return;
    }
    setBusyId(player.id);
    setError(null);
    try {
      const res = await fetch(`/api/admin/players/${player.id}/rename`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-admin-key": adminKey,
        },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "改名失败");
        return;
      }
      setEditingId(null);
      onMutate();
    } catch (e) {
      setError(e instanceof Error ? e.message : "改名失败");
    } finally {
      setBusyId(null);
    }
  }

  async function handleMerge(player: Player) {
    const targetId = Number(getMergeTarget(player.id));
    if (!Number.isFinite(targetId) || targetId === player.id) {
      setError("请选择有效的合并目标");
      return;
    }
    const target = players.find((p) => p.id === targetId);
    if (!target) {
      setError("合并目标不存在");
      return;
    }
    if (
      !confirm(
        `确定把 "${player.name}" 的所有记录合并到 "${target.name}"？\n操作后 "${player.name}" 将被删除。`
      )
    ) {
      return;
    }
    setBusyId(player.id);
    setError(null);
    try {
      const res = await fetch(`/api/admin/players/${player.id}/merge`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-admin-key": adminKey,
        },
        body: JSON.stringify({ toId: targetId }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "合并失败");
        return;
      }
      setMergeTarget(player.id, "");
      onMutate();
    } catch (e) {
      setError(e instanceof Error ? e.message : "合并失败");
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(player: Player) {
    const count = matchCounts.get(player.id) || 0;
    if (count > 0) {
      setError("该球员有比赛记录，请先合并到其他球员");
      return;
    }
    if (!confirm(`确定删除球员 "${player.name}"？`)) return;
    setBusyId(player.id);
    setError(null);
    try {
      const res = await fetch(`/api/admin/players/${player.id}`, {
        method: "DELETE",
        headers: { "x-admin-key": adminKey },
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "删除失败");
        return;
      }
      onMutate();
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {error && (
        <div className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}
      {players.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
          暂无球员
        </div>
      ) : (
        <div className="space-y-2">
          {players.map((player) => {
            const count = matchCounts.get(player.id) || 0;
            const canDelete = count === 0;
            return (
              <div
                key={player.id}
                className="rounded-2xl border border-border bg-card p-3 shadow-sm"
              >
                <div className="flex items-center justify-between gap-3">
                  {editingId === player.id ? (
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="h-9 flex-1 rounded-lg border border-border bg-background px-3 text-sm text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    />
                  ) : (
                    <div className="flex flex-col">
                      <span className="font-medium text-card-foreground">
                        {player.name}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {count} 场比赛
                      </span>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    {editingId === player.id ? (
                      <>
                        <Button
                          size="xs"
                          variant="outline"
                          onClick={cancelEdit}
                          disabled={busyId === player.id}
                        >
                          取消
                        </Button>
                        <Button
                          size="xs"
                          onClick={() => saveRename(player)}
                          disabled={busyId === player.id}
                        >
                          保存
                        </Button>
                      </>
                    ) : (
                      <Button
                        size="xs"
                        variant="outline"
                        onClick={() => startEdit(player)}
                      >
                        <Pencil className="mr-1 size-3" />
                        改名
                      </Button>
                    )}
                  </div>
                </div>

                <div className="mt-3 flex items-center gap-2">
                  <select
                    value={getMergeTarget(player.id)}
                    onChange={(e) => setMergeTarget(player.id, e.target.value)}
                    className="h-9 flex-1 rounded-lg border border-border bg-background px-2 text-sm text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <option value="">合并到…</option>
                    {players
                      .filter((p) => p.id !== player.id)
                      .map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name}
                        </option>
                      ))}
                  </select>
                  <Button
                    size="xs"
                    variant="secondary"
                    onClick={() => handleMerge(player)}
                    disabled={busyId === player.id || !getMergeTarget(player.id)}
                  >
                    合并
                  </Button>
                  <Button
                    size="xs"
                    variant={canDelete ? "destructive" : "outline"}
                    onClick={() => handleDelete(player)}
                    disabled={busyId === player.id || !canDelete}
                    title={
                      canDelete
                        ? "删除"
                        : "该球员有比赛记录，请先合并到其他球员"
                    }
                  >
                    <UserX className="mr-1 size-3" />
                    删除
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function formatMatch(match: Match): string {
  return `${match.pa1Name} / ${match.pa2Name} ${match.scoreA}:${match.scoreB} ${match.pb1Name} / ${match.pb2Name}`;
}
