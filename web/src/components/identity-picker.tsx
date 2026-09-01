"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { getMyPlayerId, setMyPlayerId } from "@/lib/identity";
import { PlayerAvatar } from "@/components/player-avatar";
import { Button } from "@/components/ui/button";
import { UserPlus } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface IdentityPickerProps {
  players: { id: number; name: string }[];
  onSelect?: (id: number) => void;
  /** Optional trigger to reopen the picker (e.g. 更换身份 button on /me). */
  trigger?: React.ReactNode;
}

export function IdentityPicker({ players, onSelect, trigger }: IdentityPickerProps) {
  const router = useRouter();
  const [open, setOpen] = React.useState(false);
  const [selectedId, setSelectedId] = React.useState<number | null>(null);
  const [adding, setAdding] = React.useState(false);
  const [newName, setNewName] = React.useState("");
  const [addError, setAddError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);

  React.useEffect(() => {
    const current = getMyPlayerId();
    if (current === null) {
      setOpen(true);
    } else {
      setSelectedId(current);
    }
  }, []);

  const handleSelect = (id: number) => {
    setMyPlayerId(id);
    setSelectedId(id);
    onSelect?.(id);
    setOpen(false);
  };

  const handleOpenChange = (next: boolean) => {
    if (!next && selectedId === null) return;
    setOpen(next);
    if (!next) {
      setAdding(false);
      setNewName("");
      setAddError(null);
    }
  };

  async function handleAdd(event: React.FormEvent) {
    event.preventDefault();
    const name = newName.trim();
    if (!name || submitting) return;
    setSubmitting(true);
    setAddError(null);
    try {
      const res = await fetch("/api/players", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const data = await res.json();
      if (!res.ok) {
        setAddError(data.error || "添加失败");
        return;
      }
      // Refresh so server-rendered player lists (e.g. /record) include the
      // new player, then auto-select the new identity.
      router.refresh();
      handleSelect(data.id);
    } catch {
      setAddError("网络错误，请重试");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      {trigger && <DialogTrigger asChild>{trigger}</DialogTrigger>}
      <DialogContent showCloseButton={false} className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{adding ? "添加新球员" : "你是谁？"}</DialogTitle>
          <DialogDescription>
            {adding
              ? "输入名字，添加后会自动选为你的身份。"
              : "点选你的名字，之后会自动记住你的身份。"}
          </DialogDescription>
        </DialogHeader>

        {adding ? (
          <form onSubmit={handleAdd} className="flex flex-col gap-3 pt-2">
            <input
              autoFocus
              value={newName}
              onChange={(e) => {
                setNewName(e.target.value);
                setAddError(null);
              }}
              maxLength={20}
              placeholder="输入名字"
              className="h-12 w-full rounded-xl border border-border bg-background px-4 text-base outline-none focus-visible:ring-2 focus-visible:ring-ring"
            />
            {addError && (
              <p className="text-sm text-destructive">{addError}</p>
            )}
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                className="flex-1"
                onClick={() => {
                  setAdding(false);
                  setNewName("");
                  setAddError(null);
                }}
              >
                返回
              </Button>
              <Button
                type="submit"
                className="flex-1"
                disabled={submitting || !newName.trim()}
              >
                {submitting ? "添加中…" : "确认添加"}
              </Button>
            </div>
          </form>
        ) : (
          <div className="grid grid-cols-3 gap-3 pt-2 sm:grid-cols-4">
            {players.map((player) => (
              <button
                key={player.id}
                onClick={() => handleSelect(player.id)}
                className="flex flex-col items-center gap-2 rounded-xl border border-transparent p-2 transition-colors hover:bg-muted focus-visible:rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <PlayerAvatar name={player.name} size="md" />
                <span className="text-sm font-medium text-foreground">
                  {player.name}
                </span>
              </button>
            ))}
            <button
              onClick={() => setAdding(true)}
              className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:rounded-xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                <UserPlus className="size-5" />
              </span>
              <span className="text-sm font-medium">新球员</span>
            </button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
