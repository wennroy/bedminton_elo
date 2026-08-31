"use client";

import * as React from "react";
import { getMyPlayerId, setMyPlayerId } from "@/lib/identity";
import { PlayerAvatar } from "@/components/player-avatar";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface IdentityPickerProps {
  players: { id: number; name: string }[];
  onSelect?: (id: number) => void;
}

export function IdentityPicker({ players, onSelect }: IdentityPickerProps) {
  const [open, setOpen] = React.useState(false);
  const [selectedId, setSelectedId] = React.useState<number | null>(null);

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
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent showCloseButton={false} className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>你是谁？</DialogTitle>
          <DialogDescription>
            点选你的名字，之后会自动记住你的身份。
          </DialogDescription>
        </DialogHeader>
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
        </div>
      </DialogContent>
    </Dialog>
  );
}
