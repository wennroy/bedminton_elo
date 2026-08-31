"use client";

import * as React from "react";
import { getMyPlayerId } from "@/lib/identity";
import { IdentityPicker } from "@/components/identity-picker";
import { PlayerAvatar } from "@/components/player-avatar";

const MOCK_PLAYERS = [
  { id: 1, name: "陈雨菲" },
  { id: 2, name: "郑思维" },
  { id: 3, name: "黄雅琼" },
  { id: 4, name: "石宇奇" },
  { id: 5, name: "凡晨组合" },
  { id: 6, name: "李俊慧" },
  { id: 7, name: "刘雨辰" },
  { id: 8, name: "何冰娇" },
];

export default function HomePage() {
  const [myId, setMyId] = React.useState<number | null>(null);

  React.useEffect(() => {
    setMyId(getMyPlayerId());
  }, []);

  const myName = React.useMemo(
    () => MOCK_PLAYERS.find((p) => p.id === myId)?.name,
    [myId]
  );

  return (
    <main className="flex min-h-full flex-col items-center justify-center px-6 text-center">
      <IdentityPicker players={MOCK_PLAYERS} onSelect={setMyId} />

      <h1 className="text-3xl font-bold tracking-tight">卷技术小分队🏸</h1>
      <p className="mt-2 text-muted-foreground">首页（排行榜）占位</p>

      {myName ? (
        <div className="mt-8 flex flex-col items-center gap-2">
          <PlayerAvatar name={myName} size="lg" />
          <p className="text-lg font-medium">当前身份：{myName}</p>
        </div>
      ) : (
        <p className="mt-8 text-sm text-muted-foreground">请选择你的身份</p>
      )}
    </main>
  );
}
