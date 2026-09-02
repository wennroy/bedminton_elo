import { ImageResponse } from "next/og";
import { NextResponse } from "next/server";
import QRCode from "qrcode";
import type { ReactNode } from "react";
import {
  buildWeeklyStats,
  getWeekRange,
  type FunMatch,
  type UpsetMatch,
} from "@/lib/weekly";

const truncate = (s: string, max: number) =>
  s.length > max ? `${s.slice(0, max - 1)}…` : s;

const FONT_FAMILY =
  '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif';

const CARD_SHADOW = "0 10px 28px rgba(15, 23, 42, 0.07)";

interface PodiumEntry {
  name: string;
  value: string;
  valueColor: string;
}

function PodiumCard({
  title,
  entries,
}: {
  title: string;
  entries: PodiumEntry[];
}) {
  const pillarHeights = [128, 96, 76];
  const pillarBgs = [
    "linear-gradient(180deg, #ffd968 0%, #f0a500 100%)",
    "linear-gradient(180deg, #e4e9f0 0%, #b4bfce 100%)",
    "linear-gradient(180deg, #e7ab7c 0%, #c97a3d 100%)",
  ];
  const order = [1, 0, 2]; // 2nd left, 1st center, 3rd right
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: 302,
        background: "#ffffff",
        borderRadius: 28,
        paddingTop: 26,
        boxShadow: CARD_SHADOW,
      }}
    >
      <div style={{ fontSize: 36, fontWeight: 700, color: "#374151" }}>
        {title}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "center",
          gap: 10,
          marginTop: 22,
        }}
      >
        {order.map((rank) => {
          const entry = entries[rank];
          if (!entry) {
            return (
              <div
                key={rank}
                style={{ display: "flex", width: 80, height: pillarHeights[rank] }}
              />
            );
          }
          return (
            <div
              key={rank}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                width: 80,
              }}
            >
              <div
                style={{
                  fontSize: 24,
                  fontWeight: 600,
                  color: "#111827",
                  whiteSpace: "nowrap",
                }}
              >
                {truncate(entry.name, 4)}
              </div>
              <div
                style={{
                  fontSize: 24,
                  fontWeight: 700,
                  color: entry.valueColor,
                  marginTop: 4,
                  whiteSpace: "nowrap",
                }}
              >
                {entry.value}
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  paddingTop: 12,
                  width: 80,
                  height: pillarHeights[rank],
                  marginTop: 12,
                  borderRadius: "16px 16px 0 0",
                  background: pillarBgs[rank],
                  fontSize: 34,
                  fontWeight: 800,
                  color: "rgba(17, 24, 39, 0.45)",
                }}
              >
                {rank + 1}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function FunCard({
  icon,
  label,
  date,
  children,
}: {
  icon: string;
  label: string;
  date?: string;
  children: ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: 464,
        height: 196,
        background: "#ffffff",
        borderRadius: 24,
        padding: "22px 28px",
        boxShadow: CARD_SHADOW,
      }}
    >
      <div style={{ display: "flex", alignItems: "center" }}>
        <div style={{ display: "flex", fontSize: 28, marginRight: 10 }}>
          {icon}
        </div>
        <div
          style={{
            display: "flex",
            flex: 1,
            fontSize: 26,
            fontWeight: 600,
            color: "#6b7280",
          }}
        >
          {label}
        </div>
        {date ? (
          <div style={{ display: "flex", fontSize: 22, color: "#9ca3af" }}>
            {date.slice(5)}
          </div>
        ) : null}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          flex: 1,
          gap: 8,
        }}
      >
        {children}
      </div>
    </div>
  );
}

function scoreText(m: FunMatch) {
  const a = `${truncate(m.teamA[0], 3)}/${truncate(m.teamA[1], 3)}`;
  const b = `${truncate(m.teamB[0], 3)}/${truncate(m.teamB[1], 3)}`;
  return `${a}  ${m.scoreA} : ${m.scoreB}  ${b}`;
}

function WeeklyCard({
  weekStart,
  qrDataUrl,
}: {
  weekStart: string;
  qrDataUrl: string;
}) {
  const stats = buildWeeklyStats(weekStart);
  const hasData = stats.attendance.length > 0;

  const attendanceEntries: PodiumEntry[] = stats.attendance
    .slice(0, 3)
    .map((s) => ({
      name: s.name,
      value: `${s.matches} 场`,
      valueColor: "#374151",
    }));
  const winEntries: PodiumEntry[] = stats.winKing.slice(0, 3).map((s) => ({
    name: s.name,
    value: `${s.wins} 胜`,
    valueColor: "#374151",
  }));
  const eloEntries: PodiumEntry[] = stats.eloChanges.slice(0, 3).map((s) => ({
    name: s.name,
    value: `${s.change > 0 ? "+" : ""}${s.change}`,
    valueColor:
      s.change > 0 ? "#16a34a" : s.change < 0 ? "#dc2626" : "#6b7280",
  }));

  const funCards: ReactNode[] = [];
  const { fun } = stats;
  if (fun.closest) {
    funCards.push(
      <FunCard key="closest" icon="🎯" label="最胶着一战" date={fun.closest.date}>
        <div style={{ fontSize: 28, fontWeight: 700, color: "#111827" }}>
          {scoreText(fun.closest)}
        </div>
        <div style={{ fontSize: 24, color: "#6b7280" }}>
          {`分差仅 ${Math.abs(fun.closest.scoreA - fun.closest.scoreB)} 分`}
        </div>
      </FunCard>
    );
  }
  if (fun.blowout) {
    funCards.push(
      <FunCard key="blowout" icon="💥" label="本周惨案" date={fun.blowout.date}>
        <div style={{ fontSize: 28, fontWeight: 700, color: "#111827" }}>
          {scoreText(fun.blowout)}
        </div>
        <div style={{ fontSize: 24, color: "#6b7280" }}>
          {`净胜 ${Math.abs(fun.blowout.scoreA - fun.blowout.scoreB)} 分`}
        </div>
      </FunCard>
    );
  }
  if (fun.streakKing) {
    funCards.push(
      <FunCard key="streak" icon="🔥" label="周连胜王">
        <div style={{ fontSize: 30, fontWeight: 700, color: "#111827" }}>
          {`${truncate(fun.streakKing.name, 6)} · ${fun.streakKing.streak} 连胜`}
        </div>
      </FunCard>
    );
  }
  if (fun.upset) {
    const u: UpsetMatch = fun.upset;
    const aWon = u.scoreA > u.scoreB;
    const winners = aWon ? u.teamA : u.teamB;
    const losers = aWon ? u.teamB : u.teamA;
    funCards.push(
      <FunCard key="upset" icon="😱" label="本周最大冷门" date={u.date}>
        <div style={{ fontSize: 28, fontWeight: 700, color: "#111827" }}>
          {`${truncate(winners[0], 3)}/${truncate(winners[1], 3)} 胜 ${truncate(losers[0], 3)}/${truncate(losers[1], 3)}`}
        </div>
        <div style={{ fontSize: 24, fontWeight: 600, color: "#dc2626" }}>
          {`赛前胜率仅 ${Math.round(u.winnerWinProb * 100)}%`}
        </div>
      </FunCard>
    );
  }
  const funRows = [funCards.slice(0, 2), funCards.slice(2, 4)].filter(
    (row) => row.length > 0
  );

  return (
    <div
      style={{
        width: "1080px",
        height: "1920px",
        display: "flex",
        flexDirection: "column",
        background: "linear-gradient(160deg, #f8fafc 0%, #eef2f7 100%)",
        padding: "64px",
        fontFamily: FONT_FAMILY,
        color: "#1f2937",
        position: "relative",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: -180,
          right: -180,
          width: 480,
          height: 480,
          borderRadius: 240,
          background: "rgba(59, 130, 246, 0.08)",
          display: "flex",
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: -220,
          left: -220,
          width: 520,
          height: 520,
          borderRadius: 260,
          background: "rgba(240, 165, 0, 0.07)",
          display: "flex",
        }}
      />

      {/* Header */}
      <div style={{ display: "flex", flexDirection: "column" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
          <div
            style={{
              width: 100,
              height: 100,
              borderRadius: 28,
              background: "#111827",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 52,
            }}
          >
            🏸
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div
              style={{ fontSize: 30, color: "#6b7280", letterSpacing: 6 }}
            >
              卷技术小分队
            </div>
            <div style={{ fontSize: 68, fontWeight: 800, color: "#111827" }}>
              {`第 ${stats.weekNumber} 周战报`}
            </div>
          </div>
        </div>
        <div style={{ marginTop: 22, fontSize: 30, color: "#6b7280" }}>
          {`${stats.weekStart} ~ ${stats.weekEnd}`}
        </div>
      </div>

      {/* Podium leaderboards */}
      {hasData && (
        <div style={{ display: "flex", gap: 22, marginTop: 48 }}>
          <PodiumCard title="出勤榜" entries={attendanceEntries} />
          <PodiumCard title="战绩王" entries={winEntries} />
          <PodiumCard title="ELO 涨跌榜" entries={eloEntries} />
        </div>
      )}

      {/* Best pair */}
      {stats.bestPair && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            marginTop: 44,
            borderRadius: 32,
            background: "#111827",
            padding: "36px 48px",
            boxShadow: CARD_SHADOW,
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", flex: 1 }}>
            <div style={{ fontSize: 28, color: "rgba(255,255,255,0.65)" }}>
              最佳组合
            </div>
            <div
              style={{
                fontSize: 50,
                fontWeight: 800,
                color: "#ffffff",
                marginTop: 10,
                whiteSpace: "nowrap",
              }}
            >
              {`${truncate(stats.bestPair.playerA, 4)} / ${truncate(stats.bestPair.playerB, 4)}`}
            </div>
            <div
              style={{
                fontSize: 30,
                color: "rgba(255,255,255,0.85)",
                marginTop: 12,
              }}
            >
              {`${stats.bestPair.wins} 胜 ${stats.bestPair.total - stats.bestPair.wins} 负`}
            </div>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-end",
            }}
          >
            <div style={{ fontSize: 76, fontWeight: 800, color: "#fbbf24" }}>
              {`${Math.round(stats.bestPair.winRate * 100)}%`}
            </div>
            <div
              style={{ fontSize: 26, color: "rgba(255,255,255,0.65)" }}
            >
              胜率
            </div>
          </div>
        </div>
      )}

      {/* Fun section */}
      {funRows.length > 0 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 22,
            marginTop: 44,
          }}
        >
          {funRows.map((row, i) => (
            <div key={i} style={{ display: "flex", gap: 22 }}>
              {row}
            </div>
          ))}
        </div>
      )}

      {!hasData && (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginTop: 80,
            fontSize: 32,
            color: "#9ca3af",
          }}
        >
          本周暂无比赛记录
        </div>
      )}

      {/* Footer */}
      <div style={{ display: "flex", alignItems: "center", marginTop: 56 }}>
        <div
          style={{
            display: "flex",
            background: "#ffffff",
            borderRadius: 22,
            padding: 12,
            boxShadow: CARD_SHADOW,
          }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={qrDataUrl} width={150} height={150} alt="QR" />
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            marginLeft: 30,
          }}
        >
          <div style={{ fontSize: 36, fontWeight: 700, color: "#111827" }}>
            扫码查看完整排行榜
          </div>
          <div style={{ fontSize: 26, color: "#6b7280", marginTop: 10 }}>
            bedminton.wennroy.com
          </div>
          <div style={{ fontSize: 22, color: "#9ca3af", marginTop: 16 }}>
            卷技术小分队 · 羽毛球双打 ELO 排行榜
          </div>
        </div>
      </div>
    </div>
  );
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const week = url.searchParams.get("week");
  if (!week || !/^\d{4}-\d{2}-\d{2}$/.test(week)) {
    return NextResponse.json({ error: "Invalid week" }, { status: 400 });
  }

  try {
    const { weekStart } = getWeekRange(week);
    const qrDataUrl = await QRCode.toDataURL(
      "https://bedminton.wennroy.com/",
      { width: 300, margin: 0, color: { dark: "#111827", light: "#ffffff" } }
    );
    return new ImageResponse(
      <WeeklyCard weekStart={weekStart} qrDataUrl={qrDataUrl} />,
      { width: 1080, height: 1920 }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
