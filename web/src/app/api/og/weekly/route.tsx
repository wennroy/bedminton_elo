import { ImageResponse } from "next/og";
import { NextResponse } from "next/server";
import { buildWeeklyStats, getWeekRange } from "@/lib/weekly";

interface WeeklyOGProps {
  weekStart: string;
}

function WeeklyCard({ weekStart }: WeeklyOGProps) {
  const stats = buildWeeklyStats(weekStart);
  const topAttendance = stats.attendance.slice(0, 3);
  const topWins = stats.winKing.slice(0, 3);
  const topElo = stats.eloChanges.slice(0, 3);

  return (
    <div
      style={{
        width: "1080px",
        height: "1350px",
        display: "flex",
        flexDirection: "column",
        background: "linear-gradient(135deg, #fafafa 0%, #f3f4f6 100%)",
        padding: "80px",
        fontFamily:
          '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif',
        color: "#1f2937",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "24px" }}>
        <div
          style={{
            width: "80px",
            height: "80px",
            borderRadius: "24px",
            background: "#111827",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "40px",
          }}
        >
          🏸
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: "28px", color: "#6b7280" }}>
            卷技术小分队
          </div>
          <div style={{ fontSize: "56px", fontWeight: "bold" }}>
            {`第 ${stats.weekNumber} 周战报`}
          </div>
        </div>
      </div>

      <div
        style={{
          marginTop: "48px",
          fontSize: "24px",
          color: "#6b7280",
        }}
      >
        {`${stats.weekStart} ~ ${stats.weekEnd}`}
      </div>

      <div
        style={{
          marginTop: "64px",
          display: "flex",
          flexDirection: "column",
          gap: "40px",
        }}
      >
        <Section title="出勤榜" items={topAttendance.map((s, i) => rankRow(i, s.name, `${s.matches} 场`))} />
        <Section title="战绩王" items={topWins.map((s, i) => rankRow(i, s.name, `${s.wins} 胜`))} />
        <Section
          title="ELO 涨跌榜"
          items={topElo.map((s, i) =>
            rankRow(i, s.name, `${s.change > 0 ? "+" : ""}${s.change}`)
          )}
        />
      </div>

      {stats.bestPair && (
        <div
          style={{
            marginTop: "64px",
            borderRadius: "32px",
            background: "#111827",
            color: "#ffffff",
            padding: "48px",
            display: "flex",
            flexDirection: "column",
            gap: "16px",
          }}
        >
          <div style={{ fontSize: "28px", opacity: 0.8 }}>最佳组合</div>
          <div style={{ fontSize: "48px", fontWeight: "bold" }}>
            {`${stats.bestPair.playerA} / ${stats.bestPair.playerB}`}
          </div>
          <div style={{ fontSize: "32px" }}>
            {`${stats.bestPair.wins} 胜 ${stats.bestPair.total - stats.bestPair.wins} 负 · ${Math.round(stats.bestPair.winRate * 100)}%`}
          </div>
        </div>
      )}

      <div
        style={{
          marginTop: "auto",
          fontSize: "24px",
          color: "#9ca3af",
          textAlign: "center",
        }}
      >
        卷技术小分队 · 羽毛球双打 ELO 排行榜
      </div>
    </div>
  );
}

function Section({
  title,
  items,
}: {
  title: string;
  items: React.ReactNode[];
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ fontSize: "32px", fontWeight: "bold", color: "#374151" }}>
        {title}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "12px",
        }}
      >
        {items}
      </div>
    </div>
  );
}

function rankRow(index: number, name: string, value: string) {
  const medals = ["#fbbf24", "#9ca3af", "#b45309"];
  return (
    <div
      key={index}
      style={{
        display: "flex",
        alignItems: "center",
        gap: "20px",
        fontSize: "32px",
      }}
    >
      <div
        style={{
          width: "48px",
          height: "48px",
          borderRadius: "50%",
          background: medals[index] ?? "#e5e7eb",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontWeight: "bold",
          color: index < 2 ? "#ffffff" : "#1f2937",
        }}
      >
        {index + 1}
      </div>
      <div style={{ flex: 1, fontWeight: 500 }}>{name}</div>
      <div style={{ color: "#6b7280", fontWeight: 600 }}>{value}</div>
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
    return new ImageResponse(<WeeklyCard weekStart={weekStart} />, {
      width: 1080,
      height: 1350,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
