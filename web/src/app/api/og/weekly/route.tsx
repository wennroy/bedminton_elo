import { ImageResponse } from "next/og";
import { NextResponse } from "next/server";
import QRCode from "qrcode";
import type { ReactNode } from "react";
import {
  buildWeeklyStats,
  getWeekRange,
  weeklyDataVersion,
  OG_DESIGN_VERSION,
  type FunMatch,
  type UpsetMatch,
  type WeeklyStats,
} from "@/lib/weekly";

// Width-aware truncation: CJK/full-width chars count 2, ASCII counts 1,
// so romanized names get roughly twice the character budget of Chinese names.
const truncate = (s: string, maxWidth: number) => {
  const w = (c: string) => (c.codePointAt(0)! > 0xff ? 2 : 1);
  const chars = Array.from(s);
  if (chars.reduce((total, c) => total + w(c), 0) <= maxWidth) return s;
  let out = "";
  let used = 0;
  for (const c of chars) {
    if (used + w(c) > maxWidth - 2) break; // reserve room for the ellipsis
    out += c;
    used += w(c);
  }
  return `${out}…`;
};

const FONT_FAMILY =
  '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif';

// COURTSIDE 浅色令牌字面值(来源 globals.css :root;Satori 不解析 CSS 变量)
const BG = "#f4f5f0";
const INK = "#242923";
const MUTED = "#72796f";
const LINE = "#e2e5dc";
const ACCENT = "#d3f36b";
const ACCENT_INK = "#263411";
const SURFACE = "#ffffff";
const SURFACE_2 = "#eeefe9";
const COURT = "#262f29"; // 记分板深底,最佳组合横条视觉锚点
const WIN = "#4e6c1d";
const LOSS = "#ae5548";

interface BoardRow {
  name: string;
  /** 大数字部分,如 "6" / "+22" */
  value: string;
  /** 大数字后的小单位,如 "场" / "胜";ELO 榜无单位 */
  unit?: string;
  valueColor?: string;
}

/** 名次徽标:榜首 lime,其余浅灰底(奖牌色收敛到小面积徽标) */
function RankBadge({ rank }: { rank: number }) {
  const first = rank === 1;
  return (
    <div
      style={{
        width: 48,
        height: 48,
        borderRadius: 14,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 24,
        fontWeight: 800,
        flexShrink: 0,
        background: first ? ACCENT : SURFACE_2,
        color: first ? ACCENT_INK : MUTED,
      }}
    >
      {String(rank).padStart(2, "0")}
    </div>
  );
}

/** 单榜:标题 + 英文小标 + 大数字榜行(细分隔线) */
function Board({
  title,
  sub,
  rows,
  showDivider,
}: {
  title: string;
  sub: string;
  rows: BoardRow[];
  showDivider: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: 300,
        ...(showDivider
          ? { borderLeft: `2px solid ${LINE}`, paddingLeft: 26, marginLeft: 26 }
          : {}),
      }}
    >
      <div style={{ display: "flex", fontSize: 30, fontWeight: 750 }}>
        {title}
      </div>
      <div
        style={{
          display: "flex",
          fontSize: 20,
          fontWeight: 700,
          letterSpacing: 3,
          color: MUTED,
          marginTop: 6,
        }}
      >
        {sub}
      </div>
      <div style={{ display: "flex", flexDirection: "column", marginTop: 20 }}>
        {rows.map((row, i) => (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "12px 0",
              ...(i > 0 ? { borderTop: `2px solid ${LINE}` } : {}),
            }}
          >
            <RankBadge rank={i + 1} />
            <div
              style={{
                display: "flex",
                fontSize: 26,
                fontWeight: 650,
                marginLeft: 16,
                whiteSpace: "nowrap",
              }}
            >
              {row.name}
            </div>
            <div
              style={{
                marginLeft: "auto",
                display: "flex",
                alignItems: "baseline",
                gap: 6,
                fontSize: 40,
                fontWeight: 800,
                letterSpacing: -1,
                color: row.valueColor ?? INK,
                flexShrink: 0,
              }}
            >
              {row.value}
              {row.unit ? (
                <span
                  style={{
                    fontSize: 22,
                    fontWeight: 600,
                    color: MUTED,
                    letterSpacing: 0,
                  }}
                >
                  {row.unit}
                </span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Winner-first two-line layout: "wA / wB  24 : 22" bold on top, losers below.
function FunMatchLines({ m }: { m: FunMatch }) {
  const aWon = m.scoreA > m.scoreB;
  const winners = aWon ? m.teamA : m.teamB;
  const losers = aWon ? m.teamB : m.teamA;
  const wScore = aWon ? m.scoreA : m.scoreB;
  const lScore = aWon ? m.scoreB : m.scoreA;
  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <div
        style={{
          display: "flex",
          fontSize: 28,
          fontWeight: 700,
          marginTop: 6,
          whiteSpace: "nowrap",
        }}
      >
        {`${truncate(winners[0], 14)} / ${truncate(winners[1], 14)}`}
        <span style={{ fontWeight: 800, letterSpacing: -0.5, marginLeft: 14 }}>
          {`${wScore} : ${lScore}`}
        </span>
      </div>
      <div
        style={{
          display: "flex",
          fontSize: 22,
          color: MUTED,
          marginTop: 4,
          whiteSpace: "nowrap",
        }}
      >
        {`击败 ${truncate(losers[0], 16)} / ${truncate(losers[1], 16)}`}
      </div>
    </div>
  );
}

/** 趣闻单列数据行:图标 + 内容 + 右侧超大关键数字 */
function FunRow({
  icon,
  label,
  date,
  noteNum,
  noteUnit,
  noteLabel,
  noteColor,
  showDivider,
  children,
}: {
  icon: string;
  label: string;
  date?: string;
  noteNum: string;
  noteUnit?: string;
  noteLabel: string;
  noteColor?: string;
  showDivider: boolean;
  children: ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        padding: "14px 0",
        ...(showDivider ? { borderTop: `2px solid ${LINE}` } : {}),
      }}
    >
      <div
        style={{
          width: 60,
          height: 60,
          borderRadius: 18,
          background: SURFACE,
          border: `2px solid ${LINE}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 30,
          flexShrink: 0,
        }}
      >
        {icon}
      </div>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          marginLeft: 20,
          width: 640,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 14,
            fontSize: 24,
            fontWeight: 700,
          }}
        >
          {label}
          {date ? (
            <span style={{ fontSize: 20, fontWeight: 500, color: MUTED }}>
              {date.slice(5).replace("-", ".")}
            </span>
          ) : null}
        </div>
        {children}
      </div>
      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            fontSize: 52,
            fontWeight: 800,
            letterSpacing: -1,
            color: noteColor ?? INK,
          }}
        >
          {noteNum}
          {noteUnit ? (
            <span style={{ fontSize: 24, fontWeight: 600 }}>{noteUnit}</span>
          ) : null}
        </div>
        <div
          style={{ display: "flex", fontSize: 20, color: MUTED, marginTop: 2 }}
        >
          {noteLabel}
        </div>
      </div>
    </div>
  );
}

function WeeklyCard({
  stats,
  qrDataUrl,
}: {
  stats: WeeklyStats;
  qrDataUrl: string;
}) {
  const hasData = stats.attendance.length > 0;

  const eloColor = (change: number) =>
    change > 0 ? WIN : change < 0 ? LOSS : MUTED;

  const boards: { title: string; sub: string; rows: BoardRow[] }[] = [
    {
      title: "出勤榜",
      sub: "ATTENDANCE",
      rows: stats.attendance.slice(0, 3).map((s) => ({
        name: truncate(s.name, 11),
        value: String(s.matches),
        unit: "场",
      })),
    },
    {
      title: "战绩王",
      sub: "MOST WINS",
      rows: stats.winKing.slice(0, 3).map((s) => ({
        name: truncate(s.name, 11),
        value: String(s.wins),
        unit: "胜",
      })),
    },
    {
      title: "ELO 涨跌榜",
      sub: "ELO CHANGE",
      rows: stats.eloChanges.slice(0, 3).map((s) => ({
        name: truncate(s.name, 11),
        value: `${s.change > 0 ? "+" : ""}${s.change}`,
        valueColor: eloColor(s.change),
      })),
    },
  ];

  const funRows: ReactNode[] = [];
  const { fun } = stats;
  if (fun.closest) {
    funRows.push(
      <FunRow
        key="closest"
        icon="🎯"
        label="最胶着一战"
        date={fun.closest.date}
        noteNum={String(Math.abs(fun.closest.scoreA - fun.closest.scoreB))}
        noteUnit=" 分"
        noteLabel="分差"
        showDivider={funRows.length > 0}
      >
        <FunMatchLines m={fun.closest} />
      </FunRow>
    );
  }
  if (fun.blowout) {
    funRows.push(
      <FunRow
        key="blowout"
        icon="💥"
        label="本周惨案"
        date={fun.blowout.date}
        noteNum={String(Math.abs(fun.blowout.scoreA - fun.blowout.scoreB))}
        noteUnit=" 分"
        noteLabel="净胜"
        showDivider={funRows.length > 0}
      >
        <FunMatchLines m={fun.blowout} />
      </FunRow>
    );
  }
  if (fun.streakKing) {
    funRows.push(
      <FunRow
        key="streak"
        icon="🔥"
        label="周连胜王"
        noteNum={String(fun.streakKing.streak)}
        noteUnit=" 连胜"
        noteLabel="当前"
        showDivider={funRows.length > 0}
      >
        <div
          style={{
            display: "flex",
            fontSize: 28,
            fontWeight: 700,
            marginTop: 6,
            whiteSpace: "nowrap",
          }}
        >
          {truncate(fun.streakKing.name, 16)}
        </div>
        <div
          style={{
            display: "flex",
            fontSize: 22,
            color: MUTED,
            marginTop: 4,
          }}
        >
          本周未逢败绩
        </div>
      </FunRow>
    );
  }
  if (fun.upset) {
    const u: UpsetMatch = fun.upset;
    funRows.push(
      <FunRow
        key="upset"
        icon="😱"
        label="本周最大冷门"
        date={u.date}
        noteNum={`${Math.round(u.winnerWinProb * 100)}%`}
        noteLabel="赛前胜率"
        noteColor={LOSS}
        showDivider={funRows.length > 0}
      >
        <FunMatchLines m={u} />
      </FunRow>
    );
  }

  return (
    <div
      style={{
        width: "1080px",
        height: "1920px",
        display: "flex",
        flexDirection: "column",
        background: BG,
        padding: "64px",
        fontFamily: FONT_FAMILY,
        color: INK,
        position: "relative",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
        <div
          style={{
            width: 100,
            height: 100,
            borderRadius: 28,
            background: ACCENT,
            color: ACCENT_INK,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 52,
            transform: "rotate(-7deg)",
          }}
        >
          🏸
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              display: "flex",
              fontSize: 26,
              fontWeight: 700,
              letterSpacing: 8,
              color: MUTED,
            }}
          >
            卷技术小分队 · WEEKLY REPORT
          </div>
          <div
            style={{
              display: "flex",
              fontSize: 68,
              fontWeight: 800,
              letterSpacing: -2,
              marginTop: 8,
            }}
          >
            {`第 ${stats.weekNumber} 周战报`}
          </div>
          <div
            style={{
              display: "flex",
              fontSize: 30,
              color: MUTED,
              marginTop: 12,
              letterSpacing: 1,
            }}
          >
            {`${stats.weekStart.replaceAll("-", ".")} — ${stats.weekEnd
              .slice(5)
              .replace("-", ".")}`}
          </div>
        </div>
      </div>
      <div
        style={{
          display: "flex",
          width: 120,
          height: 10,
          borderRadius: 5,
          background: ACCENT,
          marginTop: 20,
        }}
      />

      {/* 三榜 */}
      {hasData && (
        <div style={{ display: "flex", marginTop: 40 }}>
          {boards.map((b, i) => (
            <Board
              key={b.title}
              title={b.title}
              sub={b.sub}
              rows={b.rows}
              showDivider={i > 0}
            />
          ))}
        </div>
      )}

      {/* 最佳组合:深色 court 横条 */}
      {stats.bestPair && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            marginTop: 40,
            borderRadius: 32,
            background: COURT,
            padding: "32px 48px",
          }}
        >
          <div
            style={{ display: "flex", flexDirection: "column", width: 620 }}
          >
            <div
              style={{
                display: "flex",
                fontSize: 24,
                fontWeight: 700,
                letterSpacing: 4,
                color: "rgba(255,255,255,0.6)",
              }}
            >
              最佳组合 · BEST PAIR
            </div>
            <div
              style={{
                display: "flex",
                fontSize: 50,
                fontWeight: 800,
                color: "#ffffff",
                marginTop: 12,
                whiteSpace: "nowrap",
              }}
            >
              {`${truncate(stats.bestPair.playerA, 20)} / ${truncate(
                stats.bestPair.playerB,
                20
              )}`}
            </div>
            <div
              style={{
                display: "flex",
                fontSize: 28,
                color: "rgba(255,255,255,0.85)",
                marginTop: 12,
              }}
            >
              {`${stats.bestPair.wins} 胜 ${
                stats.bestPair.total - stats.bestPair.wins
              } 负`}
            </div>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-end",
              marginLeft: "auto",
            }}
          >
            <div
              style={{
                display: "flex",
                fontSize: 84,
                fontWeight: 800,
                color: ACCENT,
                letterSpacing: -2,
              }}
            >
              {`${Math.round(stats.bestPair.winRate * 100)}%`}
            </div>
            <div
              style={{
                display: "flex",
                fontSize: 24,
                color: "rgba(255,255,255,0.6)",
                marginTop: 4,
              }}
            >
              胜率
            </div>
          </div>
        </div>
      )}

      {/* 本周趣闻 */}
      {funRows.length > 0 && (
        <div
          style={{ display: "flex", flexDirection: "column", marginTop: 40 }}
        >
          <div
            style={{
              display: "flex",
              fontSize: 22,
              fontWeight: 700,
              letterSpacing: 4,
              color: MUTED,
            }}
          >
            本周趣闻 · HIGHLIGHTS
          </div>
          <div
            style={{ display: "flex", flexDirection: "column", marginTop: 4 }}
          >
            {funRows}
          </div>
        </div>
      )}

      {!hasData && (
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginTop: 80,
            fontSize: 32,
            color: MUTED,
          }}
        >
          本周暂无比赛记录
        </div>
      )}

      {/* 底部:绝对定位,与上方内容区无重叠风险(内容最大高度约 1500) */}
      <div
        style={{
          position: "absolute",
          left: 64,
          right: 64,
          bottom: 64,
          display: "flex",
          alignItems: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            background: SURFACE,
            border: `2px solid ${LINE}`,
            borderRadius: 22,
            padding: 10,
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
          <div style={{ display: "flex", fontSize: 34, fontWeight: 750 }}>
            扫码查看完整排行榜
          </div>
          <div
            style={{
              display: "flex",
              fontSize: 26,
              color: MUTED,
              marginTop: 8,
            }}
          >
            bedminton.wennroy.com
          </div>
          <div
            style={{
              display: "flex",
              fontSize: 20,
              color: MUTED,
              marginTop: 14,
            }}
          >
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
    const stats = buildWeeklyStats(weekStart);
    // 协商缓存:指纹不变 → 304 短路,跳过 QR 生成与 Satori 渲染。
    // no-cache = 允许存储但每次用前必须回源校验,取代 ImageResponse
    // 默认的 immutable 一年缓存(那正是数据更新后仍出旧图的根因)。
    const etag = `"${weeklyDataVersion(stats)}-${OG_DESIGN_VERSION}"`;
    const cacheHeaders = { "Cache-Control": "no-cache", ETag: etag };
    if (request.headers.get("if-none-match") === etag) {
      return new Response(null, { status: 304, headers: cacheHeaders });
    }
    const qrDataUrl = await QRCode.toDataURL(
      "https://bedminton.wennroy.com/",
      { width: 300, margin: 0, color: { dark: INK, light: "#ffffff" } }
    );
    return new ImageResponse(
      <WeeklyCard stats={stats} qrDataUrl={qrDataUrl} />,
      { width: 1080, height: 1920, headers: cacheHeaders }
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
