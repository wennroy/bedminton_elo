import { listPlayers, listMatchesByDate } from "@/lib/repo";
import { RecordForm } from "@/components/record-form";

// Players live in the runtime sqlite db — prerendering at build time would
// bake an empty player list into the static HTML.
export const dynamic = "force-dynamic";

type Slot = number | null;

interface RecordPageProps {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

export default async function RecordPage({ searchParams }: RecordPageProps) {
  const players = listPlayers();
  const matches = listMatchesByDate();
  const sp = await searchParams;

  // 配对页带过来的预填阵容(如 /record?pa1=1&pa2=2&pb1=3&pb2=4);
  // 非法或不存在的 id 对应槽位置空,用户手动补选
  const validIds = new Set(players.map((p) => p.id));
  const parsed = (["pa1", "pa2", "pb1", "pb2"] as const).map((key): Slot => {
    const raw = sp[key];
    const n = typeof raw === "string" ? Number(raw) : NaN;
    return Number.isInteger(n) && validIds.has(n) ? n : null;
  });
  const initialSlots = parsed.some((s) => s !== null)
    ? (parsed as [Slot, Slot, Slot, Slot])
    : undefined;

  return (
    <div className="flex flex-col gap-5 max-[760px]:gap-3.5">
      <div>
        <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
          MATCH RECORD
        </div>
        <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
          录入比赛
        </h1>
        <p className="mt-2 text-xs text-muted-foreground max-[760px]:text-[11px]">
          选择双方阵容，输入最终比分。
        </p>
      </div>
      <RecordForm
        players={players}
        initialSlots={initialSlots}
        matches={matches}
      />
    </div>
  );
}
