import { listPlayers } from "@/lib/repo";
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
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <h1 className="mb-6 text-xl font-bold text-foreground">快速记分</h1>
      <RecordForm players={players} initialSlots={initialSlots} />
    </main>
  );
}
