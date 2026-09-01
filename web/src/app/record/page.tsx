import { listPlayers } from "@/lib/repo";
import { RecordForm } from "@/components/record-form";

// Players live in the runtime sqlite db — prerendering at build time would
// bake an empty player list into the static HTML.
export const dynamic = "force-dynamic";

export default async function RecordPage() {
  const players = listPlayers();

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <h1 className="mb-6 text-xl font-bold text-foreground">快速记分</h1>
      <RecordForm players={players} />
    </main>
  );
}
