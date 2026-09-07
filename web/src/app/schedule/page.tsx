import { listPlayers } from "@/lib/repo";
import { ScheduleForm } from "./schedule-form";

export const dynamic = "force-dynamic";

export default async function SchedulePage() {
  const players = listPlayers();

  return (
    <div>
      <h1 className="mb-4 text-xl font-bold text-foreground">配对生成</h1>
      <ScheduleForm players={players} />
    </div>
  );
}
