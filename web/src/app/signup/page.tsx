import {
  getActiveSessionDate,
  listSignups,
  signupSummary,
} from "@/lib/signup";
import { listPlayers } from "@/lib/repo";
import { SignupForm } from "./signup-form";

export const dynamic = "force-dynamic";

export default async function SignupPage() {
  const sessionDate = getActiveSessionDate(new Date());
  const signups = listSignups(sessionDate);
  const summary = signupSummary(sessionDate);
  const players = listPlayers();

  return (
    <div className="flex flex-col gap-6 max-[760px]:gap-5">
      <div className="flex items-center justify-between gap-5">
        <div>
          <div className="text-[10px] font-bold tracking-[2px] text-muted-foreground max-[760px]:text-[9px]">
            WEEKLY SESSION
          </div>
          <h1 className="mt-2 text-[30px] font-bold tracking-[-0.8px] text-foreground max-[760px]:text-[26px]">
            每周报名
          </h1>
        </div>
        <span className="inline-flex items-center rounded-[5px] bg-win-bg px-[7px] py-1 text-[10px] font-bold text-win">
          报名中
        </span>
      </div>

      <SignupForm
        sessionDate={sessionDate}
        signups={signups}
        summary={summary}
        players={players}
      />
    </div>
  );
}
