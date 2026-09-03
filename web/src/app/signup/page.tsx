import {
  getActiveSessionDate,
  formatSessionDate,
  listSignups,
  signupSummary,
} from "@/lib/signup";
import { SignupForm } from "./signup-form";

export const dynamic = "force-dynamic";

export default async function SignupPage() {
  const sessionDate = getActiveSessionDate(new Date());
  const signups = listSignups(sessionDate);
  const summary = signupSummary(sessionDate);

  return (
    <main className="min-h-full bg-background px-4 pb-28 pt-4">
      <h1 className="mb-1 text-xl font-bold text-foreground">周三局报名</h1>
      <p className="mb-4 text-sm text-muted-foreground">
        每周三 18:00–20:00 · 本期 {formatSessionDate(sessionDate)}
      </p>
      <SignupForm
        sessionDate={sessionDate}
        signups={signups}
        summary={summary}
      />
    </main>
  );
}
