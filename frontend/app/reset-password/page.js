import { Suspense } from "react";
import ResetPasswordForm from "./ResetPasswordForm";
import ThemeToggle from "@/components/ThemeToggle";

// Step 2 of "forgot password": the page the emailed link opens
// (/reset-password?token=…). The form reads the token from the URL, so it
// sits in a Suspense boundary and renders on the client.
export default function ResetPasswordPage() {
  return (
    <div className="min-h-screen flex-1 flex items-center justify-center bg-slate-950 px-4">
      <ThemeToggle className="fixed top-4 right-4" />
      <div className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-xl">
        <h1 className="text-2xl font-semibold text-white mb-1">Choose a new password</h1>
        <Suspense fallback={<p className="text-slate-500 text-sm mt-4">Loading…</p>}>
          <ResetPasswordForm />
        </Suspense>
      </div>
    </div>
  );
}
