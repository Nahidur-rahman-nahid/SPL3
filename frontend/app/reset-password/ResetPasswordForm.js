"use client";

import { useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { resetPasswordWithToken } from "@/lib/api";

const inputClass =
  "w-full mb-4 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500";

export default function ResetPasswordForm() {
  const token = useSearchParams().get("token") || "";
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  if (!token) {
    return (
      <>
        <p className="text-sm text-rose-400 mt-4">This link is incomplete. Open the link from the email again, or request a new one.</p>
        <Link href="/forgot-password" className="mt-6 block text-center text-sm text-indigo-400 hover:text-indigo-300">
          Request a new link
        </Link>
      </>
    );
  }

  if (done) {
    return (
      <>
        <div className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">
          Your password has been changed. For safety you&apos;ve been signed out on every device.
        </div>
        <Link href="/login" className="mt-6 block w-full text-center rounded-md bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium py-2.5 transition-colors">
          Sign in
        </Link>
      </>
    );
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    if (password !== confirm) {
      setError("The two passwords don't match.");
      return;
    }
    if (password.length < 8) {
      setError("The password must be at least 8 characters.");
      return;
    }
    setLoading(true);
    try {
      await resetPasswordWithToken(token, password);
      setDone(true);
    } catch (err) {
      setError(err.message || "Could not reset the password.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <p className="text-slate-400 text-sm mb-6">Pick a new password for your account (at least 8 characters).</p>

      <label htmlFor="new-password" className="block text-xs font-medium text-slate-400 mb-1">New password</label>
      <input id="new-password" type="password" value={password} onChange={(e) => setPassword(e.target.value)}
        required minLength={8} autoFocus autoComplete="new-password" className={inputClass} />

      <label htmlFor="confirm-password" className="block text-xs font-medium text-slate-400 mb-1">Confirm new password</label>
      <input id="confirm-password" type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
        required minLength={8} autoComplete="new-password" className={inputClass} />

      {error && (
        <p className="text-sm text-rose-400 mb-4">
          {error}{" "}
          {error.includes("expired") && (
            <Link href="/forgot-password" className="text-indigo-400 hover:text-indigo-300 underline">Request a new link</Link>
          )}
        </p>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white text-sm font-medium py-2.5 transition-colors"
      >
        {loading ? "Saving…" : "Set new password"}
      </button>
    </form>
  );
}
