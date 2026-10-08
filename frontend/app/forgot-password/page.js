"use client";

import { useState } from "react";
import Link from "next/link";
import { requestPasswordReset } from "@/lib/api";
import ThemeToggle from "@/components/ThemeToggle";

// Step 1 of "forgot password": ask for the account's email. The reply is
// the same whether or not the address belongs to an account.
export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await requestPasswordReset(email);
      setSent(true);
    } catch (err) {
      setError(err.message || "Could not send the reset link. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex-1 flex items-center justify-center bg-slate-950 px-4">
      <ThemeToggle className="fixed top-4 right-4" />
      <div className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-xl">
        <h1 className="text-2xl font-semibold text-white mb-1">Forgot password</h1>

        {sent ? (
          <>
            <div className="mt-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-300">
              If an account uses <span className="font-medium text-emerald-200">{email}</span>, a reset link is on its
              way. It works once and expires in 30 minutes.
            </div>
            <p className="text-xs text-slate-500 mt-4">No email after a few minutes? Check your spam folder, or ask an admin to reset your password.</p>
            <Link href="/login" className="mt-6 block w-full text-center rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium py-2.5 transition-colors">
              Back to sign in
            </Link>
          </>
        ) : (
          <form onSubmit={handleSubmit}>
            <p className="text-slate-400 text-sm mb-6">Enter the email address of your account and we&apos;ll send you a link to choose a new password.</p>

            <label htmlFor="email" className="block text-xs font-medium text-slate-400 mb-1">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoFocus
              autoComplete="email"
              className="w-full mb-4 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />

            {error && <p className="text-sm text-rose-400 mb-4">{error}</p>}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white text-sm font-medium py-2.5 transition-colors"
            >
              {loading ? "Sending…" : "Send reset link"}
            </button>
            <Link href="/login" className="mt-4 block text-center text-xs text-slate-400 hover:text-slate-200">
              ← Back to sign in
            </Link>
          </form>
        )}
      </div>
    </div>
  );
}
