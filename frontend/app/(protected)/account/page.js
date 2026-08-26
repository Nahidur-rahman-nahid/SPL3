"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { changePassword } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

export default function AccountPage() {
  const router = useRouter();
  const { user, refetch } = useAuth();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const forced = Boolean(user?.must_change_password);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    if (newPassword !== confirmPassword) {
      setError("New password and confirmation don't match.");
      return;
    }
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    setSubmitting(true);
    try {
      await changePassword(currentPassword, newPassword);
      await refetch();
      setSuccess(true);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      if (forced) setTimeout(() => router.replace("/dashboard"), 1200);
    } catch (err) {
      setError(err.message || "Failed to change password.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex-1 flex items-center justify-center bg-slate-950 px-4">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-xl"
      >
        <h1 className="text-xl font-semibold text-white mb-1">Change password</h1>
        <p className="text-slate-400 text-sm mb-6">
          {forced
            ? "Your account was created with a temporary password — set a new one to continue."
            : `Signed in as ${user?.username}`}
        </p>

        <label className="block text-xs font-medium text-slate-400 mb-1">Current password</label>
        <input
          type="password"
          value={currentPassword}
          onChange={(e) => setCurrentPassword(e.target.value)}
          required
          autoComplete="current-password"
          className="w-full mb-4 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />

        <label className="block text-xs font-medium text-slate-400 mb-1">New password</label>
        <input
          type="password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          required
          minLength={8}
          autoComplete="new-password"
          className="w-full mb-4 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />

        <label className="block text-xs font-medium text-slate-400 mb-1">Confirm new password</label>
        <input
          type="password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          minLength={8}
          autoComplete="new-password"
          className="w-full mb-4 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />

        {error && <p className="text-sm text-rose-400 mb-4">{error}</p>}
        {success && <p className="text-sm text-emerald-400 mb-4">Password changed.</p>}

        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white text-sm font-medium py-2.5 transition-colors"
        >
          {submitting ? "Saving…" : "Change password"}
        </button>

        {!forced && (
          <button
            type="button"
            onClick={() => router.push("/dashboard")}
            className="w-full mt-2 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium py-2.5 transition-colors"
          >
            Cancel
          </button>
        )}
      </form>
    </div>
  );
}
