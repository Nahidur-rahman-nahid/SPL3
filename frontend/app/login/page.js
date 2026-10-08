"use client";

import Image from "next/image";
import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { login } from "@/lib/api";
import ThemeToggle from "@/components/ThemeToggle";
import { useAuth } from "@/lib/auth-context";

export default function LoginPage() {
  const router = useRouter();
  const { refetch } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const loggedInUser = await login(username, password);
      await refetch();
      router.push(loggedInUser.must_change_password ? "/account" : "/dashboard");
    } catch (err) {
      setError(err.message || "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex-1 flex items-center justify-center bg-slate-950 px-4">
      <ThemeToggle className="fixed top-4 right-4" />
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm bg-slate-900 border border-slate-800 rounded-xl p-8 shadow-xl"
      >
        <Image src="/fraudscope-icon.svg" alt="" width={56} height={56} priority className="mb-4" />
        <h1 className="text-2xl font-semibold text-white mb-1">FraudScope</h1>
        <p className="text-slate-400 text-sm mb-6">Sign in to the monitoring dashboard</p>

        <label className="block text-xs font-medium text-slate-400 mb-1">Username</label>
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full mb-4 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
          autoComplete="username"
          autoFocus
        />

        <div className="flex items-center justify-between mb-1">
          <label className="block text-xs font-medium text-slate-400">Password</label>
          <Link href="/forgot-password" className="text-xs text-indigo-400 hover:text-indigo-300">
            Forgot password?
          </Link>
        </div>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full mb-4 rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
          autoComplete="current-password"
        />

        {error && <p className="text-sm text-rose-400 mb-4">{error}</p>}

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white text-sm font-medium py-2.5 transition-colors"
        >
          {loading ? "Signing in…" : "Sign in"}
        </button>

        <p className="text-xs text-slate-500 mt-4">
          No account? Ask an admin to create one for you.
        </p>
      </form>
    </div>
  );
}
