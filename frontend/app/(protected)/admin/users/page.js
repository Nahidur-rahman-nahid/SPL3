"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { createUser, listUsers, resetUserPassword, updateUser } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { useToast } from "@/components/Feedback";

const ROLES = ["ADMIN", "ANALYST"];

// Inline edit of a user's email — the address "Forgot password" links go to.
function EmailCell({ user, onSave }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(user.email || "");
  const [saving, setSaving] = useState(false);

  async function save() {
    setSaving(true);
    const ok = await onSave(user, value.trim());
    setSaving(false);
    if (ok) setEditing(false);
  }

  if (!editing) {
    return (
      <span className="inline-flex items-center gap-2">
        <span className={user.email ? "text-slate-400" : "text-amber-400/80"}>{user.email || "no email"}</span>
        <button
          onClick={() => { setValue(user.email || ""); setEditing(true); }}
          className="text-xs text-indigo-400 hover:text-indigo-300"
          aria-label={`Edit email of ${user.username}`}
        >
          Edit
        </button>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      <input
        type="email"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") save();
          if (e.key === "Escape") setEditing(false);
        }}
        autoFocus
        placeholder="name@example.com"
        className="w-48 rounded bg-slate-800 border border-slate-700 px-2 py-1 text-xs text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
      />
      <button onClick={save} disabled={saving} className="text-xs px-2 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-60">
        {saving ? "…" : "Save"}
      </button>
      <button onClick={() => setEditing(false)} className="text-xs px-2 py-1 rounded text-slate-400 hover:text-white">
        Cancel
      </button>
    </span>
  );
}

function RevealPassword({ label, value, onDismiss }) {
  return (
    <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3 flex items-start justify-between gap-4">
      <div>
        <p className="text-xs text-amber-400 font-medium mb-1">{label} — shown once, share it securely</p>
        <code className="text-sm text-amber-200 select-all">{value}</code>
      </div>
      <button onClick={onDismiss} className="text-xs text-slate-400 hover:text-white shrink-0">
        Dismiss
      </button>
    </div>
  );
}

export default function AdminUsersPage() {
  const { role } = useAuth();
  const toast = useToast();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reveal, setReveal] = useState(null); // { label, value }

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [newRole, setNewRole] = useState("ANALYST");
  const [creating, setCreating] = useState(false);

  const loadUsers = useCallback(async () => {
    setLoading(true);
    try {
      setUsers(await listUsers());
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load users.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (role === "ADMIN") loadUsers();
  }, [role, loadUsers]);

  if (role !== "ADMIN") {
    return (
      <div className="flex-1 flex items-center justify-center py-20 text-slate-400">
        <div className="text-center">
          <p className="text-lg font-medium text-slate-200 mb-2">Admins only</p>
          <p className="text-sm mb-4">You don&apos;t have permission to manage users.</p>
          <Link href="/dashboard" className="text-sm text-indigo-400 hover:text-indigo-300">
            ← Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  async function handleCreate(e) {
    e.preventDefault();
    setCreating(true);
    setError("");
    try {
      const { user, temp_password } = await createUser({ username, email, role: newRole });
      setUsers((prev) => [...prev, user]);
      setReveal({ label: `Temp password for ${user.username}`, value: temp_password });
      setUsername("");
      setEmail("");
      setNewRole("ANALYST");
    } catch (err) {
      setError(err.message || "Failed to create user.");
    } finally {
      setCreating(false);
    }
  }

  async function handleToggleActive(u) {
    try {
      const updated = await updateUser(u.id, { is_active: !u.is_active });
      setUsers((prev) => prev.map((x) => (x.id === u.id ? updated : x)));
    } catch (err) {
      setError(err.message || "Failed to update user.");
    }
  }

  async function handleRoleChange(u, role) {
    try {
      const updated = await updateUser(u.id, { role });
      setUsers((prev) => prev.map((x) => (x.id === u.id ? updated : x)));
    } catch (err) {
      setError(err.message || "Failed to update user.");
    }
  }

  async function handleEmailSave(u, email) {
    try {
      const updated = await updateUser(u.id, { email });
      setUsers((prev) => prev.map((x) => (x.id === u.id ? updated : x)));
      toast.success(email ? `Email saved for ${u.username}` : `Email removed for ${u.username}`);
      return true;
    } catch {
      toast.error("Email not saved", "Enter a valid address like name@example.com.");
      return false;
    }
  }

  async function handleReset(u) {
    try {
      const { temp_password } = await resetUserPassword(u.id);
      setReveal({ label: `New temp password for ${u.username}`, value: temp_password });
      loadUsers();
    } catch (err) {
      setError(err.message || "Failed to reset password.");
    }
  }

  return (
    <div className="flex-1">
      <main className="p-6 max-w-6xl mx-auto space-y-6 w-full">
        <div>
          <h1 className="text-xl font-semibold text-white">Manage Users</h1>
          <p className="text-sm text-slate-500 mt-1">Create accounts, change roles, deactivate, reset passwords — no public signup</p>
        </div>
        {reveal && <RevealPassword label={reveal.label} value={reveal.value} onDismiss={() => setReveal(null)} />}
        {error && <p className="text-sm text-rose-400">{error}</p>}

        <form onSubmit={handleCreate} className="bg-slate-900 border border-slate-800 rounded-lg p-5 space-y-3">
          <h2 className="text-sm font-medium text-slate-300 mb-2">Create account</h2>
          <div className="grid sm:grid-cols-3 gap-3">
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Username"
              required
              minLength={3}
              className="rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <input
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email (for password reset)"
              type="email"
              className="rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <select
              value={newRole}
              onChange={(e) => setNewRole(e.target.value)}
              className="rounded-md bg-slate-800 border border-slate-700 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {ROLES.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            disabled={creating}
            className="text-sm font-medium px-4 py-2 rounded-md bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white transition-colors"
          >
            {creating ? "Creating…" : "Create account"}
          </button>
        </form>

        <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-xs text-slate-500 uppercase">
                <tr>
                  <th className="text-left px-4 py-2">Username</th>
                  <th className="text-left px-4 py-2">Email</th>
                  <th className="text-left px-4 py-2">Role</th>
                  <th className="text-left px-4 py-2">Status</th>
                  <th className="text-left px-4 py-2">Last login</th>
                  <th className="text-right px-4 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id} className="border-t border-slate-800">
                    <td className="px-4 py-2 text-slate-200">{u.username}</td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <EmailCell user={u} onSave={handleEmailSave} />
                    </td>
                    <td className="px-4 py-2">
                      <select
                        value={u.role}
                        onChange={(e) => handleRoleChange(u, e.target.value)}
                        className="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-xs text-white"
                      >
                        {ROLES.map((r) => (
                          <option key={r} value={r}>
                            {r}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="px-4 py-2">
                      <span className={`text-xs px-2 py-0.5 rounded ${u.is_active ? "bg-emerald-500/15 text-emerald-400" : "bg-slate-700 text-slate-400"}`}>
                        {u.is_active ? "active" : "deactivated"}
                      </span>
                      {u.must_change_password && (
                        <span className="ml-2 text-xs text-amber-400">must change pw</span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-slate-400 whitespace-nowrap">
                      {u.last_login_at ? new Date(u.last_login_at).toLocaleString() : "never"}
                    </td>
                    <td className="px-4 py-2 text-right whitespace-nowrap">
                      <button
                        onClick={() => handleReset(u)}
                        className="text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 mr-1.5"
                      >
                        Reset password
                      </button>
                      <button
                        onClick={() => handleToggleActive(u)}
                        className="text-xs px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300"
                      >
                        {u.is_active ? "Deactivate" : "Reactivate"}
                      </button>
                    </td>
                  </tr>
                ))}
                {!loading && users.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-slate-500">
                      No users yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
