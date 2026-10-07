"use client";

import { useState } from "react";
import { deleteTransactions } from "@/lib/api";

// Admin-only bulk delete for the row checkboxes in LiveFeed / AlertQueue.
// Hard delete on the backend (transaction + decisions + alerts), so it asks
// for confirmation first.
export default function DeleteSelectedButton({ selectedIds, onDeleted }) {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  const count = selectedIds.length;

  async function handleDelete() {
    const noun = count === 1 ? "transaction" : "transactions";
    if (!window.confirm(`Permanently delete ${count} ${noun}, with their decisions and alerts? This cannot be undone.`)) {
      return;
    }
    setDeleting(true);
    setError("");
    try {
      await deleteTransactions(selectedIds);
      onDeleted(selectedIds);
    } catch (err) {
      setError(err.message || "Delete failed.");
    } finally {
      setDeleting(false);
    }
  }

  if (count === 0) return null;

  return (
    <span className="flex items-center gap-2">
      {error && <span className="text-xs text-rose-400">{error}</span>}
      <button
        onClick={handleDelete}
        disabled={deleting}
        className="text-xs px-3 py-1.5 rounded-md bg-rose-600 hover:bg-rose-500 text-white disabled:opacity-60 transition-colors"
      >
        {deleting ? "Deleting…" : `Delete selected (${count})`}
      </button>
    </span>
  );
}
