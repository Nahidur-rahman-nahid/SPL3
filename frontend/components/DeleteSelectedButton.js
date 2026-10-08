"use client";

import { useState } from "react";
import { deleteTransactions } from "@/lib/api";
import { useConfirm, useToast } from "@/components/Feedback";

// Admin-only bulk delete for the row checkboxes in LiveFeed / AlertQueue.
// Hard delete on the backend (transaction + decisions + alerts), so it asks
// for confirmation first.
export default function DeleteSelectedButton({ selectedIds, onDeleted }) {
  const [deleting, setDeleting] = useState(false);
  const confirm = useConfirm();
  const toast = useToast();
  const count = selectedIds.length;

  async function handleDelete() {
    const noun = count === 1 ? "transaction" : "transactions";
    const ok = await confirm({
      title: `Delete ${count} ${noun}?`,
      message: "Their decisions and alerts are deleted too. This cannot be undone.",
      confirmLabel: "Delete",
    });
    if (!ok) return;
    setDeleting(true);
    try {
      const result = await deleteTransactions(selectedIds);
      onDeleted(selectedIds);
      toast.success(
        `Deleted ${result.deleted_transactions} ${result.deleted_transactions === 1 ? "transaction" : "transactions"}`,
        `${result.deleted_decisions} decisions and ${result.deleted_alerts} alerts removed.`
      );
    } catch (err) {
      toast.error("Delete failed", err.message || "Please try again.");
    } finally {
      setDeleting(false);
    }
  }

  if (count === 0) return null;

  return (
    <span className="flex items-center gap-2">
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
