"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";

// In-app replacements for window.confirm / window.alert: a styled
// confirmation dialog (useConfirm, promise-based) and toast notifications
// (useToast). Mounted once in the protected layout.

const FeedbackContext = createContext(null);

const TOAST_MS = 4500;

const TOAST_STYLES = {
  success: { ring: "border-emerald-500/40", icon: "text-emerald-400", path: "M5 13l4 4L19 7" },
  error: { ring: "border-rose-500/40", icon: "text-rose-400", path: "M6 18L18 6M6 6l12 12" },
  info: { ring: "border-indigo-500/40", icon: "text-indigo-400", path: "M13 16h-1v-4h-1m1-4h.01" },
};

function Toasts({ toasts, dismiss }) {
  return (
    <div className="fixed bottom-4 right-4 z-[60] flex flex-col gap-2 w-[calc(100%-2rem)] sm:w-96" aria-live="polite">
      {toasts.map((t) => {
        const s = TOAST_STYLES[t.type] || TOAST_STYLES.info;
        return (
          <div
            key={t.id}
            role={t.type === "error" ? "alert" : "status"}
            className={`toast-in flex items-start gap-3 rounded-lg border ${s.ring} bg-slate-900/95 backdrop-blur px-4 py-3 shadow-xl shadow-black/40`}
          >
            <svg className={`w-5 h-5 shrink-0 mt-0.5 ${s.icon}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d={s.path} />
            </svg>
            <div className="flex-1 min-w-0">
              {t.title && <p className="text-sm font-medium text-white">{t.title}</p>}
              {t.message && <p className="text-xs text-slate-400 mt-0.5">{t.message}</p>}
            </div>
            <button
              onClick={() => dismiss(t.id)}
              className="text-slate-500 hover:text-slate-300 text-lg leading-none"
              aria-label="Dismiss"
            >
              ×
            </button>
          </div>
        );
      })}
    </div>
  );
}

function ConfirmDialog({ request, onClose }) {
  const cancelRef = useRef(null);

  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (e) => e.key === "Escape" && onClose(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const danger = request.tone !== "default";
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => onClose(false)} />
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        className="dialog-in relative w-full max-w-md rounded-xl border border-slate-800 bg-slate-900 shadow-2xl shadow-black/50"
      >
        <div className="p-5 flex gap-4">
          {danger && (
            <div className="shrink-0 w-10 h-10 rounded-full bg-rose-500/15 flex items-center justify-center">
              <svg className="w-5 h-5 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m0 3.75h.008M10.3 3.86L1.82 18a1.5 1.5 0 001.29 2.25h16.96A1.5 1.5 0 0021.37 18L12.9 3.86a1.5 1.5 0 00-2.6 0z" />
              </svg>
            </div>
          )}
          <div className="min-w-0">
            <h2 id="confirm-title" className="text-base font-semibold text-white">{request.title}</h2>
            {request.message && <p className="text-sm text-slate-400 mt-1.5 leading-relaxed">{request.message}</p>}
          </div>
        </div>
        <div className="flex justify-end gap-2 px-5 py-3 border-t border-slate-800 bg-slate-950/40 rounded-b-xl">
          <button
            ref={cancelRef}
            onClick={() => onClose(false)}
            className="rounded-md px-4 py-2 text-sm text-slate-300 hover:bg-slate-800 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onClose(true)}
            className={`rounded-md px-4 py-2 text-sm font-medium text-white transition-colors ${
              danger ? "bg-rose-600 hover:bg-rose-500" : "bg-indigo-600 hover:bg-indigo-500"
            }`}
          >
            {request.confirmLabel || "Confirm"}
          </button>
        </div>
      </div>
    </div>
  );
}

export function FeedbackProvider({ children }) {
  const [toasts, setToasts] = useState([]);
  const [confirmRequest, setConfirmRequest] = useState(null);
  const nextId = useRef(0);

  const dismiss = useCallback((id) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);

  const toast = useCallback(
    (type, title, message) => {
      const id = ++nextId.current;
      setToasts((ts) => [...ts.slice(-3), { id, type, title, message }]);
      setTimeout(() => dismiss(id), TOAST_MS);
    },
    [dismiss]
  );

  // confirm({title, message, confirmLabel, tone}) -> Promise<boolean>
  const confirm = useCallback(
    (options) => new Promise((resolve) => setConfirmRequest({ ...options, resolve })),
    []
  );

  const closeConfirm = useCallback(
    (ok) => {
      setConfirmRequest((req) => {
        req?.resolve(ok);
        return null;
      });
    },
    []
  );

  return (
    <FeedbackContext.Provider value={{ toast, confirm }}>
      {children}
      <Toasts toasts={toasts} dismiss={dismiss} />
      {confirmRequest && <ConfirmDialog request={confirmRequest} onClose={closeConfirm} />}
    </FeedbackContext.Provider>
  );
}

export function useToast() {
  const { toast } = useContext(FeedbackContext);
  return {
    success: (title, message) => toast("success", title, message),
    error: (title, message) => toast("error", title, message),
    info: (title, message) => toast("info", title, message),
  };
}

export function useConfirm() {
  return useContext(FeedbackContext).confirm;
}
