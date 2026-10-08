"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { getStats } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import ThemeToggle from "@/components/ThemeToggle";

// The one navbar for every signed-in page (rendered by the (protected)
// layout). Main pages in the middle, admin-only tools grouped under an
// "Admin" menu, and the signed-in user / change password / log out in a
// single account menu on the right — so the username and role each appear
// exactly once.

const MAIN_LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/scoring", label: "Score" },
  { href: "/patterns", label: "Fraud Patterns" },
  { href: "/alerts", label: "Alerts", badge: true },
  { href: "/stream", label: "Live Stream" },
];

const ADMIN_LINKS = [
  { href: "/admin/users", label: "Manage Users" },
  { href: "/admin/data", label: "Data Cleanup" },
];

const ROLE_LABELS = { ADMIN: "Administrator", ANALYST: "Analyst" };

// The badge is a convenience, not live-critical — a light poll keeps it
// current on every page without each page opening its own WebSocket.
const BADGE_POLL_MS = 15000;

function isActive(pathname, href) {
  return pathname === href || pathname.startsWith(`${href}/`);
}

// Closes a dropdown on outside click or Escape.
function useDismiss(open, onClose) {
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return;
    function onPointerDown(e) {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    }
    function onKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);
  return ref;
}

function Chevron({ open }) {
  return (
    <svg viewBox="0 0 20 20" className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-180" : ""}`} fill="currentColor" aria-hidden="true">
      <path d="M5.3 7.3a1 1 0 011.4 0L10 10.6l3.3-3.3a1 1 0 111.4 1.4l-4 4a1 1 0 01-1.4 0l-4-4a1 1 0 010-1.4z" />
    </svg>
  );
}

function AlertBadge({ count }) {
  if (!count) return null;
  return (
    <span className="ml-1.5 min-w-[1.25rem] h-5 px-1.5 inline-flex items-center justify-center rounded-full bg-rose-500 text-white text-[11px] font-semibold tabular-nums">
      {count > 99 ? "99+" : count}
    </span>
  );
}

// Active item: soft pill background plus a short accent bar underneath.
function navLinkClass(active) {
  return `relative inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
    active ? "bg-slate-800/70 text-white" : "text-slate-300 hover:text-white hover:bg-slate-800/40"
  }`;
}

function ActiveBar({ active }) {
  if (!active) return null;
  return <span className="absolute left-1/2 -translate-x-1/2 bottom-0.5 h-0.5 w-6 rounded-full bg-indigo-500" aria-hidden="true" />;
}

function menuItemClass(active) {
  return `block px-3 py-2 rounded-md text-sm transition-colors ${
    active ? "bg-slate-800 text-white" : "text-slate-300 hover:bg-slate-800 hover:text-white"
  }`;
}

export default function AppNavbar() {
  const { user, role, logout } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const [pendingAlerts, setPendingAlerts] = useState(0);
  const [adminOpen, setAdminOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const closeAdmin = useCallback(() => setAdminOpen(false), []);
  const closeAccount = useCallback(() => setAccountOpen(false), []);
  const adminRef = useDismiss(adminOpen, closeAdmin);
  const accountRef = useDismiss(accountOpen, closeAccount);

  const isAdmin = role === "ADMIN";
  const initial = (user?.username || "?").charAt(0).toUpperCase();
  const adminActive = ADMIN_LINKS.some((l) => isActive(pathname, l.href));

  const loadBadge = useCallback(async () => {
    try {
      const stats = await getStats();
      setPendingAlerts(stats.pending_alerts ?? 0);
    } catch {
      // badge just keeps its last value; the next poll retries
    }
  }, []);

  useEffect(() => {
    loadBadge();
    const interval = setInterval(loadBadge, BADGE_POLL_MS);
    return () => clearInterval(interval);
  }, [loadBadge]);

  // Every navigation: close open menus and refresh the badge (e.g. after
  // acknowledging alerts on the Alerts page).
  useEffect(() => {
    setAdminOpen(false);
    setAccountOpen(false);
    setMobileOpen(false);
    loadBadge();
  }, [pathname, loadBadge]);

  async function handleLogout() {
    await logout();
    router.replace("/login");
  }

  return (
    <nav className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
      {/* Three columns: logo | centred links | account — the outer columns
          are equal width (1fr each), so the links sit in the true centre. */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 grid grid-cols-[1fr_auto_1fr] items-center h-16">
        <Link href="/dashboard" className="justify-self-start flex items-center gap-2 text-white">
          <Image src="/fraudscope-icon.svg" alt="" width={32} height={32} priority />
          <span className="font-semibold tracking-tight">FraudScope</span>
        </Link>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-1">
          {MAIN_LINKS.map((l) => {
            const active = isActive(pathname, l.href);
            return (
              <Link key={l.href} href={l.href} className={navLinkClass(active)}>
                {l.label}
                {l.badge && <AlertBadge count={pendingAlerts} />}
                <ActiveBar active={active} />
              </Link>
            );
          })}

          {isAdmin && (
            <div ref={adminRef} className="relative">
              <button
                onClick={() => setAdminOpen((o) => !o)}
                aria-expanded={adminOpen}
                aria-haspopup="menu"
                className={`${navLinkClass(adminActive)} gap-1`}
              >
                Admin <Chevron open={adminOpen} />
                <ActiveBar active={adminActive} />
              </button>
              {adminOpen && (
                <div role="menu" className="absolute left-1/2 -translate-x-1/2 top-full mt-2 w-48 rounded-lg border border-slate-800 bg-slate-900 p-1 shadow-xl">
                  {ADMIN_LINKS.map((l) => (
                    <Link key={l.href} href={l.href} role="menuitem" className={menuItemClass(isActive(pathname, l.href))}>
                      {l.label}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="col-start-3 justify-self-end flex items-center gap-2">
          <ThemeToggle />
          {/* Account menu */}
          <div ref={accountRef} className="relative">
            <button
              onClick={() => setAccountOpen((o) => !o)}
              aria-expanded={accountOpen}
              aria-haspopup="menu"
              aria-label="Account menu"
              className="flex items-center gap-1.5 rounded-full p-0.5 pr-1.5 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            >
              <span className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-violet-600 text-white text-sm font-semibold flex items-center justify-center">
                {initial}
              </span>
              <Chevron open={accountOpen} />
            </button>
            {accountOpen && (
              <div role="menu" className="absolute right-0 top-full mt-2 w-60 rounded-lg border border-slate-800 bg-slate-900 p-1 shadow-xl">
                <div className="px-3 py-2.5">
                  <p className="text-sm font-medium text-white truncate">{user?.username}</p>
                  <p className="text-xs text-slate-500">{ROLE_LABELS[role] || role}</p>
                </div>
                <div className="my-1 border-t border-slate-800" />
                <Link href="/account" role="menuitem" className={menuItemClass(pathname === "/account")}>
                  Change password
                </Link>
                <div className="my-1 border-t border-slate-800" />
                <button
                  onClick={handleLogout}
                  role="menuitem"
                  className="w-full text-left px-3 py-2 rounded-md text-sm text-rose-400 hover:bg-rose-500/10 transition-colors"
                >
                  Log out
                </button>
              </div>
            )}
          </div>

          {/* Mobile menu toggle */}
          <button
            onClick={() => setMobileOpen((o) => !o)}
            aria-expanded={mobileOpen}
            aria-label="Open navigation menu"
            className="md:hidden p-2 rounded-md text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              {mobileOpen ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile links */}
      {mobileOpen && (
        <div className="md:hidden border-t border-slate-800 px-4 py-3 space-y-1">
          {MAIN_LINKS.map((l) => (
            <Link key={l.href} href={l.href} className={`${menuItemClass(isActive(pathname, l.href))} flex items-center`}>
              {l.label}
              {l.badge && <AlertBadge count={pendingAlerts} />}
            </Link>
          ))}
          {isAdmin && (
            <>
              <p className="px-3 pt-3 pb-1 text-[11px] uppercase tracking-wider text-slate-500">Admin</p>
              {ADMIN_LINKS.map((l) => (
                <Link key={l.href} href={l.href} className={menuItemClass(isActive(pathname, l.href))}>
                  {l.label}
                </Link>
              ))}
            </>
          )}
        </div>
      )}
    </nav>
  );
}
