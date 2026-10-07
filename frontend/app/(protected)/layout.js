"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import AppNavbar from "@/components/AppNavbar";

// This redirect is a UX convenience only, not the security boundary — every
// protected backend endpoint enforces auth/RBAC itself (see backend/auth.py's
// get_current_user / require_role). A user who bypasses this layout still
// can't reach real data without a valid session cookie.
export default function ProtectedLayout({ children }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  const mustChangePassword = user?.must_change_password && pathname !== "/account";

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace("/login");
      return;
    }
    if (mustChangePassword) {
      router.replace("/account");
    }
  }, [loading, user, mustChangePassword, router]);

  if (loading || !user || mustChangePassword) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-500 text-sm">
        Loading…
      </div>
    );
  }

  // No navbar while a temporary password must be changed — the account page
  // is the only place that user can go anyway.
  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      {!user.must_change_password && <AppNavbar />}
      {children}
    </div>
  );
}
