"use client";

import { useEffect, useState } from "react";

// Light / dark switch. The theme lives on <html data-theme> (set before
// first paint by the script in app/layout.js); the choice is remembered in
// localStorage. Rendered in the navbar and on the signed-out pages.
const STORAGE_KEY = "theme";

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-[18px] h-[18px]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-[18px] h-[18px]" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" />
    </svg>
  );
}

export default function ThemeToggle({ className = "" }) {
  // null until mounted: the server can't know the visitor's theme, and the
  // icon must not mismatch what the pre-paint script already chose.
  const [theme, setTheme] = useState(null);

  useEffect(() => {
    setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  }, []);

  function toggle() {
    const next = theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // storage blocked (private mode): the switch still works for this visit
    }
    setTheme(next);
  }

  const label = theme === "light" ? "Switch to dark mode" : "Switch to light mode";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={label}
      title={label}
      className={`w-9 h-9 inline-flex items-center justify-center rounded-full text-slate-400 hover:text-white hover:bg-slate-800 transition-colors ${className}`}
    >
      {theme === null ? <span className="w-[18px] h-[18px]" /> : theme === "light" ? <MoonIcon /> : <SunIcon />}
    </button>
  );
}
