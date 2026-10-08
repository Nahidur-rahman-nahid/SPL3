/** @type {import('next').NextConfig} */
const BACKEND = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const nextConfig = {
  // The end-to-end tests (e2e/) build into their own folder so they never
  // touch the .next folder a running dev server is using.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  // Proxy all backend calls through Next.js so the browser sees everything
  // as same-origin (localhost:3000). This makes SameSite=lax cookies work
  // without needing HTTPS or SameSite=none.
  async rewrites() {
    return [
      { source: "/auth/:path*", destination: `${BACKEND}/auth/:path*` },
      { source: "/score",       destination: `${BACKEND}/score` },
      { source: "/api/:path*",  destination: `${BACKEND}/api/:path*` },
      { source: "/ws/:path*",   destination: `${BACKEND}/ws/:path*` },
    ];
  },
};

export default nextConfig;
