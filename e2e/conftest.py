"""
End-to-end User Acceptance Tests (UAT) for FraudScope, driven with
Playwright (Chromium) through the real UI.

What this file sets up, once per test session, fully isolated from the real
system (the production PostgreSQL / Upstash Redis in backend/.env are never
touched; every setting below overrides .env):
  - an in-process fake Redis server (fakeredis) on 127.0.0.1:6390
  - the FastAPI backend on 127.0.0.1:8001 with a throwaway SQLite database
    and the real trained GNN model (backend/model/)
  - a production build of the Next.js frontend in frontend/.next-e2e,
    served on http://localhost:3001 (your own dev server's .next folder is
    left alone)

Screenshots of every step are saved to e2e/screenshots/ (cleared at the
start of each run), numbered in the order they were taken.

Run (from the repo root):
    pip install -r e2e/requirements.txt
    python -m playwright install chromium
    python -m pytest e2e                     # headless
    python -m pytest e2e --headed --slowmo 300   # watch it in a real window
    E2E_SKIP_BUILD=1 python -m pytest e2e    # reuse the last frontend build
"""

import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"

REDIS_PORT = 6390
BACKEND_PORT = 8001
FRONTEND_PORT = 3001
BASE_URL = f"http://localhost:{FRONTEND_PORT}"
DIST_DIR = ".next-e2e"

ADMIN_USER = "admin"
ADMIN_BOOTSTRAP_PASSWORD = "E2eBootstrap#2026"
ADMIN_PASSWORD = "E2eAdmin#2026"
ANALYST_USER = "analyst1"
ANALYST_PASSWORD = "E2eAnalyst#2026"

IS_WINDOWS = os.name == "nt"
NPX = "npx.cmd" if IS_WINDOWS else "npx"


def _wait_for(url: str, timeout: float, what: str, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as res:
                if res.status < 500:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"{what} did not come up at {url} within {timeout:.0f}s")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if IS_WINDOWS:  # npx/uvicorn spawn children; kill the whole tree
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def fake_redis():
    from fakeredis import TcpFakeServer

    server = TcpFakeServer(("127.0.0.1", REDIS_PORT), server_type="redis")
    server.handle_error = lambda request, client_address: None  # backend killed at teardown resets sockets
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield
    server.shutdown()


@pytest.fixture(scope="session")
def backend(fake_redis, tmp_path_factory):
    db_path = tmp_path_factory.mktemp("e2e") / "e2e.db"
    log_path = tmp_path_factory.getbasetemp() / "backend.log"
    log = open(log_path, "w")
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
        "REDIS_URL": f"redis://127.0.0.1:{REDIS_PORT}/0",
        "SECRET_KEY": "e2e-only-secret",
        "BOOTSTRAP_ADMIN_USERNAME": ADMIN_USER,
        "BOOTSTRAP_ADMIN_PASSWORD": ADMIN_BOOTSTRAP_PASSWORD,
        "CORS_ORIGINS": BASE_URL,
        "COOKIE_SECURE": "False",
        "COOKIE_SAMESITE": "lax",
        "STREAM_ENABLED": "False",
        "STREAM_INTERVAL_SECONDS": "0.5",
        # Short SLA so the escalation path can be seen within the test run.
        "ALERT_SLA_MINUTES": "0.1",
        "ALERT_ESCALATION_CHECK_SECONDS": "2",
        # Never send real email from tests: with no SMTP host the reset link
        # is written to the backend log, where the test picks it up.
        "SMTP_HOST": "",
        "FRONTEND_BASE_URL": BASE_URL,
        "PYTHONUNBUFFERED": "1",
    }
    proc = subprocess.Popen(
        ["python", "-m", "uvicorn", "main:app", "--port", str(BACKEND_PORT)],
        cwd=BACKEND_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        _wait_for(f"http://127.0.0.1:{BACKEND_PORT}/health", 90, "backend", proc)
        yield log_path
    finally:
        _stop(proc)
        log.close()


@pytest.fixture(scope="session")
def frontend(backend, tmp_path_factory):
    env = {
        **os.environ,
        "NEXT_DIST_DIR": DIST_DIR,
        "NEXT_PUBLIC_API_BASE_URL": f"http://127.0.0.1:{BACKEND_PORT}",
        "NEXT_TELEMETRY_DISABLED": "1",
    }
    if not (os.environ.get("E2E_SKIP_BUILD") and (FRONTEND_DIR / DIST_DIR).exists()):
        build = subprocess.run([NPX, "next", "build"], cwd=FRONTEND_DIR, env=env, capture_output=True, text=True)
        if build.returncode != 0:
            raise RuntimeError("frontend build failed:\n" + build.stdout[-3000:] + build.stderr[-3000:])
    log = open(tmp_path_factory.getbasetemp() / "frontend.log", "w")
    proc = subprocess.Popen(
        [NPX, "next", "start", "-p", str(FRONTEND_PORT)],
        cwd=FRONTEND_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        _wait_for(f"{BASE_URL}/login", 90, "frontend", proc)
        yield BASE_URL
    finally:
        _stop(proc)
        log.close()


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {**browser_context_args, "base_url": BASE_URL, "viewport": {"width": 1366, "height": 860}}


class Shots:
    """Numbered, full-page screenshots in e2e/screenshots/."""

    def __init__(self):
        # Clear the previous run's images but keep the folder (Windows can't
        # delete a folder an image viewer or editor still has open).
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        for old in SCREENSHOT_DIR.glob("*.png"):
            old.unlink(missing_ok=True)
        self.n = 0

    def __call__(self, page, name: str, full_page: bool = True) -> Path:
        self.n += 1
        path = SCREENSHOT_DIR / f"{self.n:02d}_{name}.png"
        page.screenshot(path=str(path), full_page=full_page)
        return path


@pytest.fixture(scope="session")
def shot():
    return Shots()


@pytest.fixture(scope="module")
def admin(browser, frontend):
    """One signed-in admin browser session shared by the tests of a module,
    in order — the way a real user moves through the app."""
    context = browser.new_context(base_url=BASE_URL, viewport={"width": 1366, "height": 860})
    page = context.new_page()
    page.set_default_timeout(20_000)
    yield page
    context.close()


@pytest.fixture(scope="session")
def state():
    """Values handed from one test to the next (e.g. a temp password)."""
    return {}


# Keep the temp dir path visible when a run fails, so the server logs can be read.
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if exitstatus != 0:
        terminalreporter.write_line(f"Server logs: {Path(tempfile.gettempdir())} (pytest-of-*/.../backend.log, frontend.log)")
    terminalreporter.write_line(f"Screenshots: {SCREENSHOT_DIR}")
