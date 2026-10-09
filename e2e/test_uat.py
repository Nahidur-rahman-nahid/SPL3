"""
FraudScope — end-to-end User Acceptance Tests.

The tests run in order and follow a real user's journey: first login with
the bootstrap password, navigation, scoring with presets, the six fraud
patterns, the live stream, the alert review queue (acknowledge, escalation),
user management, an analyst's restricted view, error panels, data cleanup,
the mobile layout and logout. Every step leaves a screenshot in
e2e/screenshots/.
"""

import re
import time

from playwright.sync_api import expect

from conftest import (
    ADMIN_BOOTSTRAP_PASSWORD,
    ADMIN_PASSWORD,
    ADMIN_USER,
    ANALYST_PASSWORD,
    ANALYST_USER,
    BASE_URL,
)

PRESETS = {
    # chip label (start)               expected decision
    "Normal Payment": "ALLOW",
    "Suspicious (structuring": "REVIEW",
    "High-Risk (mule hub": "BLOCK",
}

ANALYST_EMAIL = "analyst.one@example.com"
ANALYST_RESET_PASSWORD = "E2eReset#2026"

PATTERNS = ["Mule hub", "Structuring", "Smurfing fan-out", "Agent collusion", "Account takeover", "Mule network"]


def login(page, username, password):
    page.goto("/login")
    page.locator("input[autocomplete=username]").fill(username)
    page.locator("input[type=password]").fill(password)
    page.get_by_role("button", name="Sign in").click()


def change_password(page, current, new, confirm=None):
    fields = page.locator("input[type=password]")
    fields.nth(0).fill(current)
    fields.nth(1).fill(new)
    fields.nth(2).fill(confirm if confirm is not None else new)
    page.get_by_role("button", name="Change password").click()


def toast(page):
    return page.locator("[role=status], [role=alert]").last


# ---------------------------------------------------------------- login

def test_01_login_page_and_wrong_password(admin, shot):
    page = admin
    page.goto("/login")
    expect(page.get_by_role("heading", name="FraudScope")).to_be_visible()
    shot(page, "login_page")

    login(page, ADMIN_USER, "wrong-password")
    expect(page.get_by_text("Invalid username or password")).to_be_visible()
    shot(page, "login_error_wrong_password")


def test_02_first_login_forces_password_change(admin, shot):
    page = admin
    login(page, ADMIN_USER, ADMIN_BOOTSTRAP_PASSWORD)
    expect(page).to_have_url(f"{BASE_URL}/account")
    expect(page.get_by_text("temporary password")).to_be_visible()
    shot(page, "forced_password_change")

    change_password(page, ADMIN_BOOTSTRAP_PASSWORD, ADMIN_PASSWORD, confirm="does-not-match")
    expect(page.get_by_text("don't match")).to_be_visible()
    shot(page, "password_mismatch_error")

    change_password(page, ADMIN_BOOTSTRAP_PASSWORD, ADMIN_PASSWORD)
    expect(page.get_by_text("Password changed")).to_be_visible()
    shot(page, "password_changed_toast", full_page=False)
    page.get_by_role("button", name="Dismiss").click()  # keep later screenshots clean
    expect(page).to_have_url(f"{BASE_URL}/dashboard")


# --------------------------------------------------------------- navbar

def test_03_navbar_menus(admin, shot):
    page = admin
    nav = page.locator("nav")
    for link in ["Dashboard", "Score", "Fraud Patterns", "Alerts", "Live Stream"]:
        expect(nav.get_by_role("link", name=link)).to_be_visible()
    shot(page, "dashboard_empty")

    nav.get_by_role("button", name="Admin").click()
    expect(page.get_by_role("menuitem", name="Manage Users")).to_be_visible()
    expect(page.get_by_role("menuitem", name="Data Cleanup")).to_be_visible()
    shot(page, "navbar_admin_menu", full_page=False)
    page.keyboard.press("Escape")

    nav.get_by_role("button", name="Account menu").click()
    expect(page.get_by_role("menuitem", name="Log out")).to_be_visible()
    expect(page.get_by_text("Administrator")).to_be_visible()
    shot(page, "navbar_account_menu", full_page=False)
    page.keyboard.press("Escape")


# ---------------------------------------------------------------- score

def test_04_score_presets(admin, shot):
    page = admin
    page.locator("nav").get_by_role("link", name="Score").click()
    expect(page).to_have_url(f"{BASE_URL}/scoring")
    shot(page, "score_page")

    for chip, expected in PRESETS.items():
        button = page.get_by_role("button", name=re.compile(re.escape(chip)))
        button.click()
        expect(button).to_have_class(re.compile("bg-indigo-600"))  # preset chip is highlighted
        page.get_by_role("button", name="Score Transaction").click()
        result = page.locator("div.text-lg.font-semibold").filter(has_text=re.compile("^(ALLOW|REVIEW|BLOCK)"))
        expect(result).to_contain_text(expected, timeout=30_000)
        if expected == "ALLOW":  # no related transactions: the isolation note instead of a graph
            expect(page.get_by_text("Scored in isolation")).to_be_visible()
        else:  # graph with the centre transaction plus its related ones
            expect(page.locator("svg circle").first).to_be_visible()
            assert page.locator("svg circle").count() >= 3
        shot(page, f"score_preset_{expected.lower()}")


def test_05_score_new_account_id(admin, shot):
    page = admin
    page.get_by_role("button", name="New ID").first.click()
    sender = page.locator("input[type=text]").first
    expect(sender).to_have_value(re.compile(r"^C_NEW_"))
    # editing a field moves the form out of every preset
    assert page.locator("button.bg-indigo-600.rounded-full").count() == 0
    page.get_by_role("button", name="Score Transaction").click()
    expect(page.locator("div.text-lg.font-semibold").filter(has_text=re.compile("^(ALLOW|REVIEW|BLOCK)"))).to_be_visible()
    shot(page, "score_brand_new_account")


# ------------------------------------------------------- fraud patterns

def test_06_fraud_patterns(admin, shot):
    page = admin
    page.locator("nav").get_by_role("link", name="Fraud Patterns").click()
    expect(page.get_by_text("Model: PaySim + injected patterns")).to_be_visible()
    shot(page, "patterns_page")

    for title in PATTERNS:
        page.locator("main .grid > button").filter(has=page.get_by_text(title, exact=True)).click()
        columns = page.locator("[data-role=run-columns] > div")
        for i in (0, 1):  # fraud pattern, then its legitimate look-alike
            columns.nth(i).get_by_role("button", name=re.compile("^Run")).click()
            expect(columns.nth(i).get_by_text("flagged", exact=True)).to_be_visible(timeout=45_000)
        fraud, lookalike = (
            columns.nth(i).locator("xpath=.//p[normalize-space()='flagged']/preceding-sibling::p").inner_text()
            for i in (0, 1)
        )
        fraud_flagged, lookalike_flagged = int(fraud.split("/")[0]), int(lookalike.split("/")[0])
        assert fraud_flagged >= 3, f"{title}: fraud group flagged only {fraud}"
        assert lookalike_flagged == 0, f"{title}: look-alike flagged {lookalike}"
        shot(page, f"pattern_{title.lower().replace(' ', '_').replace('-', '_')}")


# ------------------------------------------------------------ dashboard

def test_07_dashboard_after_scoring(admin, shot):
    page = admin
    page.locator("nav").get_by_role("link", name="Dashboard").click()
    expect(page.get_by_text("Total Scored")).to_be_visible()
    expect(page.locator("tbody tr").first).to_be_visible()
    shot(page, "dashboard_with_feed")


# ---------------------------------------------------------- live stream

def test_08_live_stream(admin, shot):
    page = admin
    page.locator("nav").get_by_role("link", name="Live Stream").click()
    expect(page.get_by_text("Connected — receiving live decisions")).to_be_visible()
    page.get_by_role("button", name="Resume Stream").click()
    expect(page.get_by_role("button", name="Pause Stream")).to_be_visible()
    expect(page.locator(".animate-slide-in").nth(9)).to_be_visible(timeout=60_000)  # 10 live events
    shot(page, "live_stream_running")
    page.get_by_role("button", name="Pause Stream").click()
    expect(page.get_by_role("button", name="Resume Stream")).to_be_visible()


# ---------------------------------------------------------------- alerts

def test_09_alert_review_and_acknowledge(admin, shot):
    page = admin
    page.locator("nav").get_by_role("link", name="Alerts").click()
    rows = page.locator("button").filter(has=page.locator("span.text-amber-400", has_text="needs review"))
    expect(rows.first).to_be_visible()
    shot(page, "alerts_needs_review")

    before = rows.count()
    rows.first.click()
    expect(page.get_by_text("Linked to").first).to_be_visible()
    expect(page.get_by_text("the actual subgraph scored")).to_be_visible()  # graph beside the table
    expect(page.get_by_text(re.compile(r"[+\-≈] ?\d+\.\d{2,3}pp")).first).to_be_visible()  # 2-3 decimals
    shot(page, "alert_expanded_explanation")

    page.get_by_role("button", name="Acknowledge").first.click()
    expect(rows).to_have_count(before - 1)
    page.get_by_role("button", name="All Alerts").click()
    expect(page.get_by_text(re.compile(rf"acked · {ADMIN_USER}")).first).to_be_visible()
    shot(page, "alerts_all_after_acknowledge")


def test_10_alert_escalation(admin, shot):
    page = admin
    # SLA is 6 seconds in the E2E backend; the watcher checks every 2 seconds.
    page.get_by_role("button", name="Escalated").click()
    expect(page.get_by_text("SLA BREACHED").first).to_be_visible(timeout=30_000)
    shot(page, "alerts_escalated")


# ----------------------------------------------------------------- users

def test_11_admin_creates_and_manages_user(admin, shot, state):
    page = admin
    page.locator("nav").get_by_role("button", name="Admin").click()
    page.get_by_role("menuitem", name="Manage Users").click()
    expect(page.get_by_role("heading", name="Manage Users")).to_be_visible()

    page.get_by_placeholder("Username").fill(ANALYST_USER)
    page.get_by_placeholder("Email (for password reset)").fill("analyst1@example.com")
    page.get_by_role("button", name="Create account").click()
    reveal = page.locator("code.select-all")
    expect(reveal).to_be_visible()
    state["analyst_temp_password"] = reveal.inner_text().strip()
    row = page.locator("tr", has_text=ANALYST_USER)
    expect(row.get_by_text("must change pw")).to_be_visible()
    shot(page, "user_created_temp_password")

    row.get_by_role("button", name="Deactivate").click()
    expect(row.get_by_text("deactivated")).to_be_visible()
    shot(page, "user_deactivated")
    row.get_by_role("button", name="Reactivate").click()
    expect(row.get_by_text("active", exact=True)).to_be_visible()

    # change the email that password-reset links go to
    row.get_by_role("button", name=f"Edit email of {ANALYST_USER}").click()
    row.locator("input[type=email]").fill(ANALYST_EMAIL)
    row.get_by_role("button", name="Save").click()
    expect(page.get_by_text(f"Email saved for {ANALYST_USER}")).to_be_visible()
    expect(row.get_by_text(ANALYST_EMAIL)).to_be_visible()
    shot(page, "user_email_edited")


def test_12_analyst_has_restricted_view(browser, frontend, shot, state):
    context = browser.new_context(base_url=BASE_URL, viewport={"width": 1366, "height": 860})
    page = context.new_page()
    try:
        login(page, ANALYST_USER, state["analyst_temp_password"])
        expect(page).to_have_url(f"{BASE_URL}/account")
        change_password(page, state["analyst_temp_password"], ANALYST_PASSWORD)
        expect(page).to_have_url(f"{BASE_URL}/dashboard")

        expect(page.locator("nav").get_by_role("button", name="Admin")).to_have_count(0)
        shot(page, "analyst_dashboard_no_admin_menu")

        page.goto("/admin/users")
        expect(page.get_by_text("Admins only")).to_be_visible()
        shot(page, "analyst_blocked_from_admin_page")

        page.goto("/stream")
        expect(page.get_by_text("admin-only control")).to_be_visible()
        shot(page, "analyst_stream_read_only")
    finally:
        context.close()


# ------------------------------------------------------- forgot password

def reset_link_from_log(log_path, timeout=15):
    # No SMTP in the E2E backend, so the email (with its link) is logged.
    pattern = re.compile(re.escape(BASE_URL) + r"/reset-password\?token=[\w%-]+")
    deadline = time.time() + timeout
    while time.time() < deadline:
        links = pattern.findall(log_path.read_text(errors="ignore"))
        if links:
            return links[-1]
        time.sleep(0.5)
    raise AssertionError("no reset link appeared in the backend log")


def test_12b_forgot_password_by_email_link(browser, frontend, backend, shot):
    context = browser.new_context(base_url=BASE_URL, viewport={"width": 1366, "height": 860})
    page = context.new_page()
    try:
        page.goto("/login")
        page.get_by_role("link", name="Forgot password?").click()
        expect(page).to_have_url(f"{BASE_URL}/forgot-password")
        shot(page, "forgot_password_page")

        page.get_by_label("Email").fill(ANALYST_EMAIL)
        page.get_by_role("button", name="Send reset link").click()
        expect(page.get_by_text("a reset link is on its way")).to_be_visible()
        shot(page, "forgot_password_link_sent")

        link = reset_link_from_log(backend)
        page.goto(link)
        expect(page.get_by_role("heading", name="Choose a new password")).to_be_visible()
        page.get_by_label("New password", exact=True).fill(ANALYST_RESET_PASSWORD)
        page.get_by_label("Confirm new password").fill("something-else")
        page.get_by_role("button", name="Set new password").click()
        expect(page.get_by_text("don't match")).to_be_visible()
        shot(page, "reset_password_mismatch")

        page.get_by_label("Confirm new password").fill(ANALYST_RESET_PASSWORD)
        page.get_by_role("button", name="Set new password").click()
        expect(page.get_by_text("Your password has been changed")).to_be_visible()
        shot(page, "reset_password_success")

        # the link works only once
        page.goto(link)
        page.get_by_label("New password", exact=True).fill("Another#Pass1")
        page.get_by_label("Confirm new password").fill("Another#Pass1")
        page.get_by_role("button", name="Set new password").click()
        expect(page.get_by_text("invalid or has expired")).to_be_visible()
        shot(page, "reset_link_used_twice")

        login(page, ANALYST_USER, ANALYST_RESET_PASSWORD)
        expect(page).to_have_url(f"{BASE_URL}/dashboard")
    finally:
        context.close()


# ---------------------------------------------------------- error panels

def test_13_login_lockout_after_five_failures(browser, frontend, shot):
    context = browser.new_context(base_url=BASE_URL, viewport={"width": 1366, "height": 860})
    page = context.new_page()
    try:
        for _ in range(5):
            login(page, "intruder", "guess")
            expect(page.get_by_text("Invalid username or password")).to_be_visible()
        login(page, "intruder", "guess")
        expect(page.get_by_text("Too many failed attempts")).to_be_visible()
        shot(page, "login_locked_out")
    finally:
        context.close()


# ---------------------------------------------------------- data cleanup

def test_14_data_cleanup_confirm_dialog_and_toast(admin, shot):
    page = admin
    page.locator("nav").get_by_role("button", name="Admin").click()
    page.get_by_role("menuitem", name="Data Cleanup").click()
    page.get_by_text("Everything", exact=True).click()
    count = page.locator("span.text-lg.font-semibold")
    expect(count).not_to_have_text("0")
    total = count.inner_text()

    page.get_by_role("button", name="Delete permanently").click()
    dialog = page.get_by_role("alertdialog")
    expect(dialog).to_be_visible()
    shot(page, "cleanup_confirm_dialog", full_page=False)
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_have_count(0)
    expect(count).to_have_text(total)  # nothing was deleted

    page.get_by_role("button", name="Delete permanently").click()
    dialog.get_by_role("button", name="Delete permanently").click()
    expect(page.get_by_text(re.compile(r"^Deleted [\d,]+ transactions$"))).to_be_visible(timeout=30_000)
    expect(count).to_have_text("0")
    shot(page, "cleanup_success_toast", full_page=False)


# ---------------------------------------------------------------- mobile

def test_15_mobile_navigation(admin, shot):
    page = admin
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto("/dashboard")
    page.get_by_role("button", name="Open navigation menu").click()
    expect(page.locator("nav").get_by_role("link", name="Fraud Patterns").last).to_be_visible()
    shot(page, "mobile_menu", full_page=False)
    page.set_viewport_size({"width": 1366, "height": 860})


# ---------------------------------------------------------------- logout

def test_16_logout_ends_session(admin, shot):
    page = admin
    page.goto("/dashboard")
    page.locator("nav").get_by_role("button", name="Account menu").click()
    page.get_by_role("menuitem", name="Log out").click()
    expect(page).to_have_url(f"{BASE_URL}/login")
    page.goto("/dashboard")
    expect(page).to_have_url(f"{BASE_URL}/login")  # protected pages need a session again
    shot(page, "after_logout")
