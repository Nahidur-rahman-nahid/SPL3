"""
Light / dark theme: follows the system setting at first, the toggle
switches every page, and the choice survives a reload and a new page.
Screenshots of the main pages in light mode go to e2e/screenshots/.
"""

import re

from playwright.sync_api import expect

from conftest import ADMIN_BOOTSTRAP_PASSWORD, ADMIN_PASSWORD, ADMIN_USER, BASE_URL


def theme(page):
    return page.evaluate("document.documentElement.dataset.theme")


def background_lightness(page):
    # average of the body's background RGB channels: ~2 dark, ~240 light
    page.wait_for_timeout(400)  # let the 0.2 s colour transition finish
    return page.evaluate(
        "(() => { const m = getComputedStyle(document.body).backgroundColor.match(/\\d+/g).map(Number);"
        " return (m[0] + m[1] + m[2]) / 3; })()"
    )


def sign_in_as_admin(page):
    """Works alone (first login, bootstrap password) or after test_uat.py."""
    for password in (ADMIN_PASSWORD, ADMIN_BOOTSTRAP_PASSWORD):
        page.goto("/login")
        page.locator("input[autocomplete=username]").fill(ADMIN_USER)
        page.locator("input[type=password]").fill(password)
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_url(re.compile(r"/(dashboard|account|login)$"))
        page.wait_for_timeout(500)
        if not page.url.endswith("/login"):
            break
    if page.url.endswith("/account"):  # first login: set the password the other tests use
        fields = page.locator("input[type=password]")
        fields.nth(0).fill(ADMIN_BOOTSTRAP_PASSWORD)
        fields.nth(1).fill(ADMIN_PASSWORD)
        fields.nth(2).fill(ADMIN_PASSWORD)
        page.get_by_role("button", name="Change password").click()
    expect(page).to_have_url(f"{BASE_URL}/dashboard")


def test_theme_toggle(browser, frontend, shot):
    context = browser.new_context(base_url=BASE_URL, viewport={"width": 1366, "height": 860}, color_scheme="dark")
    page = context.new_page()
    page.set_default_timeout(20_000)
    try:
        # first visit follows the system setting (dark here)
        page.goto("/login")
        assert theme(page) == "dark" and background_lightness(page) < 40

        page.get_by_role("button", name="Switch to light mode").click()
        assert theme(page) == "light" and background_lightness(page) > 200
        shot(page, "light_login")

        page.reload()  # remembered, and applied before the page paints
        assert theme(page) == "light"

        sign_in_as_admin(page)
        assert theme(page) == "light"
        shot(page, "light_dashboard")

        page.goto("/scoring")
        page.get_by_role("button", name=re.compile(r"^High-Risk")).click()
        page.get_by_role("button", name="Score Transaction").click()
        expect(page.get_by_text("the actual subgraph scored")).to_be_visible(timeout=30_000)
        shot(page, "light_score_result")

        page.goto("/patterns")
        page.locator("main .grid > button").filter(has=page.get_by_text("Mule network", exact=True)).click()
        columns = page.locator("[data-role=run-columns] > div")
        for i in (0, 1):
            columns.nth(i).get_by_role("button", name=re.compile("^Run")).click()
            expect(columns.nth(i).get_by_text("flagged", exact=True)).to_be_visible(timeout=45_000)
        shot(page, "light_fraud_pattern_graph")

        page.goto("/alerts")
        page.get_by_role("button", name="All Alerts").click()
        rows = page.locator("main button").filter(has=page.locator("span.font-mono"))
        if rows.count():
            rows.first.click()
        shot(page, "light_alerts")

        page.goto("/admin/users")
        expect(page.get_by_role("heading", name="Manage Users")).to_be_visible()
        shot(page, "light_manage_users")

        # back to dark from the navbar
        page.locator("nav").get_by_role("button", name="Switch to dark mode").click()
        assert theme(page) == "dark" and background_lightness(page) < 40
        page.goto("/dashboard")
        assert theme(page) == "dark"
        shot(page, "dark_dashboard_after_toggle")
    finally:
        context.close()
