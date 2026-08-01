"""
SLA-based escalation for the review queue (spec item 4's "escalation
beyond WebSocket broadcast"). A background loop — started at app startup
alongside stream_simulator.run_stream() — scans for Alerts that have sat
unacknowledged past ALERT_SLA_MINUTES, marks them escalated, logs a
warning (a real, always-on audit-log side channel independent of whether
any analyst has a dashboard open), and broadcasts a distinct
"alert_escalated" WebSocket event so it renders differently from a fresh
alert on /alerts.

No SMTP/paging-provider credentials are configured for this project, so
real email/SMS/PagerDuty notification isn't wired up here — notify_
escalation() is the integration point for that: swap its body for a real
call once credentials exist, and nothing else in this file changes. Same
"documented stand-in, clean swap path" pattern stream_simulator.py uses
for Kafka.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import models
from config import settings
from database import SessionLocal
from websocket_manager import manager

logger = logging.getLogger("escalation")


def notify_escalation(alert: models.Alert) -> None:
    logger.warning(
        "ALERT ESCALATED (unacknowledged past %.1f min SLA): alert_id=%s decision_id=%s",
        settings.ALERT_SLA_MINUTES, alert.alert_id, alert.decision_id,
    )


def _is_overdue(alert: models.Alert, now: datetime, sla: timedelta) -> bool:
    pushed_at = alert.pushed_at
    if pushed_at.tzinfo is None:  # SQLite round-trips naive UTC; Postgres may too depending on column type
        pushed_at = pushed_at.replace(tzinfo=timezone.utc)
    return (now - pushed_at) >= sla


async def run_escalation_watch():
    logger.info(
        "Alert escalation watcher started (sla=%.1fmin, check every %.0fs)",
        settings.ALERT_SLA_MINUTES, settings.ALERT_ESCALATION_CHECK_SECONDS,
    )
    while True:
        await asyncio.sleep(settings.ALERT_ESCALATION_CHECK_SECONDS)
        db = SessionLocal()
        try:
            now = datetime.now(timezone.utc)
            sla = timedelta(minutes=settings.ALERT_SLA_MINUTES)
            candidates = (
                db.query(models.Alert)
                .filter(models.Alert.acknowledged_at.is_(None), models.Alert.escalated.is_(False))
                .all()
            )
            overdue = [a for a in candidates if _is_overdue(a, now, sla)]

            for alert in overdue:
                alert.escalated = True
                alert.escalated_at = now
                notify_escalation(alert)
            if overdue:
                db.commit()
                for alert in overdue:
                    await manager.broadcast({
                        "type": "alert_escalated",
                        "alert_id": alert.alert_id,
                        "decision_id": alert.decision_id,
                        "escalated_at": alert.escalated_at.isoformat(),
                    })
        except Exception as e:
            logger.warning("Escalation watch skipped a cycle due to an error: %s", e)
        finally:
            db.close()
