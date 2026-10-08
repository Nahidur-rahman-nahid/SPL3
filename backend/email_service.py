"""Outgoing email over SMTP (settings.SMTP_*). With no SMTP_HOST configured
nothing is sent: the message is written to the server log instead, so the
forgot-password flow can be tried locally without a mail account."""

import logging
import smtplib
from email.message import EmailMessage

from config import settings

logger = logging.getLogger("email_service")


def is_configured() -> bool:
    return bool(settings.SMTP_HOST)


def send_email(to: str, subject: str, text: str, html: str | None = None) -> None:
    if not is_configured():
        logger.warning("SMTP not configured; email to %s not sent.\nSubject: %s\n%s", to, subject, text)
        return

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM or settings.SMTP_USERNAME
    message["To"] = to
    message["Subject"] = subject
    message.set_content(text)
    if html:
        message.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as smtp:
            if settings.SMTP_USE_TLS:
                smtp.starttls()
            if settings.SMTP_USERNAME:
                smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as e:
        # Runs as a background task after the response was sent, so the user
        # can't be told; log it for the operator.
        logger.error("Sending email to %s failed: %s", to, e)
