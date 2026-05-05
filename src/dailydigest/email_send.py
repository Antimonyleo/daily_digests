"""Send digest via Resend, or write to disk in dry-run / no-key mode."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import SETTINGS, ensure_data_dir

logger = logging.getLogger(__name__)


def _write_dry_run(html: str, subject: str) -> Path:
    ensure_data_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(SETTINGS.db_path).parent / f"digest-{stamp}.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Dry-run: wrote %s (subject=%r)", out, subject)
    return out


def send_digest(html: str, subject: str, dry_run: bool = False) -> bool:
    if dry_run or not SETTINGS.resend_api_key:
        _write_dry_run(html, subject)
        return False

    if not SETTINGS.digest_to:
        logger.warning("digest_to is empty; writing dry-run instead.")
        _write_dry_run(html, subject)
        return False

    try:
        import resend

        resend.api_key = SETTINGS.resend_api_key
        payload: dict[str, object] = {
            "from": SETTINGS.digest_from,
            "to": [SETTINGS.digest_to],
            "subject": subject,
            "html": html,
        }
        if SETTINGS.reply_to_email:
            payload["reply_to"] = [SETTINGS.reply_to_email]
        resend.Emails.send(payload)
        logger.info("Sent digest %r to %s", subject, SETTINGS.digest_to)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Resend send failed (%s); writing dry-run copy.", e)
        _write_dry_run(html, subject)
        return False
