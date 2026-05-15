"""Send digest via Resend, or write to disk in dry-run / no-key mode."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import ensure_data_dir, get_settings

logger = logging.getLogger(__name__)


def _is_transient_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "connection", "network", "503", "502", "429", "rate limit"))


def _write_dry_run(html: str, subject: str) -> Path:
    ensure_data_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(get_settings().db_path).parent / f"digest-{stamp}.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Dry-run: wrote %s (subject=%r)", out, subject)
    return out


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_error),
    reraise=True,
)
def _send_with_retry(payload: dict) -> None:
    import resend
    resend.Emails.send(payload)


def send_digest(html: str, subject: str, dry_run: bool = False) -> bool:
    s = get_settings()
    if dry_run or not s.resend_api_key:
        _write_dry_run(html, subject)
        return False

    if not s.digest_to:
        logger.warning("digest_to is empty; writing dry-run instead.")
        _write_dry_run(html, subject)
        return False

    try:
        import resend

        resend.api_key = s.resend_api_key
        payload: dict[str, object] = {
            "from": s.digest_from,
            "to": [s.digest_to],
            "subject": subject,
            "html": html,
        }
        if s.reply_to_email:
            payload["reply_to"] = [s.reply_to_email]
        _send_with_retry(payload)
        logger.info("Sent digest %r to %s", subject, s.digest_to)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Resend send failed after retries (%s: %s); writing dry-run copy.",
                     type(e).__name__, str(e)[:200])
        _write_dry_run(html, subject)
        return False
