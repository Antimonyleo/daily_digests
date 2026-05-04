"""Inbound email reply ingestion (Phase 7a).

Polls an IMAP mailbox for unread replies to DailyDigest emails, parses
vote tokens like ``+R3 -I5`` from the user's authored top portion of the
reply body, and feeds them into :func:`votes.record_votes`. Idempotent
via UNSEEN search + ``\\Seen`` flagging.

No external deps: ``imaplib`` and ``email`` are stdlib.
"""

from __future__ import annotations

import email
import email.utils
import imaplib
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from email.header import decode_header, make_header
from email.message import Message
from typing import Any

from pydantic import BaseModel

from . import votes as votes_mod
from .config import SETTINGS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config


class InboundConfig(BaseModel):
    imap_host: str = ""
    imap_port: int = 993
    imap_user: str = ""
    imap_password: str = ""
    mailbox: str = "INBOX"
    since_days: int = 2
    subject_filter: str = "DailyDigest"


def load_inbound_config() -> InboundConfig:
    """Build an :class:`InboundConfig` from environment variables."""
    try:
        port = int(os.environ.get("IMAP_PORT", "993"))
    except ValueError:
        port = 993
    return InboundConfig(
        imap_host=os.environ.get("IMAP_HOST", ""),
        imap_port=port,
        imap_user=os.environ.get("IMAP_USER", ""),
        imap_password=os.environ.get("IMAP_PASSWORD", ""),
        mailbox=os.environ.get("IMAP_MAILBOX", "INBOX"),
    )


# ---------------------------------------------------------------------------
# Body parsing helpers

# Match a vote token possibly with a leading sign. Word-boundary on both ends.
_VOTE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[+-]?[RIGWriguw]\d+\b")
# Strip HTML tags as a last resort (only when no text/plain part exists).
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# "On Mon, Jan 1, 2026 at 8:00 AM Foo <foo@bar> wrote:" style markers.
_REPLY_MARKER_RE = re.compile(r"^\s*On .+ wrote:\s*$", re.IGNORECASE)
# Some clients use "From: ..." block; treat that as quoted as well.
_FROM_HEADER_RE = re.compile(r"^\s*From:\s.+", re.IGNORECASE)
# Trailing YYYY-MM-DD in subject.
_DATE_IN_SUBJECT_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _decode_header(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:  # noqa: BLE001
        return value


def _strip_html(html: str) -> str:
    text = _HTML_TAG_RE.sub("", html)
    # Decode a few common entities without pulling in html.parser machinery.
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return text


def _get_body(msg: Message) -> str:
    """Return best-effort plain-text body from an email message."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if "attachment" in disp:
                continue
            if ctype == "text/plain":
                plain_parts.append(_decode_payload(part))
            elif ctype == "text/html":
                html_parts.append(_decode_payload(part))
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            plain_parts.append(_decode_payload(msg))
        elif ctype == "text/html":
            html_parts.append(_decode_payload(msg))

    if plain_parts:
        return "\n".join(plain_parts)
    if html_parts:
        return _strip_html("\n".join(html_parts))
    return ""


def _decode_payload(part: Message) -> str:
    raw = part.get_payload(decode=True)
    if raw is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def _strip_quoted(body: str) -> str:
    """Drop quoted-reply lines, signatures, and 'On ... wrote:' tails."""
    lines = body.splitlines()
    kept: list[str] = []
    for line in lines:
        # Signature delimiter per RFC 3676: literal "-- " (dash dash space)
        # on its own line. ``splitlines`` already strips the trailing CRLF,
        # so we just need to compare to "-- ". A bare "--" (markdown
        # divider) does NOT trigger truncation.
        if line == "-- ":
            break
        stripped = line.strip()
        # Common reply marker.
        if _REPLY_MARKER_RE.match(line):
            break
        # "From:" block from forwarded/quoted reply.
        if _FROM_HEADER_RE.match(line):
            break
        # Quoted lines.
        if stripped.startswith(">"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def extract_vote_line(body: str) -> str | None:
    """Return a single space-joined string of vote tokens from body, or None.

    Walks lines top-to-bottom, collects every line containing at least one
    vote-like token. Joins them with single spaces. Returns ``None`` when
    no vote tokens are present.
    """
    if not body:
        return None
    collected: list[str] = []
    for line in body.splitlines():
        if _VOTE_TOKEN_RE.search(line):
            collected.append(line.strip())
    if not collected:
        return None
    joined = " ".join(collected)
    # Collapse whitespace runs.
    return re.sub(r"\s+", " ", joined).strip()


def _parse_digest_id(subject: str) -> str | None:
    m = _DATE_IN_SUBJECT_RE.search(subject or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# IMAP fetch


def _imap_search_criteria(cfg: InboundConfig) -> list[str]:
    since = (datetime.now(timezone.utc) - timedelta(days=cfg.since_days)).strftime(
        "%d-%b-%Y"
    )
    # ``imaplib.IMAP4.search`` quotes string arguments itself when needed.
    # Passing a pre-quoted value (e.g. ``'"DailyDigest"'``) makes the server
    # search for the literal string with the quotes embedded — zero matches.
    # Always pass the bare token here.
    return [
        "UNSEEN",
        "SINCE",
        since,
        "SUBJECT",
        cfg.subject_filter,
    ]


def fetch_replies(cfg: InboundConfig) -> list[dict[str, Any]]:
    """Connect via IMAP4_SSL, fetch matching unseen messages, mark them seen."""
    if not cfg.imap_host or not cfg.imap_user or not cfg.imap_password:
        return []

    results: list[dict[str, Any]] = []
    conn = imaplib.IMAP4_SSL(cfg.imap_host, cfg.imap_port)
    try:
        conn.login(cfg.imap_user, cfg.imap_password)
        conn.select(cfg.mailbox)

        criteria = _imap_search_criteria(cfg)
        typ, data = conn.search(None, *criteria)
        if typ != "OK" or not data or not data[0]:
            return []

        ids = data[0].split()
        for msg_id in ids:
            try:
                typ, msg_data = conn.fetch(msg_id, "(RFC822)")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue
                raw = msg_data[0][1]
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                msg = email.message_from_bytes(bytes(raw))

                subject = _decode_header(msg.get("Subject"))
                sender = _decode_header(msg.get("From"))
                date_hdr = _decode_header(msg.get("Date"))
                message_id = _decode_header(msg.get("Message-ID"))
                body = _strip_quoted(_get_body(msg))
                digest_id = _parse_digest_id(subject)

                results.append(
                    {
                        "message_id": message_id,
                        "from": sender,
                        "subject": subject,
                        "date": date_hdr,
                        "body": body,
                        "digest_id": digest_id,
                    }
                )

                # Mark as Seen so we don't reprocess.
                try:
                    conn.store(msg_id, "+FLAGS", "\\Seen")
                except Exception as e:  # noqa: BLE001
                    logger.warning("inbound: failed to mark %r seen: %s", msg_id, e)
            except Exception as e:  # noqa: BLE001
                logger.warning("inbound: failed to fetch message %r: %s", msg_id, e)
                continue
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            conn.logout()
        except Exception:  # noqa: BLE001
            pass

    return results


# ---------------------------------------------------------------------------
# Top-level entry


def process_replies() -> dict[str, Any]:
    """Fetch unread DailyDigest replies and record votes from each.

    Sender authentication: when ``SETTINGS.reply_to_email`` is set, only
    replies whose ``From:`` address matches it are processed. Mismatched
    senders are counted under ``skipped_unauthorized`` and logged at
    WARNING. When the setting is unset, a one-time startup warning is
    emitted and all senders are accepted (back-compat).
    """
    cfg = load_inbound_config()
    if not cfg.imap_user or not cfg.imap_password:
        return {"skipped": True}

    expected_sender = (SETTINGS.reply_to_email or "").strip().lower()
    if not expected_sender:
        logger.warning(
            "inbound: no REPLY_TO_EMAIL configured; accepting all senders (insecure)"
        )

    if expected_sender and cfg.imap_user.lower() != expected_sender:
        # IMAP_USER and REPLY_TO_EMAIL often coincide but don't have to
        # (e.g., catch-all addresses). Lowered to DEBUG: addresses are PII.
        logger.debug(
            "inbound: IMAP_USER differs from REPLY_TO_EMAIL; proceeding.",
        )

    summary: dict[str, Any] = {
        "messages": 0,
        "votes_up": 0,
        "votes_down": 0,
        "votes_unknown": 0,
        "skipped_unauthorized": 0,
        "errors": [],
    }

    try:
        replies = fetch_replies(cfg)
    except Exception as e:  # noqa: BLE001
        logger.warning("inbound: fetch_replies failed: %s", e)
        summary["errors"].append(f"fetch: {e}")
        return summary

    for reply in replies:
        summary["messages"] += 1
        try:
            if expected_sender:
                from_hdr = reply.get("from", "") or ""
                sender_addr = email.utils.parseaddr(from_hdr)[1].strip().lower()
                if sender_addr != expected_sender:
                    summary["skipped_unauthorized"] += 1
                    logger.warning(
                        "inbound: skipping reply from non-authorized sender %s",
                        sender_addr or "<unknown>",
                    )
                    continue

            line = extract_vote_line(reply.get("body", "") or "")
            if not line:
                logger.info(
                    "inbound: no vote tokens in reply subject=%r",
                    reply.get("subject"),
                )
                continue
            counts = votes_mod.record_votes(line, digest_id=reply.get("digest_id"))
            summary["votes_up"] += int(counts.get("up", 0))
            summary["votes_down"] += int(counts.get("down", 0))
            summary["votes_unknown"] += int(counts.get("unknown", 0))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "inbound: failed to record votes for %r: %s",
                reply.get("message_id"),
                e,
            )
            summary["errors"].append(str(e))
            continue

    return summary
