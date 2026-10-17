from __future__ import annotations

from email.message import EmailMessage

import pytest

from dailydigest.inbound import extract_vote_line


def test_extract_vote_line_accepts_every_current_digest_section():
    body = "AI: +A3\nFunding: -F6\nEvents: E7"

    assert extract_vote_line(body) == "AI: +A3 Funding: -F6 Events: E7"


def test_inbound_configuration_reads_dotenv_settings(tmp_path, monkeypatch):
    from dailydigest import config, inbound

    monkeypatch.chdir(tmp_path)
    for key in ("IMAP_HOST", "IMAP_PORT", "IMAP_USER", "IMAP_PASSWORD", "IMAP_MAILBOX"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "IMAP_HOST=imap.example.test\nIMAP_PORT=1993\n"
        "IMAP_USER=inbox@example.test\nIMAP_PASSWORD=test-secret\nIMAP_MAILBOX=Replies\n"
    )
    config.get_settings.cache_clear()

    cfg = inbound.load_inbound_config()

    assert cfg.imap_host == "imap.example.test"
    assert cfg.imap_port == 1993
    assert cfg.imap_user == "inbox@example.test"
    assert cfg.imap_password == "test-secret"
    assert cfg.mailbox == "Replies"


@pytest.fixture
def reply_mailbox(monkeypatch):
    from dailydigest import inbound

    class Mailbox:
        def __init__(self):
            self.sender = "reader@example.test"
            self.fetch_queries = []
            self.stored = []

        def login(self, *_args):
            return "OK", []

        def select(self, *_args):
            return "OK", []

        def search(self, *_args):
            return "OK", [b"1"]

        def fetch(self, message_id, query):
            self.fetch_queries.append(query)
            message = EmailMessage()
            message["From"] = self.sender
            message["Subject"] = "Re: DailyDigest 2026-09-11"
            message.set_content("+R1")
            return "OK", [(b"1", message.as_bytes())]

        def store(self, *args):
            self.stored.append(args)
            return "OK", []

        def close(self):
            pass

        def logout(self):
            pass

    mailbox = Mailbox()
    monkeypatch.setattr(inbound.imaplib, "IMAP4_SSL", lambda *_args: mailbox)
    for key, value in {
        "IMAP_HOST": "imap.example.test",
        "IMAP_USER": "inbox@example.test",
        "IMAP_PASSWORD": "test-secret",
        "DIGEST_TO": "reader@example.test",
        "REPLY_TO_EMAIL": "inbox@example.test",
    }.items():
        monkeypatch.setenv(key, value)
    from dailydigest import config

    config.get_settings.cache_clear()
    return mailbox


def test_inbound_accepts_digest_recipient_and_marks_seen_after_vote_save(reply_mailbox, monkeypatch):
    from dailydigest import inbound

    votes = []

    def record(line, digest_id, *, run_id, sent_only):
        assert reply_mailbox.stored == []
        assert run_id is None and sent_only
        votes.append((line, digest_id))
        return {"up": 1}

    monkeypatch.setattr(inbound.votes_mod, "record_votes", record)
    summary = inbound.process_replies()

    assert votes == [("+R1", "2026-09-11")]
    assert summary["votes_up"] == 1
    assert reply_mailbox.fetch_queries == ["(BODY.PEEK[])"]
    assert len(reply_mailbox.stored) == 1


def test_failed_vote_save_leaves_reply_unread_for_retry(reply_mailbox, monkeypatch):
    from dailydigest import inbound

    reply_mailbox.sender = "inbox@example.test"
    monkeypatch.setenv("DIGEST_TO", reply_mailbox.sender)
    from dailydigest import config

    config.get_settings.cache_clear()

    def fail_vote(*_args, **_kwargs):
        raise OSError("database unavailable")

    monkeypatch.setattr(inbound.votes_mod, "record_votes", fail_vote)
    summary = inbound.process_replies()

    assert summary["errors"] == ["database unavailable"]
    assert reply_mailbox.stored == []
    assert reply_mailbox.fetch_queries == ["(BODY.PEEK[])"]


def test_missing_reply_to_does_not_allow_unknown_senders(reply_mailbox, monkeypatch):
    from dailydigest import config, inbound

    monkeypatch.setenv("REPLY_TO_EMAIL", "")
    config.get_settings.cache_clear()
    reply_mailbox.sender = "stranger@example.test"
    votes = []
    monkeypatch.setattr(inbound.votes_mod, "record_votes", lambda *args, **kwargs: votes.append(args))

    summary = inbound.process_replies()

    assert votes == []
    assert summary["skipped_unauthorized"] == 1


def test_fetch_without_processor_preserves_unread_state(reply_mailbox):
    from dailydigest import inbound

    replies = inbound.fetch_replies(inbound.load_inbound_config())

    assert len(replies) == 1
    assert reply_mailbox.stored == []
    assert reply_mailbox.fetch_queries == ["(BODY.PEEK[])"]


def test_html_reply_preserves_vote_boundaries_and_omits_quoted_digest():
    from dailydigest import inbound

    message = EmailMessage()
    message.set_content(
        "<div>+R1</div><div>-R2</div>"
        "<blockquote><p>DailyDigest R3 R4</p></blockquote>",
        subtype="html",
    )

    body = inbound._strip_quoted(inbound._get_body(message))
    line = inbound.extract_vote_line(body)

    assert line == "+R1 -R2"
