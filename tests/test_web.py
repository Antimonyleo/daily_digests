from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import threading
from urllib.parse import urlencode

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def _request(
    method: str = "GET",
    path: str = "/",
    *,
    headers: dict[str, str] | None = None,
    form: dict[str, str] | None = None,
    json_body: dict | None = None,
) -> Request:
    body = b""
    request_headers = {"host": "testserver"}
    if form is not None:
        body = urlencode(form).encode("utf-8")
        request_headers["content-type"] = "application/x-www-form-urlencoded"
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        request_headers["content-type"] = "application/json"
    request_headers.update(headers or {})
    header_items = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in request_headers.items()
    ]
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": header_items,
        "client": ("testclient", 123),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _json_payload(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def _text_payload(response) -> str:
    return response.body.decode("utf-8")


def test_setup_post_accepts_urlencoded_form_without_multipart(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", env_path)

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "name": "Hao",
                    "bio": "Researcher in RNA nanotechnology.",
                    "keywords": "RNA nanotechnology, protein design",
                    "downweight": "celebrity news",
                    "llm_backend": "extractive",
                    "llm_base_url": "https://api.openai.com/v1",
                    "llm_api_key": "",
                    "llm_model": "gpt-4o-mini",
                    "claude_cli_model": "",
                    "codex_cli_model": "",
                },
            )
        )
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/run"
    saved_profile = profile_path.read_text()
    assert "name: Hao" in saved_profile
    assert "RNA nanotechnology" in saved_profile
    assert "LLM_BACKEND=extractive" in env_path.read_text()


def test_profile_name_post_updates_existing_profile(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "bio: Existing bio.\nkeywords:\n- RNA\ndownweight: []\n"
    )
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)

    response = asyncio.run(
        web.profile_name_post(
            _request(
                "POST",
                "/profile/name",
                form={"_csrf_token": web._CSRF_TOKEN, "name": "Hao"},
            )
        )
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "name: Hao" in Path(profile_path).read_text()


def test_index_redirects_to_setup_when_local_profile_missing(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")

    response = web.index(_request("GET", "/"))

    assert response.status_code == 302
    assert response.headers["location"] == "/setup"


def test_index_escapes_feed_html_and_rejects_unsafe_links(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        s.add(
            store_mod.ItemRow(
                source='Source <img src=x onerror="alert(1)">',
                section="research",
                external_id="xss-web",
                url="javascript:alert(1)",
                title="<script>alert(1)</script>",
                summary="<b>summary</b>",
                digest_id="2026-05-05",
                item_label="R1",
            )
        )

    response = web.index(_request("GET", "/"))
    text = _text_payload(response)

    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in text
    assert 'Source <img src=x onerror="alert(1)">' not in text
    assert "<b>summary</b>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert 'href="javascript:alert(1)"' not in text
    assert 'href="#"' in text


def test_index_renders_reader_card_hierarchy_and_feedback_controls(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("name: Ada\nbio: Researcher.\nkeywords: []\ndownweight: []\n")
    monkeypatch.setattr(web, "_get_profile_path", lambda: profile_path)
    monkeypatch.setattr(web, "_digest_id", lambda: "2026-05-05")

    store_mod.init_db()
    with store_mod.session_scope() as s:
        s.add(store_mod.DigestRow(id="2026-05-05", item_count=1))
        s.add(
            store_mod.ItemRow(
                source="Nature",
                section="research",
                external_id="ranking-web",
                url="https://example.com/ranking-web",
                title="First-in-class RNA delivery study",
                abstract="A breakthrough primary result.",
                published_at=datetime.now(timezone.utc),
                summary=(
                    "Key finding: A delivery method improved tissue targeting.\n"
                    "Why read: It may matter for RNA therapeutic design.\n"
                    "Caveat: None obvious from the feed text."
                ),
                score=0.91,
                digest_id="2026-05-05",
                item_label="R1",
            )
        )

    response = web.index(_request("GET", "/"))
    text = _text_payload(response)

    assert response.status_code == 200
    assert "Must read first" in text
    assert "Today’s source mix" in text
    assert "Lead story" in text
    assert "Why shown?" in text
    assert "editorial-signals" in text
    assert "score-bars" not in text
    assert "High-quality source" in text
    assert 'data-filter="priority"' in text
    assert 'data-filter="unreviewed"' in text
    assert 'data-filter="published"' in text
    assert 'data-filter="preprints"' in text
    assert 'data-filter="ai-cs"' in text
    assert "summary-fields" in text
    assert "Key finding" in text
    assert "More like this" in text
    assert "Less like this" in text
    assert "No response saved yet." in text
    assert "Too promotional" in text
    assert "Update my ranking" in text
    assert "Learned" not in text


def test_load_today_uses_latest_vote_when_legacy_duplicates_exist(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy_web.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE items (
            id INTEGER PRIMARY KEY,
            source VARCHAR NOT NULL,
            section VARCHAR NOT NULL,
            external_id VARCHAR NOT NULL,
            url VARCHAR NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT DEFAULT '',
            authors TEXT DEFAULT '',
            published_at DATETIME,
            fetched_at DATETIME,
            summary TEXT DEFAULT '',
            score FLOAT,
            digest_id VARCHAR,
            item_label VARCHAR
        );
        CREATE TABLE votes (
            id INTEGER PRIMARY KEY,
            item_id INTEGER NOT NULL,
            value INTEGER NOT NULL,
            created_at DATETIME
        );
        INSERT INTO items (
            id, source, section, external_id, url, title, abstract, authors,
            published_at, fetched_at, summary, score, digest_id, item_label
        ) VALUES (
            1, 'Legacy', 'research', 'legacy-web', 'https://example.com/legacy-web',
            'Legacy item', 'Abstract', '', '2026-05-12 00:00:00',
            '2026-05-12 00:00:00', '', 0.5, '2026-05-12', 'R1'
        );
        INSERT INTO votes (id, item_id, value, created_at)
        VALUES
            (1, 1, 1, '2026-05-12 10:00:00'),
            (2, 1, -1, '2026-05-12 11:00:00');
        """
    )
    conn.commit()
    conn.close()

    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(db_path))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None

    sections, current_vote = web._load_today("2026-05-12")

    assert current_vote == {1: -1}
    assert sections[0]["entries"][0]["current_vote"] == -1


def test_setup_post_rejects_missing_csrf_token(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            web.setup_post(
                _request(
                    "POST",
                    "/setup",
                    form={"bio": "Researcher.", "keywords": "RNA", "llm_backend": "extractive"},
                )
            )
        )

    assert excinfo.value.status_code == 403
    assert not (tmp_path / "profile.yaml").exists()
    assert not (tmp_path / ".env").exists()


def test_setup_post_rejects_env_newline_injection(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_get_profile_path", lambda: tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    response = asyncio.run(
        web.setup_post(
            _request(
                "POST",
                "/setup",
                form={
                    "_csrf_token": web._CSRF_TOKEN,
                    "bio": "Researcher.",
                    "keywords": "RNA",
                    "llm_backend": "api",
                    "llm_base_url": "https://api.openai.com/v1",
                    "llm_model": "gpt-4o-mini\nTOP_RESEARCH=30",
                },
            )
        )
    )

    assert response.status_code == 400
    assert "cannot contain line breaks" in _text_payload(response)
    assert not (tmp_path / "profile.yaml").exists()
    assert not (tmp_path / ".env").exists()


def test_vote_api_requires_csrf_and_latest_vote_wins(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None

    store_mod.init_db()
    with store_mod.session_scope() as s:
        item = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="vote-api",
            url="https://example.com/vote-api",
            title="Vote API",
            abstract="Abstract.",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    with pytest.raises(HTTPException) as excinfo:
        web.vote(_request("POST", f"/vote/{item_id}/1"), item_id, 1)
    assert excinfo.value.status_code == 403

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    r1 = web.vote(_request("POST", f"/vote/{item_id}/1", headers=headers), item_id, 1)
    r2 = web.vote(_request("POST", f"/vote/{item_id}/-1", headers=headers), item_id, -1)
    r3 = web.vote(_request("POST", f"/vote/{item_id}/0", headers=headers), item_id, 0)
    p1 = _json_payload(r1)
    p2 = _json_payload(r2)
    p3 = _json_payload(r3)

    assert p1["ok"] is True
    assert p1["item_id"] == item_id
    assert p1["new_value"] == 1
    assert p1["ranking_status"]["vote_counts"]["good"] == 1
    assert p2["ok"] is True
    assert p2["item_id"] == item_id
    assert p2["new_value"] == -1
    assert p2["ranking_status"]["vote_counts"]["bad"] == 1
    assert p3["ok"] is True
    assert p3["item_id"] == item_id
    assert p3["new_value"] == 0
    assert p3["ranking_status"]["vote_counts"]["neutral"] == 1

    with store_mod.session_scope() as s:
        rows = s.execute(store_mod.select(store_mod.VoteRow)).scalars().all()
        assert len(rows) == 1
        assert rows[0].value == 0

    with pytest.raises(HTTPException) as bad_value:
        web.vote(_request("POST", f"/vote/{item_id}/7", headers=headers), item_id, 7)
    assert bad_value.value.status_code == 400
    with pytest.raises(HTTPException) as missing_item:
        web.vote(_request("POST", "/vote/999999/1", headers=headers), 999999, 1)
    assert missing_item.value.status_code == 404


def test_vote_reason_api_requires_csrf_and_persists_reason(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None

    store_mod.init_db()
    with store_mod.session_scope() as s:
        item = store_mod.ItemRow(
            source="Test",
            section="research",
            external_id="vote-reason-api",
            url="https://example.com/vote-reason-api",
            title="Vote Reason API",
            abstract="Abstract.",
        )
        s.add(item)
        s.flush()
        item_id = int(item.id)

    with pytest.raises(HTTPException) as excinfo:
        web.vote_reason(
            _request("POST", f"/vote/{item_id}/reason/low_impact"),
            item_id,
            "low_impact",
        )
    assert excinfo.value.status_code == 403

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    response = web.vote_reason(
        _request("POST", f"/vote/{item_id}/reason/low_impact", headers=headers),
        item_id,
        "low_impact",
    )

    assert response.status_code == 200
    assert _json_payload(response)["reasons"] == ["low_impact"]
    deleted = web.vote_reason_delete(
        _request("DELETE", f"/vote/{item_id}/reason/low_impact", headers=headers),
        item_id,
        "low_impact",
    )
    assert deleted.status_code == 200
    assert _json_payload(deleted)["reasons"] == []
    with pytest.raises(HTTPException) as bad_reason:
        web.vote_reason(
            _request("POST", f"/vote/{item_id}/reason/not_a_reason", headers=headers),
            item_id,
            "not_a_reason",
        )
    assert bad_reason.value.status_code == 400


def test_ranking_status_api_reports_vote_counts_without_training(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import votes as votes_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None

    class DummyRanker:
        def load(self):
            return True

    monkeypatch.setattr(votes_mod, "LRRanker", DummyRanker)

    store_mod.init_db()
    with store_mod.session_scope() as s:
        items = [
            store_mod.ItemRow(
                source="Test",
                section="research",
                external_id=f"status-{idx}",
                url=f"https://example.com/status-{idx}",
                title=f"Status {idx}",
            )
            for idx in range(3)
        ]
        s.add_all(items)
        s.flush()
        s.add(store_mod.VoteRow(item_id=items[0].id, value=1))
        s.add(store_mod.VoteRow(item_id=items[1].id, value=-1))
        s.add(store_mod.VoteRow(item_id=items[2].id, value=0))

    response = web.ranking_status(_request("GET", "/ranking/status"))

    assert response.status_code == 200
    status = _json_payload(response)["status"]
    assert status["vote_counts"] == {
        "good": 1,
        "bad": 1,
        "neutral": 1,
        "signed": 2,
        "total": 3,
    }
    assert status["model_trained"] is True
    assert status["training_status"] == "needs_votes"
    assert status["ranking_status"] == "cosine_baseline"


def test_ranking_train_api_uses_monkeypatched_dataset_and_ranker(tmp_path, monkeypatch):
    from dailydigest import config as config_mod
    from dailydigest import store as store_mod
    from dailydigest import votes as votes_mod
    from dailydigest import web

    monkeypatch.setenv("DB_PATH", str(tmp_path / "digest.db"))
    config_mod.reload_settings()
    store_mod.SETTINGS = config_mod.SETTINGS
    store_mod._ENGINE = None
    store_mod._SessionLocal = None
    monkeypatch.setattr(votes_mod, "MIN_VOTES_FOR_LR", 2)

    class DummyRanker:
        trained = False

        def load(self):
            return self.trained

    monkeypatch.setattr(votes_mod, "LRRanker", DummyRanker)
    release_train = threading.Event()

    def _slow_train():
        release_train.wait(5)
        return {"ok": True, "trained": True, "status": votes_mod.lr_training_status()}

    monkeypatch.setattr(votes_mod, "train_lr_ranker", _slow_train)

    store_mod.init_db()
    with store_mod.session_scope() as s:
        items = [
            store_mod.ItemRow(
                source="Test",
                section="research",
                external_id=f"train-{idx}",
                url=f"https://example.com/train-{idx}",
                title=f"Train {idx}",
            )
            for idx in range(2)
        ]
        s.add_all(items)
        s.flush()
        s.add(store_mod.VoteRow(item_id=items[0].id, value=1))
        s.add(store_mod.VoteRow(item_id=items[1].id, value=-1))

    with pytest.raises(HTTPException) as excinfo:
        web.ranking_train(_request("POST", "/ranking/train"))
    assert excinfo.value.status_code == 403
    web._TRAIN_JOB["running"] = False
    web._TRAIN_JOB["last_result"] = None

    response = web.ranking_train(
        _request("POST", "/ranking/train", headers={"X-CSRF-Token": web._CSRF_TOKEN})
    )

    assert response.status_code == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert payload["started"] is True
    assert payload["running"] is True
    assert payload["message"] == "Ranking training started."

    # A second click while training is in progress should return immediately
    # instead of starting another fit or hanging the UI.
    response2 = web.ranking_train(
        _request("POST", "/ranking/train", headers={"X-CSRF-Token": web._CSRF_TOKEN})
    )
    assert response2.status_code == 200
    payload2 = _json_payload(response2)
    assert payload2["ok"] is True
    assert payload2["started"] is False
    assert payload2["running"] is True
    release_train.set()


def test_run_start_rejects_foreign_origin_and_detects_duplicate(monkeypatch):
    from dailydigest import web

    run_ids = []
    monkeypatch.setattr(web, "_kick_off_run", lambda run_id: run_ids.append(run_id))
    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            web.run_start(
                _request(
                    "POST",
                    "/run/start",
                    json_body={"run_id": "abc"},
                    headers={**headers, "Origin": "https://evil.example"},
                )
            )
        )
    assert excinfo.value.status_code == 403

    first = asyncio.run(
        web.run_start(
            _request("POST", "/run/start", json_body={"run_id": "abc"}, headers=headers)
        )
    )
    second = asyncio.run(
        web.run_start(
            _request("POST", "/run/start", json_body={"run_id": "abc"}, headers=headers)
        )
    )

    assert _json_payload(first) == {"ok": True, "run_id": "abc"}
    assert _json_payload(second) == {"ok": True, "run_id": "abc", "already_started": True}
    assert run_ids == ["abc"]

    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()
