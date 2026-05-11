from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def test_setup_post_accepts_urlencoded_form_without_multipart(tmp_path, monkeypatch):
    from dailydigest import web

    profile_path = tmp_path / "profile.yaml"
    env_path = tmp_path / ".env"
    monkeypatch.setattr(web, "_PROFILE_PATH", profile_path)
    monkeypatch.setattr(web, "_ENV_PATH", env_path)

    client = TestClient(web.app, raise_server_exceptions=False)
    response = client.request(
        "POST",
        "/setup",
        data={
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
        follow_redirects=False,
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
    monkeypatch.setattr(web, "_PROFILE_PATH", profile_path)

    client = TestClient(web.app, raise_server_exceptions=False)
    response = client.request(
        "POST",
        "/profile/name",
        data={"_csrf_token": web._CSRF_TOKEN, "name": "Hao"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "name: Hao" in Path(profile_path).read_text()


def test_index_redirects_to_setup_when_local_profile_missing(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_PROFILE_PATH", tmp_path / "profile.yaml")

    client = TestClient(web.app, raise_server_exceptions=False)
    response = client.request("GET", "/", follow_redirects=False)

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
    monkeypatch.setattr(web, "_PROFILE_PATH", profile_path)
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

    client = TestClient(web.app, raise_server_exceptions=False)
    response = client.request("GET", "/")

    assert response.status_code == 200
    assert "<script>alert(1)</script>" not in response.text
    assert 'Source <img src=x onerror="alert(1)">' not in response.text
    assert "<b>summary</b>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert 'href="javascript:alert(1)"' not in response.text
    assert 'href="#"' in response.text


def test_setup_post_rejects_missing_csrf_token(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_PROFILE_PATH", tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    client = TestClient(web.app, raise_server_exceptions=False)
    response = client.request(
        "POST",
        "/setup",
        data={"bio": "Researcher.", "keywords": "RNA", "llm_backend": "extractive"},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert not (tmp_path / "profile.yaml").exists()
    assert not (tmp_path / ".env").exists()


def test_setup_post_rejects_env_newline_injection(tmp_path, monkeypatch):
    from dailydigest import web

    monkeypatch.setattr(web, "_PROFILE_PATH", tmp_path / "profile.yaml")
    monkeypatch.setattr(web, "_ENV_PATH", tmp_path / ".env")

    client = TestClient(web.app, raise_server_exceptions=False)
    response = client.request(
        "POST",
        "/setup",
        data={
            "_csrf_token": web._CSRF_TOKEN,
            "bio": "Researcher.",
            "keywords": "RNA",
            "llm_backend": "api",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_model": "gpt-4o-mini\nTOP_RESEARCH=30",
        },
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "cannot contain line breaks" in response.text
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

    client = TestClient(web.app, raise_server_exceptions=False)
    assert client.request("POST", f"/vote/{item_id}/1").status_code == 403

    headers = {"X-CSRF-Token": web._CSRF_TOKEN}
    r1 = client.request("POST", f"/vote/{item_id}/1", headers=headers)
    r2 = client.request("POST", f"/vote/{item_id}/-1", headers=headers)
    r3 = client.request("POST", f"/vote/{item_id}/0", headers=headers)

    assert r1.json()["ok"] is True
    assert r1.json()["item_id"] == item_id
    assert r1.json()["new_value"] == 1
    assert r1.json()["ranking_status"]["vote_counts"]["good"] == 1
    assert r2.json()["ok"] is True
    assert r2.json()["item_id"] == item_id
    assert r2.json()["new_value"] == -1
    assert r2.json()["ranking_status"]["vote_counts"]["bad"] == 1
    assert r3.json()["ok"] is True
    assert r3.json()["item_id"] == item_id
    assert r3.json()["new_value"] == 0
    assert r3.json()["ranking_status"]["vote_counts"]["neutral"] == 1

    with store_mod.session_scope() as s:
        rows = s.execute(store_mod.select(store_mod.VoteRow)).scalars().all()
        assert len(rows) == 1
        assert rows[0].value == 0

    assert client.request("POST", f"/vote/{item_id}/7", headers=headers).status_code == 400
    assert client.request("POST", "/vote/999999/1", headers=headers).status_code == 404


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

    client = TestClient(web.app, raise_server_exceptions=False)
    response = client.request("GET", "/ranking/status")

    assert response.status_code == 200
    status = response.json()["status"]
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

    fitted: list[tuple[np.ndarray, np.ndarray]] = []
    reset_calls: list[bool] = []

    class DummyRanker:
        trained = False

        def load(self):
            return self.trained

        def fit(self, X, y):
            fitted.append((X, y))
            DummyRanker.trained = True

    monkeypatch.setattr(votes_mod, "LRRanker", DummyRanker)
    monkeypatch.setattr(votes_mod, "reset_lr_cache", lambda: reset_calls.append(True))
    monkeypatch.setattr(
        votes_mod,
        "vote_dataset",
        lambda: (
            np.asarray([[0.1, 0.2], [0.9, 0.8]], dtype=np.float32),
            np.asarray([1, -1], dtype=np.float32),
        ),
    )

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

    client = TestClient(web.app, raise_server_exceptions=False)
    assert client.request("POST", "/ranking/train").status_code == 403

    response = client.request(
        "POST",
        "/ranking/train",
        headers={"X-CSRF-Token": web._CSRF_TOKEN},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["trained"] is True
    assert payload["trained_votes"] == 2
    assert payload["status"]["model_trained"] is True
    assert payload["status"]["ranking_status"] == "lr_active"
    assert len(fitted) == 1
    assert fitted[0][0].shape == (2, 2)
    assert reset_calls == [True]


def test_run_start_rejects_foreign_origin_and_detects_duplicate(monkeypatch):
    from dailydigest import web

    run_ids = []
    monkeypatch.setattr(web, "_kick_off_run", lambda run_id: run_ids.append(run_id))
    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()

    client = TestClient(web.app, raise_server_exceptions=False)
    headers = {"X-CSRF-Token": web._CSRF_TOKEN}

    foreign = client.request(
        "POST",
        "/run/start",
        json={"run_id": "abc"},
        headers={**headers, "Origin": "https://evil.example"},
    )
    assert foreign.status_code == 403

    first = client.request("POST", "/run/start", json={"run_id": "abc"}, headers=headers)
    second = client.request("POST", "/run/start", json={"run_id": "abc"}, headers=headers)

    assert first.json() == {"ok": True, "run_id": "abc"}
    assert second.json() == {"ok": True, "run_id": "abc", "already_started": True}
    assert run_ids == ["abc"]

    web._RUN_QUEUES.clear()
    web._RUN_STARTED.clear()
