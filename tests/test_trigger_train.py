"""Tests for /trigger-train endpoint — auth, config gating, multipart upload."""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rest_app.app import create_app
from rest_app.config import Settings
from rest_app.ports.orchestration import OrchestrationAdapter
from rest_app.ports.storage import ArtifactStore
from tests.test_publisher import CapturingOrchestrator, FakeStore


@pytest.fixture
def configured_settings(monkeypatch) -> Settings:
    """Settings with training_repo + token set — simulates production config."""
    monkeypatch.setenv("GITHUB_TRAINING_REPO", "prescienceds/mlops")
    monkeypatch.setenv("GITHUB_PAT", "ghp_dummy_token_for_test")
    return Settings.from_env()


@pytest.fixture
def fake_store() -> ArtifactStore:
    return FakeStore()


@pytest.fixture
def fake_orchestrator() -> OrchestrationAdapter:
    return CapturingOrchestrator()


@pytest.fixture
def app_with_trigger(configured_settings, cache, fake_store, fake_orchestrator):
    return create_app(
        settings=configured_settings,
        cache=cache,
        writable_store=fake_store,
        orchestrator=fake_orchestrator,
    )


def _multipart_payload() -> dict[str, Any]:
    return {
        "category": (None, "mlops"),
        "project": (None, "product_dq"),
        "model_name": (None, "sentiment_analysis"),
        "model_family": (None, "forecasting"),
        "dataset": ("data.parquet", io.BytesIO(b"PARQUET_FAKE"), "application/octet-stream"),
        "params": ("params.yaml", io.BytesIO(b"model: lgbm\n"), "application/x-yaml"),
    }


def test_trigger_train_returns_503_when_training_repo_unconfigured(make_app, admin_token):
    """make_app uses default test settings (no GITHUB_TRAINING_REPO)."""
    client = TestClient(make_app)
    r = client.post(
        "/trigger-train",
        files=_multipart_payload(),
        headers={"X-Admin-Token": admin_token},
    )
    assert r.status_code == 503
    assert "GITHUB_TRAINING_REPO" in r.json()["detail"]


def test_trigger_train_requires_admin_token(app_with_trigger):
    client = TestClient(app_with_trigger)
    r = client.post("/trigger-train", files=_multipart_payload())
    assert r.status_code == 401


def test_trigger_train_rejects_wrong_admin_token(app_with_trigger):
    client = TestClient(app_with_trigger)
    r = client.post(
        "/trigger-train",
        files=_multipart_payload(),
        headers={"X-Admin-Token": "wrong"},
    )
    assert r.status_code == 401


def test_trigger_train_happy_path(app_with_trigger, admin_token, fake_store, fake_orchestrator):
    client = TestClient(app_with_trigger)
    r = client.post(
        "/trigger-train",
        files=_multipart_payload(),
        headers={"X-Admin-Token": admin_token},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "trigger_id" in body
    assert body["trigger_uri"].startswith("s3://mlops-artifacts/_triggers/product_dq/")
    assert body["status_url"].startswith("/trigger-status/")
    # All three files landed in the fake store
    keys = [k for (_b, k) in fake_store.objects.keys()]
    assert any(k.endswith("dataset.parquet") for k in keys)
    assert any(k.endswith("params.yaml") for k in keys)
    assert any(k.endswith("trigger.json") for k in keys)
    # And the orchestrator was called
    assert len(fake_orchestrator.calls) == 1
    assert fake_orchestrator.calls[0]["project"] == "product_dq"


def test_trigger_train_rejects_invalid_project(app_with_trigger, admin_token):
    client = TestClient(app_with_trigger)
    payload = _multipart_payload()
    payload["project"] = (None, "INVALID-UPPERCASE")
    r = client.post(
        "/trigger-train",
        files=payload,
        headers={"X-Admin-Token": admin_token},
    )
    assert r.status_code == 400
    assert "invalid project" in r.json()["detail"]


def test_trigger_train_rejects_oversized_dataset(
    configured_settings, cache, fake_store, fake_orchestrator, admin_token, monkeypatch
):
    monkeypatch.setenv("MAX_DATASET_BYTES", "100")
    tight_settings = Settings.from_env()
    app = create_app(
        settings=tight_settings,
        cache=cache,
        writable_store=fake_store,
        orchestrator=fake_orchestrator,
    )
    client = TestClient(app)
    payload = _multipart_payload()
    big = b"X" * 1000
    payload["dataset"] = ("data.parquet", io.BytesIO(big), "application/octet-stream")
    r = client.post(
        "/trigger-train",
        files=payload,
        headers={"X-Admin-Token": admin_token},
    )
    assert r.status_code == 413


def test_trigger_train_returns_502_on_dispatch_refusal(
    configured_settings, cache, fake_store, admin_token
):
    orch = CapturingOrchestrator(raise_runtime=True)
    app = create_app(
        settings=configured_settings,
        cache=cache,
        writable_store=fake_store,
        orchestrator=orch,
    )
    client = TestClient(app)
    r = client.post(
        "/trigger-train",
        files=_multipart_payload(),
        headers={"X-Admin-Token": admin_token},
    )
    assert r.status_code == 502
    # failed.json marker was written
    keys = [k for (_b, k) in fake_store.objects.keys()]
    assert any(k.endswith("failed.json") for k in keys)
