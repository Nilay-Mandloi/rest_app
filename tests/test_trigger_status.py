"""Tests for /trigger-status/{trigger_id} — lifecycle state from S3 markers."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rest_app.app import create_app
from rest_app.config import Settings
from tests.test_publisher import CapturingOrchestrator, FakeStore


@pytest.fixture
def app_with_store():
    settings = Settings.from_env()
    store = FakeStore()
    app = create_app(
        settings=settings,
        writable_store=store,
        orchestrator=CapturingOrchestrator(),
    )
    return app, store, settings


def _write_marker(store, bucket, key, body):
    store.put_bytes(bucket, json.dumps(body).encode(), key, content_type="application/json")


def test_trigger_status_404_when_no_marker(app_with_store):
    app, _store, _settings = app_with_store
    client = TestClient(app)
    r = client.get(
        "/trigger-status/no-such-trigger?project=product_dq&category=mlops",
    )
    assert r.status_code == 404


def test_trigger_status_pending_when_only_metadata(app_with_store, bucket_name):
    app, store, _ = app_with_store
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/trigger.json",
        {"trigger_id": "abc123", "category": "mlops"},
    )
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=product_dq&category=mlops")
    assert r.status_code == 200
    assert r.json() == {"trigger_id": "abc123", "status": "pending"}


def test_trigger_status_running_when_running_marker(app_with_store, bucket_name):
    app, store, _ = app_with_store
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/trigger.json",
        {"trigger_id": "abc123"},
    )
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/running.json",
        {"trigger_id": "abc123", "status": "running"},
    )
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=product_dq&category=mlops")
    assert r.status_code == 200
    assert r.json() == {"trigger_id": "abc123", "status": "running"}


def test_trigger_status_failed_wins_over_running(app_with_store, bucket_name):
    app, store, _ = app_with_store
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/trigger.json",
        {"trigger_id": "abc123"},
    )
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/running.json",
        {"trigger_id": "abc123", "status": "running"},
    )
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/failed.json",
        {"trigger_id": "abc123", "status": "failed", "reason": "CI exploded"},
    )
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=product_dq&category=mlops")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert body["reason"] == "CI exploded"


def test_trigger_status_completed_no_promotion(app_with_store, bucket_name):
    """Completed without a stable pointer → just status, no model artifact fields."""
    app, store, _ = app_with_store
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/trigger.json",
        {"trigger_id": "abc123", "model_name": "sales_model"},
    )
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/running.json",
        {"trigger_id": "abc123", "status": "running"},
    )
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/completed.json",
        {"trigger_id": "abc123", "status": "completed"},
    )
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=product_dq&category=mlops")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert "model_pkl_uri" not in body


def test_trigger_status_completed_with_model_pkl_uri(app_with_store, bucket_name):
    """Completed + stable pointer present → response includes model_pkl_uri."""
    app, store, settings = app_with_store
    # Write trigger.json with model_name so the endpoint knows what to look up.
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/trigger.json",
        {"trigger_id": "abc123", "model_name": "sales_model"},
    )
    _write_marker(
        store,
        bucket_name,
        "_triggers/product_dq/abc123/completed.json",
        {"trigger_id": "abc123", "status": "completed"},
    )
    # Write stable pointer as the training workflow would after promotion.
    from rest_app.layout import pointer_key

    ptr_key = pointer_key(settings.prefix, "product_dq", "sales_model", "stable")
    manifest_uri = (
        f"s3://{bucket_name}/{settings.prefix + '/' if settings.prefix else ''}"
        "product_dq/sales_model/v3/manifest.json"
    )
    _write_marker(
        store,
        bucket_name,
        ptr_key,
        {
            "version_id": "v3",
            "version": 3,
            "manifest_uri": manifest_uri,
            "status": "stable",
        },
    )
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=product_dq&category=mlops")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["version_id"] == "v3"
    assert body["model_pkl_uri"] == manifest_uri.replace("/manifest.json", "/model.pkl")
    assert body["manifest_uri"] == manifest_uri


def test_trigger_status_failed_wins_over_completed(app_with_store, bucket_name):
    """failed.json written after completed.json (shouldn't happen, but guard it)."""
    app, store, _ = app_with_store
    for name in ("trigger.json", "running.json", "completed.json", "failed.json"):
        _write_marker(
            store,
            bucket_name,
            f"_triggers/product_dq/abc123/{name}",
            {"trigger_id": "abc123", "status": name.split(".")[0]},
        )
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=product_dq&category=mlops")
    assert r.status_code == 200
    assert r.json()["status"] == "failed"


def test_trigger_status_400_on_invalid_project(app_with_store):
    app, _store, _ = app_with_store
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=BAD-CASE&category=mlops")
    assert r.status_code == 400
