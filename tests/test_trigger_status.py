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


def test_trigger_status_400_on_invalid_project(app_with_store):
    app, _store, _ = app_with_store
    client = TestClient(app)
    r = client.get("/trigger-status/abc123?project=BAD-CASE&category=mlops")
    assert r.status_code == 400
