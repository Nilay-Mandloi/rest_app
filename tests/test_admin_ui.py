"""Tests for the trigger.html admin UI mount."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_redirects_to_trigger_form(make_app):
    client = TestClient(make_app)
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/static/trigger.html"


def test_trigger_form_served(make_app):
    client = TestClient(make_app)
    r = client.get("/static/trigger.html")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    # Trigger tab inputs
    assert 'id="t-category"' in body
    assert 'id="t-project"' in body
    assert 'id="t-model_name"' in body
    assert 'id="t-model_family"' in body
    assert 'id="t-dataset"' in body
    assert 'id="t-params"' in body
    # Predict tab — single baked model, no routing inputs
    assert 'id="p-load"' in body
    assert 'id="p-i-version"' in body
    assert 'id="p-i-features"' in body
    assert 'id="p-features-form"' in body
    assert 'id="p-predict"' in body
    # The old multi-model routing inputs must be gone
    assert 'id="p-channel"' not in body
    assert 'id="p-version"' not in body
    # All four endpoints the UI calls
    assert "/trigger-train" in body
    assert "/trigger-status/" in body
    assert "/model/info" in body
    assert "/predict" in body
    # Two tabs present
    assert 'data-tab="trigger"' in body
    assert 'data-tab="predict"' in body
