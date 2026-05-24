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
    # Sanity-check the form has the inputs the API requires.
    assert 'name="category"' in body
    assert 'name="project"' in body
    assert 'name="model_name"' in body
    assert 'name="model_family"' in body
    assert 'name="dataset"' in body
    assert 'name="params"' in body
    # And actually POSTs to the right endpoint.
    assert "/trigger-train" in body
    assert "/trigger-status/" in body
