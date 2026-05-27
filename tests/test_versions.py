"""Tests for /projects/{cat}/{project}/models/{model}/versions endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import BUCKET, CATEGORY, MODEL_NAME, PROJECT


@pytest.fixture
def client_with_versions(make_app, s3_world, publish_artifacts):
    """App pre-loaded with v1 (stable) and v2 (latest) for MODEL_NAME."""
    publish_artifacts(
        s3_world,
        BUCKET,
        PROJECT,
        MODEL_NAME,
        2,
        feature_columns=["a", "b"],
        channel="latest",
    )
    return TestClient(make_app)


# ── list endpoint ──────────────────────────────────────────────────────────────


def test_list_versions_returns_all_published(client_with_versions):
    r = client_with_versions.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions")
    assert r.status_code == 200
    versions = r.json()["versions"]
    assert len(versions) == 2
    version_ids = [v["version_id"] for v in versions]
    assert "v1" in version_ids
    assert "v2" in version_ids


def test_list_versions_sorted_newest_first(client_with_versions):
    r = client_with_versions.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions")
    ids = [v["version_id"] for v in r.json()["versions"]]
    assert ids == ["v2", "v1"]


def test_list_versions_channels_labelled(client_with_versions):
    r = client_with_versions.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions")
    by_id = {v["version_id"]: v for v in r.json()["versions"]}
    assert "stable" in by_id["v1"]["channels"]
    assert "latest" in by_id["v2"]["channels"]


def test_list_versions_includes_s3_uris(make_app, s3_world):
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions")
    assert r.status_code == 200
    v = r.json()["versions"][0]
    assert v["model_pkl_uri"].startswith("s3://")
    assert v["model_pkl_uri"].endswith("/model.pkl")
    assert v["manifest_uri"].startswith("s3://")
    assert v["manifest_uri"].endswith("/manifest.json")


def test_list_versions_empty_for_unknown_model(make_app, s3_world):
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/nonexistent_model/versions")
    assert r.status_code == 200
    assert r.json()["versions"] == []


def test_list_versions_400_bad_category(make_app, s3_world):
    client = TestClient(make_app)
    r = client.get(f"/projects/BAD-CASE/{PROJECT}/models/{MODEL_NAME}/versions")
    assert r.status_code == 400


def test_list_versions_400_bad_model_name(make_app, s3_world):
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/BAD-NAME!/versions")
    assert r.status_code == 400


def test_list_versions_skips_partial_upload(make_app, s3_world):
    """A v{N} directory without manifest.json must not appear in the listing."""

    import boto3

    # Write model.pkl but NOT manifest.json for v99
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{PROJECT}/{MODEL_NAME}/v99/model.pkl",
        Body=b"fake-pkl",
    )
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions")
    ids = [v["version_id"] for v in r.json()["versions"]]
    assert "v99" not in ids


# ── detail endpoint ────────────────────────────────────────────────────────────


def test_get_version_returns_details(make_app, s3_world):
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions/v1")
    assert r.status_code == 200
    body = r.json()
    assert body["version_id"] == "v1"
    assert body["model_pkl_uri"].endswith("/model.pkl")
    assert body["manifest_uri"].endswith("/manifest.json")
    assert "stable" in body["channels"]
    assert "schema_contract" in body
    assert "artifact_checksums" in body


def test_get_version_404_unknown(make_app, s3_world):
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions/v99")
    assert r.status_code == 404


def test_get_version_400_bad_version_id(make_app, s3_world):
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions/notaversion")
    assert r.status_code == 400


def test_get_version_channel_labelled_latest(make_app, s3_world, publish_artifacts):
    publish_artifacts(s3_world, BUCKET, PROJECT, MODEL_NAME, 2, channel="latest")
    client = TestClient(make_app)
    r = client.get(f"/projects/{CATEGORY}/{PROJECT}/models/{MODEL_NAME}/versions/v2")
    assert r.status_code == 200
    assert "latest" in r.json()["channels"]
