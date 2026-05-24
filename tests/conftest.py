from __future__ import annotations

import hashlib
import json
import pickle
from datetime import UTC, datetime

import boto3
import pytest
from moto import mock_aws
from rest_app.adapters.s3_store import S3ReadStore
from rest_app.config import Settings
from rest_app.loader import ModelCache

CATEGORY = "mlops"
PROJECT = "product_dq"
MODEL_NAME = "sentiment_analysis"
BUCKET = f"{CATEGORY}-artifacts"
PREFIX = ""
ADMIN_TOKEN = "test-token"


class ToyModel:
    """Picklable toy that returns the sum of each row."""

    def predict(self, X):
        out = []
        for row in X:
            try:
                out.append(float(sum(float(v or 0) for v in row)))
            except Exception:
                out.append(0.0)
        return out


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DEFAULT_CATEGORY", CATEGORY)
    monkeypatch.setenv("DEFAULT_PROJECT", PROJECT)
    monkeypatch.setenv("DEFAULT_MODEL_NAME", MODEL_NAME)
    monkeypatch.setenv("APP_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")
    yield


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _publish_artifacts(
    s3,
    bucket: str,
    project: str,
    model_name: str,
    version: int,
    *,
    category: str = CATEGORY,
    channel: str = "stable",
    feature_columns: list[str] | None = None,
    bad_checksum: bool = False,
    create_bucket: bool = True,
):
    pkl_bytes = pickle.dumps(ToyModel())
    sha = hashlib.sha256(pkl_bytes).hexdigest()
    if bad_checksum:
        sha = "0" * 64
    if create_bucket:
        try:
            s3.create_bucket(Bucket=bucket)
        except s3.exceptions.BucketAlreadyOwnedByYou:
            pass
    root = f"{project}/{model_name}/v{version}"
    pkl_key = f"{root}/model.pkl"
    man_key = f"{root}/manifest.json"
    s3.put_object(Bucket=bucket, Key=pkl_key, Body=pkl_bytes)

    manifest = {
        "category": category,
        "project": project,
        "model_name": model_name,
        "version": version,
        "run_id": "run-xyz",
        "registry_version": "1",
        "model_type": "toy",
        "schema_hash": "abcdef",
        "artifact_checksums": {"model.pkl": sha},
        "schema_contract": ({"feature_columns": feature_columns} if feature_columns else {}),
        "published_at": _now(),
        "schema_version": "1.0",
    }
    s3.put_object(Bucket=bucket, Key=man_key, Body=json.dumps(manifest).encode())

    pointer = {
        "category": category,
        "project": project,
        "model_name": model_name,
        "version": version,
        "version_id": f"v{version}",
        "run_id": "run-xyz",
        "registry_version": "1",
        "manifest_uri": f"s3://{bucket}/{man_key}",
        "status": channel,
        "updated_at": _now(),
        "schema_version": "1.0",
    }
    s3.put_object(
        Bucket=bucket,
        Key=f"{project}/{model_name}/{channel}.json",
        Body=json.dumps(pointer).encode(),
    )
    return manifest, pointer


@pytest.fixture
def s3_world():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        _publish_artifacts(client, BUCKET, PROJECT, MODEL_NAME, 1, feature_columns=["a", "b"])
        yield client


@pytest.fixture
def cache(settings, s3_world) -> ModelCache:
    return ModelCache(settings, store=S3ReadStore(client=s3_world))


@pytest.fixture
def publish_artifacts():
    """Expose helper for tests that need to publish their own variants."""
    return _publish_artifacts


@pytest.fixture
def admin_token() -> str:
    return ADMIN_TOKEN


@pytest.fixture
def bucket_name() -> str:
    return BUCKET


@pytest.fixture
def make_app(settings, cache):
    from rest_app.app import create_app

    return create_app(settings=settings, cache=cache)
