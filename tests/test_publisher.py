"""Unit tests for publisher.publish_trigger — the core S3 upload + dispatch flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rest_app.ports.orchestration import OrchestrationAdapter
from rest_app.ports.storage import ArtifactStore
from rest_app.publisher import publish_trigger


class FakeStore(ArtifactStore):
    """In-memory ArtifactStore for tests."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deletes: list[tuple[str, str]] = []
        self.upload_calls: list[tuple[str, str, str, str | None]] = []
        self._fail_on = fail_on

    def upload_file(self, bucket, local_path, logical_key, *, content_type=None):
        self.upload_calls.append((bucket, str(local_path), logical_key, content_type))
        if self._fail_on and logical_key.endswith(self._fail_on):
            raise OSError("simulated upload failure")
        self.objects[(bucket, logical_key)] = Path(local_path).read_bytes()

    def put_bytes(self, bucket, data, logical_key, *, content_type=None):
        if self._fail_on and logical_key.endswith(self._fail_on):
            raise OSError("simulated put_bytes failure")
        self.objects[(bucket, logical_key)] = data

    def delete(self, bucket, logical_key):
        self.deletes.append((bucket, logical_key))
        self.objects.pop((bucket, logical_key), None)

    def get_json(self, bucket, logical_key):
        raw = self.objects.get((bucket, logical_key))
        return json.loads(raw) if raw else None

    def download_file(self, bucket, logical_key, local_path):
        Path(local_path).write_bytes(self.objects[(bucket, logical_key)])

    def list_subkeys(self, bucket, prefix):
        return iter([])


class CapturingOrchestrator(OrchestrationAdapter):
    def __init__(self, *, raise_runtime: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raise = raise_runtime

    def dispatch_training(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise RuntimeError("simulated dispatch refusal")


@pytest.fixture
def trigger_files(tmp_path) -> tuple[Path, Path]:
    dataset = tmp_path / "data.parquet"
    dataset.write_bytes(b"PARQUET_FAKE_PAYLOAD")
    params = tmp_path / "params.yaml"
    params.write_text("model: lgbm\nfeatures: [a, b]\n")
    return dataset, params


def test_publish_trigger_writes_three_files_in_order(trigger_files):
    dataset, params = trigger_files
    store = FakeStore()
    orch = CapturingOrchestrator()

    trigger_id, uri = publish_trigger(
        dataset_path=dataset,
        params_path=params,
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        model_family="forecasting",
        bucket="mlops-artifacts",
        prefix="",
        auto_promote=False,
        store=store,
        orchestrator=orch,
    )

    keys = [k for (_b, k) in store.objects.keys()]
    assert any(k.endswith("dataset.parquet") for k in keys)
    assert any(k.endswith("params.yaml") for k in keys)
    assert any(k.endswith("trigger.json") for k in keys)
    assert trigger_id in uri
    assert uri.startswith("s3://mlops-artifacts/_triggers/product_dq/")


def test_publish_trigger_writes_trigger_json_last(trigger_files):
    """If params upload fails, trigger.json must NOT have been written."""
    dataset, params = trigger_files
    store = FakeStore(fail_on="params.yaml")
    orch = CapturingOrchestrator()

    with pytest.raises(OSError):
        publish_trigger(
            dataset_path=dataset,
            params_path=params,
            category="mlops",
            project="product_dq",
            model_name="sentiment_analysis",
            model_family="forecasting",
            bucket="mlops-artifacts",
            prefix="",
            auto_promote=False,
            store=store,
            orchestrator=orch,
        )

    keys = [k for (_b, k) in store.objects.keys()]
    assert not any(k.endswith("trigger.json") for k in keys)
    # Orphan dataset.parquet should have been cleaned up
    assert any(d[1].endswith("dataset.parquet") for d in store.deletes)


def test_publish_trigger_writes_failure_marker_on_dispatch_refusal(trigger_files):
    dataset, params = trigger_files
    store = FakeStore()
    orch = CapturingOrchestrator(raise_runtime=True)

    with pytest.raises(RuntimeError):
        publish_trigger(
            dataset_path=dataset,
            params_path=params,
            category="mlops",
            project="product_dq",
            model_name="sentiment_analysis",
            model_family="forecasting",
            bucket="mlops-artifacts",
            prefix="",
            auto_promote=False,
            store=store,
            orchestrator=orch,
        )

    failed_keys = [k for (_b, k) in store.objects.keys() if k.endswith("failed.json")]
    assert len(failed_keys) == 1
    payload = store.get_json("mlops-artifacts", failed_keys[0])
    assert payload["status"] == "failed"
    assert "orchestrator dispatch failed" in payload["reason"]


def test_publish_trigger_csv_format_inferred_from_extension(tmp_path):
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b\n1,2\n")
    params = tmp_path / "params.yaml"
    params.write_text("model: lgbm\n")
    store = FakeStore()
    orch = CapturingOrchestrator()

    publish_trigger(
        dataset_path=dataset,
        params_path=params,
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        model_family="forecasting",
        bucket="mlops-artifacts",
        prefix="",
        auto_promote=False,
        store=store,
        orchestrator=orch,
    )
    keys = [k for (_b, k) in store.objects.keys()]
    assert any(k.endswith("dataset.csv") for k in keys)


def test_publish_trigger_unknown_extension_rejected(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text("[]")
    params = tmp_path / "params.yaml"
    params.write_text("x: 1\n")
    with pytest.raises(ValueError, match="Cannot infer dataset_format"):
        publish_trigger(
            dataset_path=dataset,
            params_path=params,
            category="mlops",
            project="product_dq",
            model_name="sentiment_analysis",
            model_family="forecasting",
            bucket="mlops-artifacts",
            prefix="",
            auto_promote=False,
            store=FakeStore(),
            orchestrator=CapturingOrchestrator(),
        )


def test_publish_trigger_with_prefix(trigger_files):
    dataset, params = trigger_files
    store = FakeStore()
    orch = CapturingOrchestrator()

    _, uri = publish_trigger(
        dataset_path=dataset,
        params_path=params,
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        model_family="forecasting",
        bucket="mlops-artifacts",
        prefix="prod",
        auto_promote=False,
        store=store,
        orchestrator=orch,
    )
    assert uri.startswith("s3://mlops-artifacts/prod/_triggers/product_dq/")
    keys = [k for (_b, k) in store.objects.keys()]
    assert all(k.startswith("prod/_triggers/") for k in keys)


def test_published_trigger_json_uris_have_single_prefix(trigger_files):
    """Regression — _full_uri must not double-prepend the prefix that
    trigger_*_key() already includes. If this fails, the training-side
    puller will 404 because dataset_uri in trigger.json points to
    s3://bucket/prod/prod/_triggers/... instead of s3://bucket/prod/_triggers/."""
    dataset, params = trigger_files
    store = FakeStore()
    orch = CapturingOrchestrator()

    publish_trigger(
        dataset_path=dataset,
        params_path=params,
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        model_family="forecasting",
        bucket="mlops-artifacts",
        prefix="prod",
        auto_promote=False,
        store=store,
        orchestrator=orch,
    )

    trigger_json_blob = next(
        v for (_b, k), v in store.objects.items() if k.endswith("trigger.json")
    )
    payload = json.loads(trigger_json_blob)
    # Single 'prod/' segment, not 'prod/prod/'
    assert "/prod/prod/" not in payload["dataset_uri"]
    assert "/prod/prod/" not in payload["params_uri"]
    assert payload["dataset_uri"].startswith("s3://mlops-artifacts/prod/_triggers/product_dq/")
    assert payload["params_uri"].startswith("s3://mlops-artifacts/prod/_triggers/product_dq/")
    # And the dataset_uri must point at a key that actually exists in the store
    uploaded_key = payload["dataset_uri"].removeprefix("s3://mlops-artifacts/")
    assert ("mlops-artifacts", uploaded_key) in store.objects


def test_publish_trigger_forwards_auto_promote_to_orchestrator(trigger_files):
    dataset, params = trigger_files
    store = FakeStore()
    orch = CapturingOrchestrator()

    publish_trigger(
        dataset_path=dataset,
        params_path=params,
        category="mlops",
        project="product_dq",
        model_name="sentiment_analysis",
        model_family="forecasting",
        bucket="mlops-artifacts",
        prefix="",
        auto_promote=True,
        store=store,
        orchestrator=orch,
    )
    assert orch.calls[0]["auto_promote"] is True
