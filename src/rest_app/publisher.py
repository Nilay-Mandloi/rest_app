"""Publisher — pushes a training trigger (dataset + params + metadata) and
asks the orchestrator to start a run. Backend-neutral.

Producer side of the training contract. The training repo's `pull_trigger`
in quantity_forecast.trigger reads the same folder shape.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from rest_app.contracts import TriggerFile
from rest_app.layout import (
    trigger_dataset_key,
    trigger_failure_key,
    trigger_metadata_key,
    trigger_params_key,
)
from rest_app.ports.orchestration import OrchestrationAdapter
from rest_app.ports.storage import ArtifactStore

_EXTENSION_TO_FORMAT = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def _infer_dataset_format(dataset_path: Path, override: str | None) -> str:
    if override:
        if override not in {"csv", "parquet"}:
            raise ValueError(
                f"dataset_format override must be 'csv' or 'parquet'; got {override!r}"
            )
        return override
    fmt = _EXTENSION_TO_FORMAT.get(dataset_path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"Cannot infer dataset_format from extension {dataset_path.suffix!r} for path "
            f"{dataset_path}. Supported: .csv, .parquet (.pq). Pass dataset_format= "
            "explicitly to override."
        )
    return fmt


def _full_uri(bucket: str, prefix: str, logical_key: str) -> str:
    prefix = prefix.strip("/")
    key = f"{prefix}/{logical_key}" if prefix else logical_key
    return f"s3://{bucket}/{key}"


def publish_trigger(
    *,
    dataset_path: str | Path,
    params_path: str | Path,
    category: str,
    project: str,
    model_name: str,
    model_family: str,
    bucket: str,
    prefix: str,
    auto_promote: bool,
    store: ArtifactStore,
    orchestrator: OrchestrationAdapter,
    description: str = "",
    requested_by: str = "",
    dataset_format: str | None = None,
) -> tuple[str, str]:
    """Push a trigger folder via the storage port, then dispatch via the
    orchestration port. Returns (trigger_id, trigger_uri).

    Layout: ``s3://{bucket}/[<prefix>/]_triggers/{project}/{trigger_id}/...``
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trigger_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    dataset_path = Path(dataset_path)
    params_path = Path(params_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"params_path does not exist: {params_path}")

    fmt = _infer_dataset_format(dataset_path, dataset_format)

    dataset_key = trigger_dataset_key(prefix, project, trigger_id, fmt)
    params_key = trigger_params_key(prefix, project, trigger_id)
    metadata_key = trigger_metadata_key(prefix, project, trigger_id)

    metadata = TriggerFile(
        trigger_id=trigger_id,
        category=category,
        project=project,
        model_name=model_name,
        model_family=model_family,
        dataset_uri=_full_uri(bucket, prefix, dataset_key),
        params_uri=_full_uri(bucket, prefix, params_key),
        dataset_format=fmt,
        requested_by=requested_by,
        description=description,
    )

    # Order is load-bearing: dataset + params first, trigger.json LAST. The
    # puller treats trigger.json as the completion marker — its presence
    # guarantees the other two keys are already in place. On any upload
    # failure delete already-uploaded keys so the trigger folder does not
    # linger as orphaned partial data.
    uploaded: list[str] = []
    content_type = "text/csv" if fmt == "csv" else "application/octet-stream"
    try:
        store.upload_file(bucket, dataset_path, dataset_key, content_type=content_type)
        uploaded.append(dataset_key)
        store.upload_file(bucket, params_path, params_key, content_type="application/x-yaml")
        uploaded.append(params_key)
        store.put_bytes(
            bucket,
            json.dumps(metadata.to_dict(), indent=2).encode("utf-8"),
            metadata_key,
            content_type="application/json",
        )
        uploaded.append(metadata_key)
    except Exception:
        for key in reversed(uploaded):
            try:
                store.delete(bucket, key)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean up orphaned key {}: {}", key, cleanup_exc)
        raise

    prefix_part = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    trigger_uri = f"s3://{bucket}/{prefix_part}_triggers/{project}/{trigger_id}/"
    logger.info(
        "Published trigger {} ({}/{}, format={}) -> {}",
        trigger_id,
        project,
        model_name,
        fmt,
        trigger_uri,
    )

    # Hand off to the orchestrator. If it refuses, write a failed.json marker
    # so /trigger-status/<id> reports "failed" instead of hanging in "pending".
    try:
        orchestrator.dispatch_training(
            trigger_id=trigger_id,
            category=category,
            project=project,
            model_name=model_name,
            bucket=bucket,
            prefix=prefix,
            auto_promote=auto_promote,
        )
    except RuntimeError as dispatch_exc:
        failure_body = json.dumps(
            {
                "status": "failed",
                "reason": f"orchestrator dispatch failed: {dispatch_exc}",
                "trigger_id": trigger_id,
            }
        ).encode("utf-8")
        try:
            store.put_bytes(
                bucket,
                failure_body,
                trigger_failure_key(prefix, project, trigger_id),
                content_type="application/json",
            )
        except Exception as marker_exc:
            logger.warning(
                "Could not write dispatch-failure marker for trigger {}: {}",
                trigger_id,
                marker_exc,
            )
        raise

    return trigger_id, trigger_uri
