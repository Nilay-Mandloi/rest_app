from __future__ import annotations

import re

from rest_app.config import MODEL_NAME_RE, PROJECT_RE


def _check(value: str, pattern: re.Pattern[str], name: str) -> str:
    if not pattern.match(value):
        raise ValueError(f"{name} must match {pattern.pattern}; got {value!r}")
    return value


def _vtag(version: int | str) -> str:
    s = str(version).lstrip("v")
    if not s.isdigit() or int(s) < 1:
        raise ValueError(f"version must be a positive integer; got {version!r}")
    return f"v{int(s)}"


def _join(prefix: str, *parts: str) -> str:
    pieces = [p for p in (prefix.strip("/"), *parts) if p]
    return "/".join(pieces)


def model_root(prefix: str, project: str, model_name: str) -> str:
    return _join(
        prefix,
        _check(project, PROJECT_RE, "project"),
        _check(model_name, MODEL_NAME_RE, "model_name"),
    )


def pointer_key(prefix: str, project: str, model_name: str, channel: str) -> str:
    if not channel or "/" in channel or channel.startswith("_"):
        raise ValueError(f"channel must be a simple name; got {channel!r}")
    return f"{model_root(prefix, project, model_name)}/{channel}.json"


def artifact_key(
    prefix: str, project: str, model_name: str, version: int | str, filename: str
) -> str:
    return f"{model_root(prefix, project, model_name)}/{_vtag(version)}/{filename}"


def model_pkl_key(prefix: str, project: str, model_name: str, version: int | str) -> str:
    return artifact_key(prefix, project, model_name, version, "model.pkl")


def manifest_key(prefix: str, project: str, model_name: str, version: int | str) -> str:
    return artifact_key(prefix, project, model_name, version, "manifest.json")


def project_prefix(prefix: str, project: str) -> str:
    """Top-level list prefix for discovery: '{prefix?}/{project}/'."""
    return _join(prefix, _check(project, PROJECT_RE, "project")) + "/"


TRIGGER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SUPPORTED_DATASET_FORMATS: frozenset[str] = frozenset({"csv", "parquet"})


def _check_trigger_id(trigger_id: str) -> str:
    return _check(trigger_id, TRIGGER_ID_RE, "trigger_id")


def _check_dataset_format(fmt: str) -> str:
    if fmt not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(
            f"dataset_format must be one of {sorted(SUPPORTED_DATASET_FORMATS)}; got {fmt!r}"
        )
    return fmt


def trigger_root(prefix: str, project: str, trigger_id: str) -> str:
    return _join(
        prefix,
        "_triggers",
        _check(project, PROJECT_RE, "project"),
        _check_trigger_id(trigger_id),
    )


def trigger_dataset_key(
    prefix: str, project: str, trigger_id: str, dataset_format: str = "parquet"
) -> str:
    fmt = _check_dataset_format(dataset_format)
    return f"{trigger_root(prefix, project, trigger_id)}/dataset.{fmt}"


def trigger_params_key(prefix: str, project: str, trigger_id: str) -> str:
    return f"{trigger_root(prefix, project, trigger_id)}/params.yaml"


def trigger_metadata_key(prefix: str, project: str, trigger_id: str) -> str:
    return f"{trigger_root(prefix, project, trigger_id)}/trigger.json"


def trigger_running_key(prefix: str, project: str, trigger_id: str) -> str:
    return f"{trigger_root(prefix, project, trigger_id)}/running.json"


def trigger_failure_key(prefix: str, project: str, trigger_id: str) -> str:
    return f"{trigger_root(prefix, project, trigger_id)}/failed.json"


def trigger_completed_key(prefix: str, project: str, trigger_id: str) -> str:
    return f"{trigger_root(prefix, project, trigger_id)}/completed.json"
