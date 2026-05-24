from __future__ import annotations

import re

PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


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
