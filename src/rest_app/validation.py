from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"


def _schemas_dir() -> Path:
    """JSON Schemas ship inside the package (src/rest_app/schemas/) so they
    are available whether rest_app is run from a source checkout or an
    installed wheel / Docker image."""
    if not _SCHEMAS_DIR.is_dir() or not (_SCHEMAS_DIR / "pointer.v1.json").exists():
        raise FileNotFoundError(
            f"schemas/ directory missing from rest_app package at {_SCHEMAS_DIR}. "
            "If running from source, ensure src/rest_app/schemas/*.json exist; "
            "if installed, the wheel was built without package-data — check pyproject.toml."
        )
    return _SCHEMAS_DIR


@cache
def _load(name: str) -> dict[str, Any]:
    return json.loads((_schemas_dir() / name).read_text(encoding="utf-8"))


@cache
def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(_load(name))


def validate_pointer(data: dict[str, Any]) -> None:
    errs = sorted(_validator("pointer.v1.json").iter_errors(data), key=lambda e: e.path)
    if errs:
        raise ValueError(
            "pointer.json validation failed: "
            + "; ".join(f"{list(e.path)}: {e.message}" for e in errs)
        )


def validate_manifest(data: dict[str, Any]) -> None:
    errs = sorted(_validator("manifest.v1.json").iter_errors(data), key=lambda e: e.path)
    if errs:
        raise ValueError(
            "manifest.json validation failed: "
            + "; ".join(f"{list(e.path)}: {e.message}" for e in errs)
        )
