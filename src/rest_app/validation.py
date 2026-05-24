from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def _schemas_dir() -> Path:
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        cand = ancestor / "schemas"
        if cand.is_dir() and (cand / "pointer.v1.json").exists():
            return cand
    raise FileNotFoundError("schemas/ directory not found relative to rest_app package")


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
