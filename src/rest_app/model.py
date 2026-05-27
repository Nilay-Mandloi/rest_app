"""Loads the single model baked into the Docker image at build time.

The prediction path is entirely storage-agnostic: it reads local files written
during ``docker build`` and never touches S3 (or any cloud store) at runtime.
To swap storage backends (S3 → GCS), only the CI step that generates the
presigned download URL changes — this file stays identical.
"""

from __future__ import annotations

import json
import pickle  # noqa: S403
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PKL_PATH = "/app/model.pkl"
DEFAULT_MANIFEST_PATH = "/app/manifest.json"


@dataclass
class BakedModel:
    obj: Any
    feature_columns: list[str]
    version_id: str
    model_type: str
    run_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    category: str = ""
    project: str = ""
    model_name: str = ""


def load_baked_model(
    pkl_path: str = DEFAULT_PKL_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> BakedModel:
    with Path(manifest_path).open() as f:
        manifest = json.load(f)
    sc = manifest.get("schema_contract") or {}
    cols = sc.get("feature_columns") or []
    version = manifest.get("version", 0)
    with Path(pkl_path).open("rb") as f:
        obj = pickle.load(f)  # noqa: S301
    return BakedModel(
        obj=obj,
        feature_columns=[str(c) for c in cols],
        version_id=f"v{version}",
        model_type=manifest.get("model_type", ""),
        run_id=manifest.get("run_id", ""),
        metrics=manifest.get("metrics") or {},
        category=manifest.get("category", ""),
        project=manifest.get("project", ""),
        model_name=manifest.get("model_name", ""),
    )
