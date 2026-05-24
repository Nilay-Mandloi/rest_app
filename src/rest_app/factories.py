"""Adapter factories — the only file that names concrete adapters.

Selection:
    STORAGE_BACKEND  (default: "s3")
"""

from __future__ import annotations

import os

from rest_app.config import Settings
from rest_app.ports.storage import ReadOnlyArtifactStore


def _storage_backend() -> str:
    return os.environ.get("STORAGE_BACKEND", "s3").strip().lower() or "s3"


def get_artifact_store(settings: Settings) -> ReadOnlyArtifactStore:
    backend = _storage_backend()
    if backend == "s3":
        from rest_app.adapters.s3_store import S3ReadStore

        return S3ReadStore(region=settings.region)
    raise ValueError(
        f"Unknown STORAGE_BACKEND='{backend}'. Supported: s3. "
        "Add an adapter under adapters/<backend>_store.py and a branch here."
    )
