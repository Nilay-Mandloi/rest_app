"""Adapter factories — the only file that names concrete adapters.

Selection:
    STORAGE_BACKEND        (default: "s3")
    ORCHESTRATION_BACKEND  (default: "github" if GITHUB_TRAINING_REPO+GITHUB_PAT
                             configured, else "noop")
"""

from __future__ import annotations

import os

from rest_app.config import Settings
from rest_app.ports.orchestration import OrchestrationAdapter
from rest_app.ports.storage import ArtifactStore, ReadOnlyArtifactStore


def _storage_backend() -> str:
    return os.environ.get("STORAGE_BACKEND", "s3").strip().lower() or "s3"


def _orchestration_backend(settings: Settings) -> str:
    raw = os.environ.get("ORCHESTRATION_BACKEND", "").strip().lower()
    if raw:
        return raw
    if settings.training_repo and settings.training_repo_token:
        return "github"
    return "noop"


def get_artifact_store(settings: Settings) -> ReadOnlyArtifactStore:
    """Read-only store used by the prediction path."""
    backend = _storage_backend()
    if backend == "s3":
        from rest_app.adapters.s3_store import S3ReadStore

        return S3ReadStore(region=settings.region)
    raise ValueError(
        f"Unknown STORAGE_BACKEND='{backend}'. Supported: s3. "
        "Add an adapter under adapters/<backend>_store.py and a branch here."
    )


def get_writable_artifact_store(settings: Settings) -> ArtifactStore:
    """Read+write store used by the trigger path."""
    backend = _storage_backend()
    if backend == "s3":
        from rest_app.adapters.s3_store import S3Store

        return S3Store(region=settings.region)
    raise ValueError(
        f"Unknown STORAGE_BACKEND='{backend}'. Supported: s3. "
        "Add an adapter under adapters/<backend>_store.py and a branch here."
    )


def get_orchestrator(settings: Settings) -> OrchestrationAdapter:
    backend = _orchestration_backend(settings)
    if backend == "github":
        from rest_app.adapters.github_dispatch import GitHubDispatchAdapter

        return GitHubDispatchAdapter(
            training_repo=settings.training_repo,
            training_repo_token=settings.training_repo_token,
            training_branch=settings.training_branch,
        )
    if backend == "noop":
        from rest_app.adapters.github_dispatch import NoopDispatchAdapter

        return NoopDispatchAdapter()
    raise ValueError(
        f"Unknown ORCHESTRATION_BACKEND='{backend}'. Supported: github, noop. "
        "Add an adapter under adapters/<backend>_dispatch.py and a branch here."
    )
