"""Storage ports for the inference gateway.

``ReadOnlyArtifactStore`` is what the prediction path uses — it only resolves
pointers, downloads pickles, and discovers models.

``ArtifactStore`` extends it with write methods used by the trigger path
(``/trigger-train`` uploads dataset + params + trigger.json to S3 before
firing the orchestrator). Adapters implement these to swap S3 for GCS/Azure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path


class ReadOnlyArtifactStore(ABC):
    @abstractmethod
    def get_json(self, bucket: str, logical_key: str) -> dict | None:
        """Return parsed JSON at (bucket, logical_key), or None if absent."""

    @abstractmethod
    def download_file(self, bucket: str, logical_key: str, local_path: Path | str) -> None:
        """Download (bucket, logical_key) to local_path."""

    @abstractmethod
    def list_subkeys(self, bucket: str, prefix: str) -> Iterator[str]:
        """Yield immediate child names (one level deep) under (bucket, prefix)."""


class ArtifactStore(ReadOnlyArtifactStore):
    @abstractmethod
    def upload_file(
        self,
        bucket: str,
        local_path: Path | str,
        logical_key: str,
        *,
        content_type: str | None = None,
    ) -> None:
        """Upload local_path to (bucket, logical_key). Atomic from caller's POV."""

    @abstractmethod
    def put_bytes(
        self,
        bucket: str,
        data: bytes,
        logical_key: str,
        *,
        content_type: str | None = None,
    ) -> None:
        """Write data to (bucket, logical_key). Used for trigger.json marker."""

    @abstractmethod
    def delete(self, bucket: str, logical_key: str) -> None:
        """Best-effort delete. Used to clean up orphans on partial-upload failure."""
