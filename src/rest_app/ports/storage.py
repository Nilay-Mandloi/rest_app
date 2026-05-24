"""Read-only storage port for the inference gateway.

The gateway never writes — it only resolves pointers, downloads pickles,
and discovers models. Adapters implement this to swap S3 for GCS/Azure.
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
