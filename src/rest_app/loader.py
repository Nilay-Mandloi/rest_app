from __future__ import annotations

import hashlib
import pickle  # noqa: S403
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from .config import Settings
from .contracts import ArtifactManifest, PointerFile
from .layout import manifest_key, model_pkl_key, pointer_key
from .ports.storage import ReadOnlyArtifactStore
from .validation import validate_manifest, validate_pointer


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LoadedModel:
    obj: Any
    pointer: PointerFile | None  # None when loaded by explicit version (no pointer involved)
    manifest: ArtifactManifest
    loaded_at: str
    # The cache key components, surfaced for /model/info responses.
    category: str
    project: str
    model_name: str
    version: int
    version_id: str


CacheKey = tuple[str, str, str, str]  # (category, project, model_name, version_id)


class ModelCache:
    """LRU cache of loaded models keyed by (category, project, model_name, version_id).

    Channel-based requests resolve the channel to a version via the pointer file,
    then hit this cache. Version-pinned requests skip the pointer entirely.
    Bounded eviction prevents unbounded memory growth as projects accumulate.

    Backend-neutral: takes a ReadOnlyArtifactStore at construction; no S3 SDK
    imports here.
    """

    def __init__(self, settings: Settings, store: ReadOnlyArtifactStore | None = None) -> None:
        self._settings = settings
        if store is None:
            from .factories import get_artifact_store

            store = get_artifact_store(settings)
        self._store = store
        self._lock = threading.Lock()
        self._entries: OrderedDict[CacheKey, LoadedModel] = OrderedDict()

    # ------------------------------------------------------------------
    # Cache introspection
    # ------------------------------------------------------------------

    def list_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "category": k[0],
                    "project": k[1],
                    "model_name": k[2],
                    "version_id": k[3],
                    "loaded_at": v.loaded_at,
                }
                for k, v in self._entries.items()
            ]

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            return n

    def evict(self, category: str, project: str, model_name: str, version_id: str) -> bool:
        with self._lock:
            return self._entries.pop((category, project, model_name, version_id), None) is not None

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def resolve_and_load(
        self,
        *,
        category: str,
        project: str,
        model_name: str,
        version_id: str | None = None,
        channel: str | None = None,
    ) -> LoadedModel:
        """Load by exact version_id, OR resolve channel -> version then load."""
        if version_id:
            return self._load_version(
                category=category,
                project=project,
                model_name=model_name,
                version_id=version_id,
                pointer=None,
            )
        if not channel:
            channel = self._settings.default_channel
        pointer = self._read_pointer(category, project, model_name, channel)
        return self._load_version(
            category=category,
            project=project,
            model_name=model_name,
            version_id=pointer.version_id,
            pointer=pointer,
        )

    # ------------------------------------------------------------------
    # Storage shortcuts
    # ------------------------------------------------------------------

    def _bucket(self, category: str) -> str:
        return self._settings.bucket_for(category)

    @property
    def store(self) -> ReadOnlyArtifactStore:
        """Exposed for endpoints (e.g. /projects/.../models discovery)."""
        return self._store

    def _read_pointer(
        self, category: str, project: str, model_name: str, channel: str
    ) -> PointerFile:
        bucket = self._bucket(category)
        key = pointer_key(self._settings.prefix, project, model_name, channel)
        raw = self._store.get_json(bucket, key)
        if raw is None:
            raise FileNotFoundError(
                f"pointer not found at s3://{bucket}/{key} — has any model been promoted for "
                f"{category}/{project}/{model_name} on channel {channel!r}?"
            )
        validate_pointer(raw)
        pointer = PointerFile.model_validate(raw)
        if (
            pointer.category != category
            or pointer.project != project
            or pointer.model_name != model_name
        ):
            raise RuntimeError(
                f"pointer tenant mismatch at s3://{bucket}/{key}: "
                f"expected {category}/{project}/{model_name}; "
                f"got {pointer.category}/{pointer.project}/{pointer.model_name}"
            )
        return pointer

    # ------------------------------------------------------------------
    # Cache miss path: download + verify + pickle.load
    # ------------------------------------------------------------------

    def _load_version(
        self,
        *,
        category: str,
        project: str,
        model_name: str,
        version_id: str,
        pointer: PointerFile | None,
    ) -> LoadedModel:
        key: CacheKey = (category, project, model_name, version_id)

        # Fast path: cache hit. Move-to-end keeps LRU semantics.
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing

        # Slow path: download + verify + pickle.load without holding the cache
        # lock, so concurrent loads of different models don't serialise.
        loaded = self._download_and_verify(
            category=category,
            project=project,
            model_name=model_name,
            version_id=version_id,
            pointer=pointer,
        )

        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                self._entries.move_to_end(key)
                return existing
            self._entries[key] = loaded
            self._entries.move_to_end(key)
            while len(self._entries) > self._settings.cache_max_entries:
                evicted_key, _ = self._entries.popitem(last=False)
                logger.info(f"cache evicted: {evicted_key}")
        return loaded

    def _download_and_verify(
        self,
        *,
        category: str,
        project: str,
        model_name: str,
        version_id: str,
        pointer: PointerFile | None,
    ) -> LoadedModel:
        bucket = self._bucket(category)
        version_int = int(version_id.lstrip("v"))

        m_key = manifest_key(self._settings.prefix, project, model_name, version_int)
        man_raw = self._store.get_json(bucket, m_key)
        if man_raw is None:
            raise FileNotFoundError(
                f"manifest not found at s3://{bucket}/{m_key} — was this version published?"
            )
        validate_manifest(man_raw)
        manifest = ArtifactManifest.model_validate(man_raw)

        if (
            manifest.category != category
            or manifest.project != project
            or manifest.model_name != model_name
        ):
            raise RuntimeError(
                f"manifest tenant mismatch at s3://{bucket}/{m_key}: "
                f"expected {category}/{project}/{model_name}; "
                f"got {manifest.category}/{manifest.project}/{manifest.model_name}"
            )
        if manifest.version != version_int:
            raise RuntimeError(
                f"manifest/version mismatch: requested v{version_int}; "
                f"manifest says v{manifest.version}"
            )
        if pointer is not None and manifest.version != pointer.version:
            raise RuntimeError(
                f"pointer/manifest version mismatch: pointer={pointer.version}, "
                f"manifest={manifest.version}"
            )

        expected_sha = manifest.artifact_checksums.get("model.pkl")
        if not expected_sha:
            raise RuntimeError(
                f"manifest at s3://{bucket}/{m_key} missing checksum entry for 'model.pkl'"
            )

        pkl_key = model_pkl_key(self._settings.prefix, project, model_name, version_int)
        with tempfile.TemporaryDirectory(prefix="rest_app_pkl_") as tmp_dir:
            tmp_path = f"{tmp_dir}/model.pkl"
            logger.info(f"downloading s3://{bucket}/{pkl_key} -> {tmp_path}")
            self._store.download_file(bucket, pkl_key, tmp_path)

            actual_sha = _sha256_file(tmp_path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"REFUSING TO SERVE: model.pkl checksum mismatch for "
                    f"{category}/{project}/{model_name} v{version_int} "
                    f"(expected {expected_sha}, got {actual_sha})"
                )

            with open(tmp_path, "rb") as fh:
                obj = pickle.load(fh)  # noqa: S301

        loaded = LoadedModel(
            obj=obj,
            pointer=pointer,
            manifest=manifest,
            loaded_at=datetime.now(UTC).isoformat(),
            category=category,
            project=project,
            model_name=model_name,
            version=version_int,
            version_id=version_id,
        )
        logger.info(
            f"loaded {category}/{project}/{model_name} {version_id} (run_id={manifest.run_id})"
        )
        return loaded
