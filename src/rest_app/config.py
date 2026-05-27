from __future__ import annotations

import os
import re
from dataclasses import dataclass

CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")
PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
VERSION_ID_RE = re.compile(r"^v[1-9][0-9]*$")


@dataclass(frozen=True)
class Settings:
    # S3 / AWS — boto3 picks AWS_ACCESS_KEY_ID/SECRET/REGION from the
    # standard credential chain. We never read keys explicitly.
    bucket_override: str  # if empty, derive {category}-artifacts per-request
    prefix: str
    region: str
    # Admin
    admin_token: str
    # Serving
    host: str
    port: int
    max_batch_size: int
    # Baked-model paths (written into the image at docker build time)
    model_pkl_path: str
    manifest_path: str
    # Training trigger (/trigger-train). Both empty => endpoint returns 503.
    training_repo: str
    training_repo_token: str
    training_auto_promote: bool
    # Bound on dataset upload size (bytes). Default 100 MB.
    max_dataset_bytes: int

    def bucket_for(self, category: str) -> str:
        return self.bucket_override or f"{category}-artifacts"

    @classmethod
    def from_env(cls) -> Settings:
        bucket_override = os.environ.get("ARTIFACT_STORE_BUCKET", "").strip()
        prefix = os.environ.get("ARTIFACT_STORE_PREFIX", "").strip("/")
        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        admin_token = os.environ.get("APP_ADMIN_TOKEN", "").strip()
        host = os.environ.get("APP_HOST", "0.0.0.0")  # noqa: S104
        port = int(os.environ.get("APP_PORT", "8000"))
        max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "1000"))
        if max_batch_size < 1:
            raise ValueError("MAX_BATCH_SIZE must be >= 1")
        model_pkl_path = os.environ.get("MODEL_PKL_PATH", "/app/model.pkl").strip()
        manifest_path = os.environ.get("MANIFEST_PATH", "/app/manifest.json").strip()
        training_repo = os.environ.get("GITHUB_TRAINING_REPO", "").strip()
        training_repo_token = os.environ.get("GITHUB_PAT", "").strip()
        training_auto_promote = os.environ.get("TRAINING_AUTO_PROMOTE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        max_dataset_bytes = int(os.environ.get("MAX_DATASET_BYTES", str(100 * 1024 * 1024)))
        if max_dataset_bytes < 1:
            raise ValueError("MAX_DATASET_BYTES must be >= 1")
        return cls(
            bucket_override=bucket_override,
            prefix=prefix,
            region=region,
            admin_token=admin_token,
            host=host,
            port=port,
            max_batch_size=max_batch_size,
            model_pkl_path=model_pkl_path,
            manifest_path=manifest_path,
            training_repo=training_repo,
            training_repo_token=training_repo_token,
            training_auto_promote=training_auto_promote,
            max_dataset_bytes=max_dataset_bytes,
        )
