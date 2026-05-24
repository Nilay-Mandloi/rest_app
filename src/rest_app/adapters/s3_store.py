"""S3 implementation of ReadOnlyArtifactStore. The only file in this package
that imports boto3.

The gateway resolves models for multiple tenants, so each port method takes
``bucket`` explicitly rather than baking it into the adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from rest_app.ports.storage import ReadOnlyArtifactStore


class S3ReadStore(ReadOnlyArtifactStore):
    def __init__(self, *, region: str | None = None, client: Any | None = None) -> None:
        self._client = client or boto3.client("s3", region_name=region)

    @property
    def raw_client(self) -> Any:
        """Used by tests / discovery that need to inspect bucket-level errors
        with backend-specific exception types."""
        return self._client

    def get_json(self, bucket: str, logical_key: str) -> dict | None:
        try:
            resp = self._client.get_object(Bucket=bucket, Key=logical_key)
        except self._client.exceptions.NoSuchKey:
            return None
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return json.loads(resp["Body"].read())

    def download_file(self, bucket: str, logical_key: str, local_path: Path | str) -> None:
        self._client.download_file(bucket, logical_key, str(local_path))

    def list_subkeys(self, bucket: str, prefix: str) -> Iterator[str]:
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []) or []:
                key_dir = cp.get("Prefix", "").rstrip("/")
                if not key_dir:
                    continue
                yield key_dir.rsplit("/", 1)[-1]
