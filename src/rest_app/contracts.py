from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PointerFile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    project: str
    model_name: str
    version: int
    version_id: str
    run_id: str
    registry_version: str
    manifest_uri: str
    status: str
    updated_at: str = ""
    mlflow_tracking_uri: str | None = None
    mlflow_run_url: str | None = None
    mlflow_model_url: str | None = None
    promoted_at: str | None = None
    promoted_by: str | None = None
    schema_version: str = "1.0"


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    project: str
    model_name: str
    version: int
    run_id: str
    registry_version: str
    model_type: str
    schema_hash: str
    artifact_checksums: dict[str, str]
    schema_contract: dict[str, Any] = Field(default_factory=dict)
    published_at: str = ""
    mlflow_tracking_uri: str | None = None
    mlflow_run_url: str | None = None
    mlflow_model_url: str | None = None
    git_commit: str | None = None
    code_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    schema_version: str = "1.0"
