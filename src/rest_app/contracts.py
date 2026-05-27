from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MODEL_FAMILIES = Literal[
    "regression",
    "classification",
    "forecasting",
    "clustering",
    "ranking",
    "nlp",
    "vision",
    "other",
]


class TriggerFile(BaseModel):
    """Marker payload written LAST when publishing a training trigger.

    Its presence in S3 signals to the puller (training repo) that dataset.*
    and params.yaml are already in place. Schema is byte-shared with mlops
    via the canonical schemas/trigger.v1.json — keep field names + types
    in sync. Field validators here enforce the same constraints the JSON
    schema does, so bad triggers fail at request time (HTTP 400/422) rather
    than silently producing trigger.json files that the training side rejects.
    """

    model_config = ConfigDict(extra="forbid")

    trigger_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    category: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,30}$")
    project: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")
    model_name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,62}$")
    model_family: _MODEL_FAMILIES
    dataset_uri: str = Field(pattern=r"^s3://[^/]+/.+\.(csv|parquet)$")
    params_uri: str = Field(pattern=r"^s3://[^/]+/.+\.ya?ml$")
    dataset_format: Literal["csv", "parquet"] = "parquet"
    requested_by: str = ""
    created_at: str = ""
    description: str = ""
    schema_version: Literal["1.0"] = "1.0"

    @field_validator("created_at")
    @classmethod
    def _default_created_at(cls, v: str) -> str:
        return v or datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
