"""Verify rest_app's vendored trigger contract matches the mlops source of truth.

mlops owns the canonical trigger contract (``schemas/trigger.v1.json`` plus the
``TriggerFile`` dataclass). rest_app produces ``trigger.json`` from a pinned copy
and must not drift from it. This test lives on the consumer side: rest_app
checks itself against the producer, so mlops has zero awareness of rest_app.

The JSON Schema must match byte-for-byte. The Python class *shape* is allowed
to diverge in implementation (rest_app uses Pydantic, mlops uses a dataclass);
only the field NAMES and required/optional split must agree.

mlops is located via the MLOPS_ROOT environment variable, or as a sibling
checkout (``../mlops``). The test skips when neither is available (e.g. CI that
checks out only this repo).

To run locally:
    pytest tests/test_contract_sync.py
or, if the repos are not siblings:
    MLOPS_ROOT=/path/to/mlops pytest tests/test_contract_sync.py

To manually sync after a contract change in mlops:
    cp ../mlops/schemas/trigger.v1.json src/rest_app/schemas/trigger.v1.json
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from rest_app.contracts import TriggerFile

_REST_ROOT = Path(__file__).parents[1]

_MLOPS_ROOT: Path | None = None
if os.environ.get("MLOPS_ROOT"):
    _MLOPS_ROOT = Path(os.environ["MLOPS_ROOT"])
else:
    candidate = _REST_ROOT.parent / "mlops"
    if candidate.is_dir():
        _MLOPS_ROOT = candidate

_SKIP_MLOPS = pytest.mark.skipif(
    _MLOPS_ROOT is None,
    reason="mlops repo not found. Set MLOPS_ROOT or place repos as siblings.",
)

_SCHEMA_NAMES = ("trigger.v1.json",)


@_SKIP_MLOPS
@pytest.mark.parametrize("schema_name", _SCHEMA_NAMES)
def test_vendored_schema_matches_mlops(schema_name):
    """rest_app's packaged schema (src/rest_app/schemas/, shipped in the wheel /
    Docker image) must stay byte-identical with mlops's source-of-truth schemas/."""
    vendored = (_REST_ROOT / "src" / "rest_app" / "schemas" / schema_name).read_bytes()
    canonical = (_MLOPS_ROOT / "schemas" / schema_name).read_bytes()
    assert vendored == canonical, (
        f"src/rest_app/schemas/{schema_name} has drifted from mlops.\n"
        "Re-sync with:\n"
        f"  cp {_MLOPS_ROOT}/schemas/{schema_name} "
        f"src/rest_app/schemas/{schema_name}"
    )


def _pydantic_fields(model_cls) -> dict[str, bool]:
    """Return {field_name: is_required} for a Pydantic v2 model."""
    return {name: info.is_required() for name, info in model_cls.model_fields.items()}


def _dataclass_fields(cls) -> dict[str, bool]:
    """Return {field_name: is_required} for a frozen dataclass."""
    from dataclasses import MISSING

    return {
        f.name: (f.default is MISSING and f.default_factory is MISSING)
        for f in cls.__dataclass_fields__.values()
    }


@_SKIP_MLOPS
@pytest.mark.parametrize("class_name", ["TriggerFile"])
def test_python_class_field_parity(class_name):
    """rest_app's Pydantic model must have the same field NAMES (and the same
    required/optional split) as mlops's dataclass.

    mlops's contracts.py is loaded from disk (stdlib-only, no install needed)
    so this test does not require mlops to be installed in rest_app's venv.
    """
    spec = importlib.util.spec_from_file_location(
        "_mlops_contracts",
        _MLOPS_ROOT / "src" / "quantity_forecast" / "contracts.py",
    )
    assert spec and spec.loader, "could not load mlops contracts.py"
    mlops_mod = importlib.util.module_from_spec(spec)
    sys.modules["_mlops_contracts"] = mlops_mod
    try:
        spec.loader.exec_module(mlops_mod)
    except ImportError as exc:
        pytest.skip(f"mlops contracts imports unavailable: {exc}")

    rest_cls = TriggerFile
    mlops_cls = getattr(mlops_mod, class_name)

    rest_fields = _pydantic_fields(rest_cls)
    mlops_fields = _dataclass_fields(mlops_cls)

    assert set(rest_fields) == set(mlops_fields), (
        f"{class_name} field-NAME drift between rest_app and mlops.\n"
        f"  rest_app only: {sorted(set(rest_fields) - set(mlops_fields))}\n"
        f"  mlops only: {sorted(set(mlops_fields) - set(rest_fields))}"
    )

    required_mismatches = [name for name in rest_fields if rest_fields[name] != mlops_fields[name]]
    assert not required_mismatches, (
        f"{class_name} required/optional drift between rest_app and mlops: {required_mismatches}"
    )
