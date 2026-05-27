from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from . import factories
from .config import CATEGORY_RE, MODEL_NAME_RE, PROJECT_RE, VERSION_ID_RE, Settings
from .layout import (
    manifest_key,
    model_pkl_key,
    model_root,
    pointer_key,
    project_prefix,
    trigger_completed_key,
    trigger_failure_key,
    trigger_metadata_key,
    trigger_running_key,
)
from .model import BakedModel, load_baked_model
from .ports.orchestration import OrchestrationAdapter
from .ports.storage import ArtifactStore, ReadOnlyArtifactStore
from .publisher import publish_trigger

_STATIC_DIR = Path(__file__).parent / "static"


class PredictRequest(BaseModel):
    features: dict[str, Any]


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)


def _to_matrix(rows: list[dict[str, Any]], columns: list[str]) -> list[list[Any]]:
    return [[r.get(c) for c in columns] for r in rows]


def _predict(model: Any, X: list[list[Any]], columns: list[str]) -> list[Any]:
    frame = pd.DataFrame(X, columns=columns)
    out = model.predict(frame)
    try:
        return list(out)
    except TypeError:
        return [out]


def _s3_list_error(exc: Exception, bucket: str, prefix: str) -> HTTPException:
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code", "")
    if code == "NoSuchBucket":
        return HTTPException(status_code=404, detail=f"bucket not found: {bucket}")
    status = 403 if code in ("AccessDenied", "AllAccessDisabled") else 502
    return HTTPException(
        status_code=status,
        detail=f"S3 list failed for s3://{bucket}/{prefix}: {code or exc}",
    )


def _check_admin_token(settings: Settings, token: str | None) -> None:
    if not settings.admin_token:
        raise HTTPException(status_code=501, detail="admin disabled (APP_ADMIN_TOKEN not set)")
    if not token or token != settings.admin_token:
        raise HTTPException(status_code=401, detail="invalid or missing admin token")


def create_app(
    settings: Settings | None = None,
    model: BakedModel | None = None,
    store: ReadOnlyArtifactStore | None = None,
    writable_store: ArtifactStore | None = None,
    orchestrator: OrchestrationAdapter | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        if model is not None:
            _app.state.model = model
            logger.info(f"using injected model: {model.version_id}")
        else:
            pkl_path = cfg.model_pkl_path
            manifest_path = cfg.manifest_path
            if Path(pkl_path).exists():
                try:
                    _app.state.model = load_baked_model(pkl_path, manifest_path)
                    m = _app.state.model
                    logger.info(
                        f"loaded baked model: {m.version_id} "
                        f"({m.project}/{m.model_name}) run={m.run_id}"
                    )
                except Exception as exc:
                    logger.error(f"failed to load model from {pkl_path}: {exc}")
                    _app.state.model = None
            else:
                logger.warning(f"no model at {pkl_path} — /predict will return 503")
                _app.state.model = None
        yield

    app = FastAPI(title="rest_app", version="0.5.0", lifespan=_lifespan)
    app.state.settings = cfg
    # An injected model is live immediately (tests, programmatic use). When none
    # is injected, _lifespan loads the baked model from disk at startup.
    app.state.model = model
    app.state.store = store
    app.state.writable_store = writable_store
    app.state.orchestrator = orchestrator

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def _root() -> RedirectResponse:
            return RedirectResponse(url="/static/trigger.html", status_code=302)

    def get_settings(request: Request) -> Settings:
        return request.app.state.settings

    def get_model(request: Request) -> BakedModel:
        m = request.app.state.model
        if m is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return m

    def get_store(request: Request) -> ReadOnlyArtifactStore:
        if request.app.state.store is None:
            request.app.state.store = factories.get_artifact_store(request.app.state.settings)
        return request.app.state.store

    def get_writable_store(request: Request) -> ArtifactStore:
        if request.app.state.writable_store is None:
            request.app.state.writable_store = factories.get_writable_artifact_store(
                request.app.state.settings
            )
        return request.app.state.writable_store

    def get_orchestrator(request: Request) -> OrchestrationAdapter:
        if request.app.state.orchestrator is None:
            request.app.state.orchestrator = factories.get_orchestrator(request.app.state.settings)
        return request.app.state.orchestrator

    # ------------------------------------------------------------------
    # Health / readiness
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> Any:
        loaded = request.app.state.model is not None
        body: dict[str, Any] = {
            "status": "ready" if loaded else "standby",
            "process_alive": True,
            "model_loaded": loaded,
        }
        if not loaded:
            return JSONResponse(status_code=503, content=body)
        return body

    # ------------------------------------------------------------------
    # Prediction — uses the single baked model, no runtime S3 calls
    # ------------------------------------------------------------------

    @app.post("/predict")
    def predict(
        req: PredictRequest,
        baked: BakedModel = Depends(get_model),
    ) -> dict[str, Any]:
        cols = baked.feature_columns
        if cols:
            missing = [c for c in cols if c not in req.features]
            if missing:
                raise HTTPException(status_code=422, detail=f"missing features: {missing}")
        X = _to_matrix([req.features], cols) if cols else [[v for v in req.features.values()]]
        try:
            preds = _predict(baked.obj, X, cols)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
        return {
            "prediction": preds[0] if preds else None,
            "model": {
                "version_id": baked.version_id,
                "model_type": baked.model_type,
                "run_id": baked.run_id,
            },
        }

    @app.post("/predict/batch")
    def predict_batch(
        req: BatchPredictRequest,
        baked: BakedModel = Depends(get_model),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        if not req.rows:
            return {"predictions": [], "model": {"version_id": baked.version_id}}
        if len(req.rows) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"batch size {len(req.rows)} exceeds MAX_BATCH_SIZE={settings.max_batch_size}"
                ),
            )
        cols = baked.feature_columns
        if cols:
            for i, r in enumerate(req.rows):
                missing = [c for c in cols if c not in r]
                if missing:
                    raise HTTPException(
                        status_code=422, detail=f"row {i} missing features: {missing}"
                    )
        X = _to_matrix(req.rows, cols) if cols else [[v for v in r.values()] for r in req.rows]
        try:
            preds = _predict(baked.obj, X, cols)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
        return {
            "predictions": list(preds),
            "model": {"version_id": baked.version_id},
        }

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    @app.get("/model/info")
    def model_info(baked: BakedModel = Depends(get_model)) -> dict[str, Any]:
        return {
            "version_id": baked.version_id,
            "model_type": baked.model_type,
            "run_id": baked.run_id,
            "category": baked.category,
            "project": baked.project,
            "model_name": baked.model_name,
            "feature_columns": baked.feature_columns,
            "metrics": baked.metrics,
        }

    # ------------------------------------------------------------------
    # Model discovery — reads S3 via the storage port (GCP-swappable)
    # ------------------------------------------------------------------

    @app.get("/projects/{category}/{project}/models")
    def list_models(
        category: str,
        project: str,
        store: ReadOnlyArtifactStore = Depends(get_store),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        if not CATEGORY_RE.match(category):
            raise HTTPException(status_code=400, detail=f"invalid category: {category!r}")
        if not PROJECT_RE.match(project):
            raise HTTPException(status_code=400, detail=f"invalid project: {project!r}")
        bucket = settings.bucket_for(category)
        prefix = project_prefix(settings.prefix, project)

        models: list[dict[str, Any]] = []
        channel_errors: list[dict[str, str]] = []

        def _read_channel(model_name: str, channel: str) -> dict[str, Any] | None:
            key = pointer_key(settings.prefix, project, model_name, channel)
            try:
                data = store.get_json(bucket, key)
            except Exception as exc:
                logger.warning(f"s3://{bucket}/{key} read failed: {exc}")
                channel_errors.append({"key": key, "error": str(exc)})
                return None
            if data is None:
                return None
            return {
                "version_id": data.get("version_id"),
                "version": data.get("version"),
                "run_id": data.get("run_id"),
                "registry_version": data.get("registry_version"),
                "promoted_at": data.get("promoted_at"),
                "updated_at": data.get("updated_at"),
                "mlflow_model_url": data.get("mlflow_model_url"),
            }

        try:
            model_names = list(store.list_subkeys(bucket, prefix))
        except Exception as exc:
            raise _s3_list_error(exc, bucket, prefix) from exc

        for model_name in model_names:
            if not MODEL_NAME_RE.match(model_name):
                continue
            models.append(
                {
                    "model_name": model_name,
                    "stable": _read_channel(model_name, "stable"),
                    "latest": _read_channel(model_name, "latest"),
                }
            )

        result: dict[str, Any] = {
            "category": category,
            "project": project,
            "bucket": bucket,
            "models": models,
        }
        if channel_errors:
            result["channel_errors"] = channel_errors
        return result

    @app.get("/projects/{category}/{project}/models/{model_name}/versions")
    def list_model_versions(
        category: str,
        project: str,
        model_name: str,
        settings: Settings = Depends(get_settings),
        store: ReadOnlyArtifactStore = Depends(get_store),
    ) -> dict[str, Any]:
        if not CATEGORY_RE.match(category):
            raise HTTPException(status_code=400, detail=f"invalid category: {category!r}")
        if not PROJECT_RE.match(project):
            raise HTTPException(status_code=400, detail=f"invalid project: {project!r}")
        if not MODEL_NAME_RE.match(model_name):
            raise HTTPException(status_code=400, detail=f"invalid model_name: {model_name!r}")

        bucket = settings.bucket_for(category)

        channel_version: dict[str, str] = {}
        for ch in ("stable", "latest"):
            try:
                ptr = store.get_json(bucket, pointer_key(settings.prefix, project, model_name, ch))
                if ptr and ptr.get("version_id"):
                    channel_version[ch] = ptr["version_id"]
            except Exception:
                pass

        root_prefix = model_root(settings.prefix, project, model_name) + "/"
        try:
            children = list(store.list_subkeys(bucket, root_prefix))
        except Exception as exc:
            raise _s3_list_error(exc, bucket, root_prefix) from exc

        versions = []
        for ver in children:
            if not VERSION_ID_RE.match(ver):
                continue
            mkey = manifest_key(settings.prefix, project, model_name, ver)
            manifest_data = store.get_json(bucket, mkey)
            if manifest_data is None:
                continue
            sc = manifest_data.get("schema_contract") or {}
            channels = [ch for ch, v in channel_version.items() if v == ver]
            versions.append(
                {
                    "version_id": ver,
                    "version": manifest_data.get("version"),
                    "model_type": manifest_data.get("model_type"),
                    "run_id": manifest_data.get("run_id"),
                    "metrics": manifest_data.get("metrics") or {},
                    "feature_columns": sc.get("feature_columns"),
                    "manifest_uri": f"s3://{bucket}/{mkey}",
                    "model_pkl_uri": (
                        f"s3://{bucket}/{model_pkl_key(settings.prefix, project, model_name, ver)}"
                    ),
                    "channels": channels,
                    "published_at": manifest_data.get("published_at"),
                }
            )

        versions.sort(key=lambda v: v.get("version") or 0, reverse=True)
        return {
            "category": category,
            "project": project,
            "model_name": model_name,
            "bucket": bucket,
            "versions": versions,
        }

    @app.get("/projects/{category}/{project}/models/{model_name}/versions/{version_id}")
    def get_model_version(
        category: str,
        project: str,
        model_name: str,
        version_id: str,
        settings: Settings = Depends(get_settings),
        store: ReadOnlyArtifactStore = Depends(get_store),
    ) -> dict[str, Any]:
        if not CATEGORY_RE.match(category):
            raise HTTPException(status_code=400, detail=f"invalid category: {category!r}")
        if not PROJECT_RE.match(project):
            raise HTTPException(status_code=400, detail=f"invalid project: {project!r}")
        if not MODEL_NAME_RE.match(model_name):
            raise HTTPException(status_code=400, detail=f"invalid model_name: {model_name!r}")
        if not VERSION_ID_RE.match(version_id):
            raise HTTPException(status_code=400, detail=f"invalid version_id: {version_id!r}")

        bucket = settings.bucket_for(category)
        mkey = manifest_key(settings.prefix, project, model_name, version_id)
        manifest_data = store.get_json(bucket, mkey)
        if manifest_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"version {version_id} not found for {project}/{model_name}",
            )

        channel_version: dict[str, str] = {}
        for ch in ("stable", "latest"):
            try:
                ptr = store.get_json(bucket, pointer_key(settings.prefix, project, model_name, ch))
                if ptr and ptr.get("version_id"):
                    channel_version[ch] = ptr["version_id"]
            except Exception:
                pass
        channels = [ch for ch, v in channel_version.items() if v == version_id]

        sc = manifest_data.get("schema_contract") or {}
        return {
            "category": category,
            "project": project,
            "model_name": model_name,
            "version_id": version_id,
            "version": manifest_data.get("version"),
            "model_type": manifest_data.get("model_type"),
            "run_id": manifest_data.get("run_id"),
            "metrics": manifest_data.get("metrics") or {},
            "feature_columns": sc.get("feature_columns"),
            "schema_contract": sc,
            "artifact_checksums": manifest_data.get("artifact_checksums"),
            "manifest_uri": f"s3://{bucket}/{mkey}",
            "model_pkl_uri": (
                f"s3://{bucket}/{model_pkl_key(settings.prefix, project, model_name, version_id)}"
            ),
            "channels": channels,
            "published_at": manifest_data.get("published_at"),
            "mlflow_run_url": manifest_data.get("mlflow_run_url"),
            "mlflow_model_url": manifest_data.get("mlflow_model_url"),
            "bucket": bucket,
        }

    # ------------------------------------------------------------------
    # Training trigger — storage port used here too (GCP-swappable)
    # ------------------------------------------------------------------

    def _validate_category_project(category: str, project: str) -> None:
        if not CATEGORY_RE.match(category):
            raise HTTPException(status_code=400, detail=f"invalid category: {category!r}")
        if not PROJECT_RE.match(project):
            raise HTTPException(status_code=400, detail=f"invalid project: {project!r}")

    def _validate_target(category: str, project: str, model_name: str) -> None:
        _validate_category_project(category, project)
        if not MODEL_NAME_RE.match(model_name):
            raise HTTPException(status_code=400, detail=f"invalid model_name: {model_name!r}")

    @app.post("/trigger-train")
    async def trigger_train(
        category: str = Form(...),
        project: str = Form(...),
        model_name: str = Form(...),
        model_family: str = Form(...),
        dataset: UploadFile = File(...),
        params: UploadFile = File(...),
        dataset_format: str | None = Form(default=None),
        description: str = Form(default=""),
        requested_by: str = Form(default=""),
        auto_promote: bool | None = Form(default=None),
        settings: Settings = Depends(get_settings),
        store: ArtifactStore = Depends(get_writable_store),
        orchestrator: OrchestrationAdapter = Depends(get_orchestrator),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _check_admin_token(settings, x_admin_token)
        if not settings.training_repo or not settings.training_repo_token:
            raise HTTPException(
                status_code=503,
                detail=(
                    "training trigger not configured: set GITHUB_TRAINING_REPO + "
                    "GITHUB_PAT in this service's environment"
                ),
            )
        _validate_target(category, project, model_name)

        bucket = settings.bucket_for(category)
        effective_auto_promote = (
            settings.training_auto_promote if auto_promote is None else auto_promote
        )

        with tempfile.TemporaryDirectory(prefix="trigger_") as tmpdir:
            tmp = Path(tmpdir)
            dataset_local = tmp / (dataset.filename or "dataset.bin")
            params_local = tmp / (params.filename or "params.yaml")

            await _save_capped(dataset, dataset_local, settings.max_dataset_bytes)
            await _save_capped(params, params_local, 1 * 1024 * 1024)

            try:
                trigger_id, trigger_uri = publish_trigger(
                    dataset_path=dataset_local,
                    params_path=params_local,
                    category=category,
                    project=project,
                    model_name=model_name,
                    model_family=model_family,
                    bucket=bucket,
                    prefix=settings.prefix,
                    auto_promote=effective_auto_promote,
                    store=store,
                    orchestrator=orchestrator,
                    description=description,
                    requested_by=requested_by,
                    dataset_format=dataset_format,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=f"dispatch failed: {exc}") from exc

        return {
            "trigger_id": trigger_id,
            "trigger_uri": trigger_uri,
            "status_url": f"/trigger-status/{trigger_id}?project={project}&category={category}",
            "auto_promote": effective_auto_promote,
        }

    @app.get("/trigger-status/{trigger_id}")
    def trigger_status(
        trigger_id: str,
        project: str,
        category: str,
        settings: Settings = Depends(get_settings),
        store: ArtifactStore = Depends(get_writable_store),
    ) -> dict[str, Any]:
        _validate_category_project(category, project)
        bucket = settings.bucket_for(category)

        failed = store.get_json(bucket, trigger_failure_key(settings.prefix, project, trigger_id))
        if failed is not None:
            return {
                "trigger_id": trigger_id,
                "status": "failed",
                "reason": failed.get("reason", ""),
            }

        meta = store.get_json(bucket, trigger_metadata_key(settings.prefix, project, trigger_id))
        completed = store.get_json(
            bucket, trigger_completed_key(settings.prefix, project, trigger_id)
        )
        running = store.get_json(bucket, trigger_running_key(settings.prefix, project, trigger_id))

        if meta is None and running is None and completed is None:
            raise HTTPException(status_code=404, detail=f"no such trigger: {trigger_id}")

        if completed is not None:
            resp: dict[str, Any] = {"trigger_id": trigger_id, "status": "completed"}
            mn = (meta or {}).get("model_name", "")
            if mn:
                try:
                    ptr_raw = store.get_json(
                        bucket, pointer_key(settings.prefix, project, mn, "stable")
                    )
                    if ptr_raw and ptr_raw.get("manifest_uri"):
                        manifest_uri: str = ptr_raw["manifest_uri"]
                        resp["version_id"] = ptr_raw.get("version_id")
                        resp["manifest_uri"] = manifest_uri
                        resp["model_pkl_uri"] = manifest_uri.replace("/manifest.json", "/model.pkl")
                except Exception:
                    pass
            return resp

        if running is not None:
            return {"trigger_id": trigger_id, "status": "running"}
        return {"trigger_id": trigger_id, "status": "pending"}

    return app


async def _save_capped(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    written = 0
    chunk_size = 1024 * 1024
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeded {max_bytes} bytes for {upload.filename}",
                )
            fh.write(chunk)


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:  # pragma: no cover - uvicorn entrypoint
    if name == "app":
        global _app
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
