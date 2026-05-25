from __future__ import annotations

import tempfile
from collections.abc import Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from .config import CATEGORY_RE, MODEL_NAME_RE, PROJECT_RE, VERSION_ID_RE, Settings
from .layout import (
    pointer_key,
    project_prefix,
    trigger_completed_key,
    trigger_failure_key,
    trigger_metadata_key,
    trigger_running_key,
)
from .loader import LoadedModel, ModelCache
from .ports.orchestration import OrchestrationAdapter
from .ports.storage import ArtifactStore

_STATIC_DIR = Path(__file__).parent / "static"


class PredictRequest(BaseModel):
    features: dict[str, Any]
    category: str | None = None
    project: str | None = None
    model_name: str | None = None
    version: str | None = None  # "v3" — pins exactly; takes priority over channel
    channel: str | None = None  # "stable" | "latest" | ... — resolved via pointer


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    category: str | None = None
    project: str | None = None
    model_name: str | None = None
    version: str | None = None
    channel: str | None = None


def _to_matrix(rows: Iterable[dict[str, Any]], columns: list[str] | None) -> list[list[Any]]:
    rows = list(rows)
    if columns:
        return [[r.get(c) for c in columns] for r in rows]
    if not rows:
        return []
    keys = sorted(rows[0].keys())
    return [[r.get(k) for k in keys] for r in rows]


def _predict(model: Any, X: list[list[Any]], columns: list[str] | None = None) -> list[Any]:
    import pandas as pd

    frame: Any = pd.DataFrame(X, columns=columns) if columns else X
    out = model.predict(frame)
    try:
        return list(out)
    except TypeError:
        return [out]


def _feature_columns(loaded: LoadedModel) -> list[str] | None:
    contract = loaded.manifest.schema_contract or {}
    cols = contract.get("feature_columns")
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols]
    return None


def _check_admin_token(settings: Settings, token: str | None) -> None:
    if not settings.admin_token:
        raise HTTPException(status_code=501, detail="admin disabled (APP_ADMIN_TOKEN not set)")
    if not token or token != settings.admin_token:
        raise HTTPException(status_code=401, detail="invalid or missing admin token")


def _resolve_target(
    settings: Settings,
    *,
    category: str | None,
    project: str | None,
    model_name: str | None,
    version: str | None,
    channel: str | None,
) -> tuple[str, str, str, str | None, str | None]:
    """Apply env defaults, validate, return (cat, project, model_name, version, channel).

    Either version or channel is returned non-None; never both.
    """
    cat = (category or settings.default_category).strip()
    proj = (project or settings.default_project).strip()
    mn = (model_name or settings.default_model_name).strip()
    if not cat or not proj or not mn:
        raise HTTPException(
            status_code=400,
            detail=(
                "category, project, model_name must be supplied either in the request body "
                "or via DEFAULT_CATEGORY/DEFAULT_PROJECT/DEFAULT_MODEL_NAME env vars"
            ),
        )
    if not CATEGORY_RE.match(cat):
        raise HTTPException(status_code=400, detail=f"invalid category: {cat!r}")
    if not PROJECT_RE.match(proj):
        raise HTTPException(status_code=400, detail=f"invalid project: {proj!r}")
    if not MODEL_NAME_RE.match(mn):
        raise HTTPException(status_code=400, detail=f"invalid model_name: {mn!r}")

    if version and channel:
        raise HTTPException(status_code=400, detail="specify version OR channel, not both")
    if version:
        if not VERSION_ID_RE.match(version):
            raise HTTPException(
                status_code=400, detail=f"version must look like 'v3'; got {version!r}"
            )
        return cat, proj, mn, version, None
    ch = (channel or settings.default_channel).strip()
    return cat, proj, mn, None, ch


def create_app(
    settings: Settings | None = None,
    cache: ModelCache | None = None,
    writable_store: ArtifactStore | None = None,
    orchestrator: OrchestrationAdapter | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()
    model_cache = cache or ModelCache(cfg)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        if cfg.default_category and cfg.default_project and cfg.default_model_name:
            try:
                loaded = model_cache.resolve_and_load(
                    category=cfg.default_category,
                    project=cfg.default_project,
                    model_name=cfg.default_model_name,
                    channel=cfg.default_channel,
                )
                logger.info(f"preloaded default model: {loaded.version_id}")
            except Exception as exc:
                logger.warning(f"default preload failed; serving in lazy mode: {exc}")
        yield

    app = FastAPI(title="rest_app", version="0.4.0", lifespan=_lifespan)
    app.state.settings = cfg
    app.state.cache = model_cache
    app.state.writable_store = writable_store  # built lazily on first /trigger-train
    app.state.orchestrator = orchestrator  # built lazily on first /trigger-train

    # Serve the trigger.html admin form at /static/trigger.html, root redirects to it.
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def _root() -> RedirectResponse:
            return RedirectResponse(url="/static/trigger.html", status_code=302)

    def get_settings(request: Request) -> Settings:
        return request.app.state.settings

    def get_cache(request: Request) -> ModelCache:
        return request.app.state.cache

    def get_writable_store(request: Request) -> ArtifactStore:
        if request.app.state.writable_store is None:
            from .factories import get_writable_artifact_store

            request.app.state.writable_store = get_writable_artifact_store(
                request.app.state.settings
            )
        return request.app.state.writable_store

    def get_orchestrator(request: Request) -> OrchestrationAdapter:
        if request.app.state.orchestrator is None:
            from .factories import get_orchestrator as build_orchestrator

            request.app.state.orchestrator = build_orchestrator(request.app.state.settings)
        return request.app.state.orchestrator

    # ------------------------------------------------------------------
    # Health / readiness
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        """Readiness with explicit semantics.

        - process_alive: true once the app object exists.
        - default_loaded: true only when a default target is configured AND
          its current pointer-resolved version sits in the cache. When no
          default is configured this is null (gateway mode — nothing to preload).
        - cached_models: count of currently-loaded entries.

        Returns 503 only when a default target is configured but is not loaded.
        Pure-gateway mode (no defaults) always returns 200.
        """
        entries = cache.list_entries()
        has_default = bool(
            settings.default_category and settings.default_project and settings.default_model_name
        )
        default_loaded: bool | None = None
        if has_default:
            default_loaded = any(
                e["category"] == settings.default_category
                and e["project"] == settings.default_project
                and e["model_name"] == settings.default_model_name
                for e in entries
            )
        body: dict[str, Any] = {
            "status": "ready" if (not has_default or default_loaded) else "standby",
            "process_alive": True,
            "default_loaded": default_loaded,
            "cached_models": len(entries),
        }
        if has_default and not default_loaded:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content=body)
        return body

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _load_or_400(
        cache: ModelCache,
        settings: Settings,
        body: PredictRequest | BatchPredictRequest,
    ) -> LoadedModel:
        cat, proj, mn, ver, ch = _resolve_target(
            settings,
            category=body.category,
            project=body.project,
            model_name=body.model_name,
            version=body.version,
            channel=body.channel,
        )
        try:
            return cache.resolve_and_load(
                category=cat, project=proj, model_name=mn, version_id=ver, channel=ch
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/predict")
    def predict(
        req: PredictRequest,
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        loaded = _load_or_400(cache, settings, req)
        cols = _feature_columns(loaded)
        if cols is not None:
            missing = [c for c in cols if c not in req.features]
            if missing:
                raise HTTPException(status_code=422, detail=f"missing features: {missing}")
        X = _to_matrix([req.features], cols)
        try:
            preds = _predict(loaded.obj, X, cols)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
        return {
            "prediction": preds[0] if preds else None,
            "model": {
                "category": loaded.category,
                "project": loaded.project,
                "model_name": loaded.model_name,
                "version_id": loaded.version_id,
            },
        }

    @app.post("/predict/batch")
    def predict_batch(
        req: BatchPredictRequest,
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        loaded = _load_or_400(cache, settings, req)
        if not req.rows:
            return {"predictions": [], "model": {"version_id": loaded.version_id}}
        if len(req.rows) > settings.max_batch_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"batch size {len(req.rows)} exceeds MAX_BATCH_SIZE={settings.max_batch_size}"
                ),
            )
        cols = _feature_columns(loaded)
        if cols is not None:
            for i, r in enumerate(req.rows):
                missing = [c for c in cols if c not in r]
                if missing:
                    raise HTTPException(
                        status_code=422, detail=f"row {i} missing features: {missing}"
                    )
        X = _to_matrix(req.rows, cols)
        try:
            preds = _predict(loaded.obj, X, cols)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
        return {
            "predictions": list(preds),
            "model": {
                "category": loaded.category,
                "project": loaded.project,
                "model_name": loaded.model_name,
                "version_id": loaded.version_id,
            },
        }

    # ------------------------------------------------------------------
    # Model info & discovery
    # ------------------------------------------------------------------

    @app.get("/model/info")
    def model_info(
        category: str | None = None,
        project: str | None = None,
        model_name: str | None = None,
        version: str | None = None,
        channel: str | None = None,
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        cat, proj, mn, ver, ch = _resolve_target(
            settings,
            category=category,
            project=project,
            model_name=model_name,
            version=version,
            channel=channel,
        )
        try:
            loaded = cache.resolve_and_load(
                category=cat, project=proj, model_name=mn, version_id=ver, channel=ch
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        p = loaded.pointer
        m = loaded.manifest
        return {
            "category": loaded.category,
            "project": loaded.project,
            "model_name": loaded.model_name,
            "version": loaded.version,
            "version_id": loaded.version_id,
            "run_id": m.run_id,
            "registry_version": m.registry_version,
            "model_type": m.model_type,
            "channel": ch,
            "loaded_at": loaded.loaded_at,
            "promoted_at": p.promoted_at if p else None,
            "updated_at": p.updated_at if p else None,
            "mlflow_run_url": (p.mlflow_run_url if p else None) or m.mlflow_run_url,
            "mlflow_model_url": (p.mlflow_model_url if p else None) or m.mlflow_model_url,
        }

    @app.get("/projects/{category}/{project}/models")
    def list_models(
        category: str,
        project: str,
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
    ) -> dict[str, Any]:
        if not CATEGORY_RE.match(category):
            raise HTTPException(status_code=400, detail=f"invalid category: {category!r}")
        if not PROJECT_RE.match(project):
            raise HTTPException(status_code=400, detail=f"invalid project: {project!r}")
        bucket = settings.bucket_for(category)
        prefix = project_prefix(settings.prefix, project)

        store = cache.store
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
            code = ""
            response = getattr(exc, "response", None)
            if isinstance(response, dict):
                code = response.get("Error", {}).get("Code", "")
            if code == "NoSuchBucket":
                raise HTTPException(status_code=404, detail=f"bucket not found: {bucket}") from exc
            status = 403 if code in ("AccessDenied", "AllAccessDisabled") else 502
            raise HTTPException(
                status_code=status,
                detail=f"discovery list failed for s3://{bucket}/{prefix}: {code or exc}",
            ) from exc

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

    # ------------------------------------------------------------------
    # Cache admin
    # ------------------------------------------------------------------

    @app.get("/cache")
    def list_cache(
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _check_admin_token(settings, x_admin_token)
        entries = cache.list_entries()
        return {
            "count": len(entries),
            "max_entries": settings.cache_max_entries,
            "entries": entries,
        }

    @app.post("/cache/clear")
    def clear_cache(
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        _check_admin_token(settings, x_admin_token)
        return {"evicted": cache.clear()}

    @app.post("/reload")
    def reload_model(
        category: str | None = None,
        project: str | None = None,
        model_name: str | None = None,
        channel: str | None = None,
        version: str | None = None,
        cache: ModelCache = Depends(get_cache),
        settings: Settings = Depends(get_settings),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        """Evict and re-load a specific (category, project, model_name) target.

        Resolves the channel (default stable) or honours an explicit version,
        evicts the existing cache entry, and forces a fresh download.
        """
        _check_admin_token(settings, x_admin_token)
        cat, proj, mn, ver, ch = _resolve_target(
            settings,
            category=category,
            project=project,
            model_name=model_name,
            version=version,
            channel=channel,
        )
        if ver is None:
            try:
                pointer = cache._read_pointer(cat, proj, mn, ch or settings.default_channel)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            ver = pointer.version_id
        cache.evict(cat, proj, mn, ver)
        try:
            loaded = cache.resolve_and_load(
                category=cat, project=proj, model_name=mn, version_id=ver
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "reloaded", "version_id": loaded.version_id}

    # ------------------------------------------------------------------
    # Training trigger
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
        """Accept dataset+params upload, write trigger folder to S3, fire training.

        Auth: requires X-Admin-Token header matching APP_ADMIN_TOKEN.
        Multipart form fields: see endpoint signature. dataset/params are file uploads.
        Returns: {trigger_id, trigger_uri, status_url, auto_promote}.
        """
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

        # Stream uploads to a temp dir with size enforcement, then hand paths to publisher.
        from .publisher import publish_trigger

        with tempfile.TemporaryDirectory(prefix="trigger_") as tmpdir:
            tmp = Path(tmpdir)
            dataset_local = tmp / (dataset.filename or "dataset.bin")
            params_local = tmp / (params.filename or "params.yaml")

            await _save_capped(dataset, dataset_local, settings.max_dataset_bytes)
            await _save_capped(params, params_local, 1 * 1024 * 1024)  # params.yaml: 1 MB cap

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
                # orchestrator refused — failed.json marker already written
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
        """Resolve trigger lifecycle by reading the four marker files in S3.

        States: pending (only trigger.json) → running (running.json present) →
        completed (stable pointer was updated after trigger creation) or
        failed (failed.json present).
        """
        _validate_category_project(category, project)
        bucket = settings.bucket_for(category)

        # Priority: failed > completed > running > pending.
        # meta (trigger.json) is always read — needed for the 404 guard and to
        # look up model_name when surfacing artifact paths on completion.
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
            # Best-effort: read stable pointer to surface the pkl path.
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
                    pass  # non-fatal; caller just won't see artifact paths
            return resp

        if running is not None:
            return {"trigger_id": trigger_id, "status": "running"}
        return {"trigger_id": trigger_id, "status": "pending"}

    return app


async def _save_capped(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    """Stream UploadFile to dest, raising 413 if it exceeds max_bytes."""
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
