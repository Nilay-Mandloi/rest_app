from __future__ import annotations

from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from .config import CATEGORY_RE, MODEL_NAME_RE, PROJECT_RE, VERSION_ID_RE, Settings
from .layout import pointer_key, project_prefix
from .loader import LoadedModel, ModelCache


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


def _predict(model: Any, X: list[list[Any]]) -> list[Any]:
    out = model.predict(X)
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

    app = FastAPI(title="rest_app", version="0.2.0", lifespan=_lifespan)
    app.state.settings = cfg
    app.state.cache = model_cache

    def get_settings(request: Request) -> Settings:
        return request.app.state.settings

    def get_cache(request: Request) -> ModelCache:
        return request.app.state.cache

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
            preds = _predict(loaded.obj, X)
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
            preds = _predict(loaded.obj, X)
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

    @app.post("/trigger-train", status_code=501)
    def trigger_train() -> dict[str, str]:
        return {
            "status": "not_implemented",
            "detail": "training trigger is deferred to Phase 8",
        }

    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:  # pragma: no cover - uvicorn entrypoint
    if name == "app":
        global _app
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
