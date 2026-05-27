# rest_app

Lean FastAPI inference service for a model produced by the `mlops` training
pipeline. Standalone repo with **zero imports** from `mlops` / `mlops_app`.

## Design — one model, baked into the image

The model is **baked into the Docker image at build time**, not loaded from S3
at runtime. The prediction path is completely storage-agnostic: it reads two
local files and never touches AWS.

```
docker build --build-arg MODEL_URL=<presigned> --build-arg MANIFEST_URL=<presigned> .
        │
        ├── curl  MODEL_URL    -> /app/model.pkl
        └── curl  MANIFEST_URL -> /app/manifest.json
```

At startup the app calls `load_baked_model()` which reads `/app/manifest.json`
(feature columns, version, run_id, model_type) and `pickle.load`s
`/app/model.pkl`. That's it — no buckets, no pointers, no cache.

Swapping storage backends (S3 → GCS → Azure Blob) only changes the CI step that
generates the presigned download URL; the Dockerfile `curl` and the app stay
identical. See [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).

## Endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/`                            | Redirects to `/static/trigger.html` — the admin UI. |
| GET  | `/docs`                        | Swagger UI for the full API. |
| GET  | `/health`                      | Liveness. Always 200 if the process is up. |
| GET  | `/ready`                       | 200 once the baked model is loaded, else 503. |
| GET  | `/model/info`                  | Returns the baked model's version, run_id, type, and feature columns. |
| POST | `/predict`                     | Body: `{"features": {...}}`. |
| POST | `/predict/batch`               | Body: `{"rows": [{...}, ...]}`. Capped by `MAX_BATCH_SIZE`. |
| GET  | `/projects/{category}/{project}/models` | Discovery: lists each model's `stable`/`latest` pointer (reads S3). |
| GET  | `/projects/{category}/{project}/models/{model}/versions` | Discovery: all published versions + S3 URIs. |
| POST | `/trigger-train`               | Multipart upload (dataset + params.yaml). Writes a trigger folder to S3 and fires a GitHub Actions training run. Admin-token guarded. |
| GET  | `/trigger-status/{trigger_id}` | Query `?project=&category=`. Returns `pending \| running \| completed \| failed`. |

The discovery and trigger endpoints are the **only** ones that talk to S3 — they
back the admin UI's model browser and the training trigger. Prediction does not.

### `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"a": 1.0, "b": 2.0}}'
```

```json
{
  "prediction": 0.42,
  "model": {"version_id": "v3", "model_type": "lightgbm", "run_id": "abc123"}
}
```

If the manifest declares `feature_columns`, requests missing any of them get a
`422` listing the missing names.

### `/trigger-train`

Uploads a dataset and a `params.yaml`, writes the trigger folder to S3 in the
canonical order (dataset → params → `trigger.json` **last**), then fires a
`train-model` `repository_dispatch` on the configured training repo. Requires
the `X-Admin-Token` header. Returns `503` if `GITHUB_TRAINING_REPO` + `GITHUB_PAT`
are not both set.

## AWS credentials

The trigger + discovery endpoints use boto3's default credential chain via
**AWS env-var auth**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_DEFAULT_REGION`. Never hardcode keys in code, the Dockerfile, or committed
`.env` files. The prediction path needs no credentials at all.

See [`.env.example`](.env.example) for the full env-var list.

## Local run

```powershell
cd c:\rest_app
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:APP_ADMIN_TOKEN="dev"
# Point at a locally-exported model + manifest, or omit to start in /ready=503 standby.
$env:MODEL_PKL_PATH="C:\path\to\model.pkl"
$env:MANIFEST_PATH="C:\path\to\manifest.json"
.\.venv\Scripts\python.exe -m uvicorn rest_app.app:app --reload
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

## Docker

The image is built and deployed by CI (`deploy.yml`), which presigns the model
artifacts and bakes them in. To build locally against an already-presigned URL:

```bash
docker build \
  --build-arg MODEL_URL="$MODEL_URL" \
  --build-arg MANIFEST_URL="$MANIFEST_URL" \
  -t rest_app:latest .
docker run -p 8000:8000 -e APP_ADMIN_TOKEN=secret rest_app:latest
```
