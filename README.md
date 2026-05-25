# rest_app

Lean FastAPI **multi-model inference gateway** for models published by the
`mlops` training pipeline. Standalone repo with **zero imports** from `mlops`
or `mlops_app`.

## What it does

On every prediction request, resolves `(category, project, model_name) +
version | channel` to an artifact in S3:

```
s3://{category}-artifacts/[<prefix>/]{project}/{model_name}/v{N}/model.pkl
                                                            /manifest.json
                                                            /stable.json   ← pointer
                                                            /latest.json
```

The flow is: read the pointer → validate against bundled JSON Schema → fetch
manifest → verify sha256 of `model.pkl` against `manifest.artifact_checksums`
→ `pickle.load` → cache under `(category, project, model_name, version_id)`.
Subsequent requests for the same key hit the cache directly.

Bounded LRU eviction (`CACHE_MAX_ENTRIES`, default 8) keeps memory predictable
when serving many distinct projects/models from one instance.

## Endpoints

| Method | Path                                          | Notes |
|--------|-----------------------------------------------|-------|
| GET    | `/`                                           | Redirects to `/static/trigger.html` — the admin UI. |
| GET    | `/static/trigger.html`                        | Single-page admin UI with two tabs: **Trigger** (upload dataset + params, fire training, watch live status) and **Predict** (load any model by `(category, project, model_name)` + channel/version, see schema, enter features, run a prediction). Uses the same `X-Admin-Token` as the API. |
| GET    | `/docs`                                       | FastAPI-generated Swagger UI for the full API. |
| GET    | `/health`                                     | Liveness, no S3 dependency. |
| GET    | `/ready`                                      | Returns the number of cached models. |
| GET    | `/model/info`                                 | Query: `?category=&project=&model_name=&version|channel=`. Loads into cache if needed. |
| POST   | `/predict`                                    | See body schema below. |
| POST   | `/predict/batch`                              | Same, but `rows: [...]`. Capped by `MAX_BATCH_SIZE`. |
| GET    | `/projects/{category}/{project}/models`       | Discovery: lists each model with its `stable` and `latest` pointer details. |
| GET    | `/cache`                                      | Admin: list cached entries. |
| POST   | `/cache/clear`                                | Admin: nuke the cache. |
| POST   | `/reload`                                     | Admin: evict + reload one target. Query params same as `/model/info`. |
| POST   | `/trigger-train`                              | Multipart upload (dataset + params.yaml). Writes trigger folder to S3, fires GitHub Actions training. Admin-token guarded. |
| GET    | `/trigger-status/{trigger_id}`                | Query: `?project=&category=`. Returns `pending | running | completed | failed`. |

### `/predict` request body

```json
{
  "features": {"a": 1.0, "b": 2.0},
  "category":   "mlops",          // optional, default = DEFAULT_CATEGORY env
  "project":    "product_dq",     // optional, default = DEFAULT_PROJECT env
  "model_name": "sentiment_analysis",  // optional, default = DEFAULT_MODEL_NAME env
  "version":    "v3",             // optional; pins exactly. Takes priority over `channel`.
  "channel":    "stable"          // optional, default = DEFAULT_CHANNEL env (=stable)
}
```

`version` and `channel` are mutually exclusive. If `version` is set, the
service skips reading the pointer and goes straight to the requested artifact.
If `channel` is set (or both are omitted), the service reads
`{project}/{model_name}/{channel}.json` to resolve the current version.

### `/trigger-train` (multipart)

The user uploads a dataset and a params.yaml in a single request. The service
writes the trigger folder to S3 in the canonical order (dataset → params →
trigger.json **last**) and fires a `train-model` `repository_dispatch` event
on the configured GitHub training repo. Requires `X-Admin-Token` header.

```bash
curl -X POST http://localhost:8000/trigger-train \
  -H "X-Admin-Token: $APP_ADMIN_TOKEN" \
  -F category=mlops \
  -F project=product_dq \
  -F model_name=sentiment_analysis \
  -F model_family=lgbm \
  -F dataset=@my_data.parquet \
  -F params=@params.yaml
```

Response:

```json
{
  "trigger_id": "20260525T143012Z_a1b2c3d4",
  "trigger_uri": "s3://mlops-artifacts/_triggers/product_dq/20260525T143012Z_a1b2c3d4/",
  "status_url": "/trigger-status/20260525T143012Z_a1b2c3d4?project=product_dq&category=mlops",
  "auto_promote": false
}
```

Poll `status_url` to watch the run progress:

```bash
curl "http://localhost:8000/trigger-status/20260525T143012Z_a1b2c3d4?project=product_dq&category=mlops"
# {"trigger_id": "...", "status": "pending"}    ← waiting for CI to start
# {"trigger_id": "...", "status": "running"}    ← CI is training
# {"trigger_id": "...", "status": "failed", "reason": "..."}
```

Returns `503` if `GITHUB_TRAINING_REPO` + `GITHUB_PAT` aren't both set.

## AWS credentials

The service uses **AWS env-var auth** through boto3's default credential chain:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

Set these in GitHub Secrets for CI, in the container environment for prod
deploys, or via `aws configure` for local dev. **Never** hardcode keys in
code, `Dockerfile`, or committed `.env` files.

## Configuration

See [`.env.example`](.env.example) for the full env-var list. The three
`DEFAULT_*` vars are optional — leave them blank to run as a pure
multi-model gateway where callers always supply the target in the request
body.

### Required env vars for `/trigger-train`

| Var | Purpose |
|---|---|
| `GITHUB_TRAINING_REPO` | `org/repo` of the training repo to dispatch to (e.g. `prescienceds/mlops`). |
| `GITHUB_PAT` | Fine-grained PAT with `contents: read` + `actions: write` on the training repo. |
| `APP_ADMIN_TOKEN` | Shared bearer guarding the endpoint (sent via `X-Admin-Token` header). |
| `MAX_DATASET_BYTES` | Optional; defaults to 100 MB. |
| `TRAINING_AUTO_PROMOTE` | Optional; `1`/`true` to auto-promote after training. Default `false`. |

## Local run

```powershell
cd c:\rest_app
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:AWS_ACCESS_KEY_ID="AKIA..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_DEFAULT_REGION="us-east-1"
$env:DEFAULT_CATEGORY="mlops"
$env:DEFAULT_PROJECT="product_dq"
$env:DEFAULT_MODEL_NAME="sentiment_analysis"
$env:APP_ADMIN_TOKEN="dev"
.\.venv\Scripts\python.exe -m uvicorn rest_app.app:app --reload
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

## Docker

```bash
docker build -t rest_app:latest .
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e APP_ADMIN_TOKEN=secret \
  rest_app:latest
```

Mount the AWS creds via your orchestrator's secret manager — never bake
them into the image.

## Example: predict with explicit target

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "category": "mlops",
        "project": "product_dq",
        "model_name": "sentiment_analysis",
        "channel": "stable",
        "features": {"a": 1.0, "b": 2.0}
      }'
```

Response:

```json
{
  "prediction": 0.42,
  "model": {
    "category": "mlops",
    "project": "product_dq",
    "model_name": "sentiment_analysis",
    "version_id": "v3"
  }
}
```
