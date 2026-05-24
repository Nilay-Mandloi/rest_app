from fastapi.testclient import TestClient


def test_predict_uses_env_defaults(make_app):
    client = TestClient(make_app)
    r = client.post("/predict", json={"features": {"a": 2, "b": 3}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["version_id"] == "v1"
    assert body["model"]["project"] == "product_dq"
    assert body["prediction"] == 5.0


def test_predict_with_explicit_target(make_app):
    client = TestClient(make_app)
    r = client.post(
        "/predict",
        json={
            "features": {"a": 1, "b": 1},
            "category": "mlops",
            "project": "product_dq",
            "model_name": "sentiment_analysis",
            "version": "v1",
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["prediction"] == 2.0


def test_predict_batch(make_app):
    client = TestClient(make_app)
    r = client.post(
        "/predict/batch",
        json={"rows": [{"a": 1, "b": 1}, {"a": 10, "b": 5}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predictions"] == [2.0, 15.0]
    assert body["model"]["version_id"] == "v1"


def test_predict_missing_feature_rejected(make_app):
    client = TestClient(make_app)
    r = client.post("/predict", json={"features": {"a": 1}})
    assert r.status_code == 422


def test_predict_rejects_version_and_channel_together(make_app):
    client = TestClient(make_app)
    r = client.post(
        "/predict",
        json={"features": {"a": 1, "b": 1}, "version": "v1", "channel": "stable"},
    )
    assert r.status_code == 400


def test_model_info_default_target(make_app):
    client = TestClient(make_app)
    r = client.get("/model/info")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version_id"] == "v1"
    assert body["project"] == "product_dq"


def test_ready_503_when_default_target_not_yet_loaded(make_app):
    client = TestClient(make_app)
    r = client.get("/ready")
    # Defaults configured (see conftest) but cache empty → standby.
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "standby"
    assert body["default_loaded"] is False
    assert body["cached_models"] == 0

    client.post("/predict", json={"features": {"a": 1, "b": 1}})
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["default_loaded"] is True
    assert body["cached_models"] == 1


def test_reload_requires_admin_token(make_app):
    client = TestClient(make_app)
    r = client.post("/reload")
    assert r.status_code == 401
    r = client.post("/reload", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401


def test_reload_with_admin_token(make_app, admin_token):
    client = TestClient(make_app)
    r = client.post("/reload", headers={"X-Admin-Token": admin_token})
    assert r.status_code == 200
    assert r.json()["version_id"] == "v1"


def test_cache_admin_endpoints(make_app, admin_token):
    client = TestClient(make_app)
    client.post("/predict", json={"features": {"a": 1, "b": 1}})

    r = client.get("/cache", headers={"X-Admin-Token": admin_token})
    assert r.status_code == 200
    assert r.json()["count"] == 1

    r = client.post("/cache/clear", headers={"X-Admin-Token": admin_token})
    assert r.json()["evicted"] == 1

    r = client.get("/cache", headers={"X-Admin-Token": admin_token})
    assert r.json()["count"] == 0


def test_discovery_lists_stable_and_latest(make_app, s3_world, publish_artifacts, bucket_name):
    # Publish a latest.json at v2 so the model has both stable (v1) and latest (v2).
    publish_artifacts(
        s3_world,
        bucket_name,
        "product_dq",
        "sentiment_analysis",
        2,
        feature_columns=["a", "b"],
        channel="latest",
    )
    # Add a second model so the listing has more than one entry.
    publish_artifacts(
        s3_world,
        bucket_name,
        "product_dq",
        "speech_recognition",
        1,
        feature_columns=["a"],
        channel="stable",
    )

    client = TestClient(make_app)
    r = client.get("/projects/mlops/product_dq/models")
    assert r.status_code == 200, r.text
    body = r.json()
    by_name = {m["model_name"]: m for m in body["models"]}
    assert set(by_name.keys()) == {"sentiment_analysis", "speech_recognition"}

    sa = by_name["sentiment_analysis"]
    assert sa["stable"]["version_id"] == "v1"
    assert sa["latest"]["version_id"] == "v2"

    sr = by_name["speech_recognition"]
    assert sr["stable"]["version_id"] == "v1"
    assert sr["latest"] is None  # no latest.json published for this one
