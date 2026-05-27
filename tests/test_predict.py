from fastapi.testclient import TestClient


def test_predict_single(make_app):
    client = TestClient(make_app)
    r = client.post("/predict", json={"features": {"a": 2, "b": 3}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] == 5.0
    assert body["model"]["version_id"] == "v1"


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


def test_predict_batch_empty_rows(make_app):
    client = TestClient(make_app)
    r = client.post("/predict/batch", json={"rows": []})
    assert r.status_code == 200
    assert r.json()["predictions"] == []


def test_model_info(make_app):
    client = TestClient(make_app)
    r = client.get("/model/info")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version_id"] == "v1"
    assert body["feature_columns"] == ["a", "b"]
    assert body["project"] == "product_dq"


def test_ready_returns_200_when_model_loaded(make_app):
    client = TestClient(make_app)
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_discovery_lists_stable_and_latest(make_app, s3_world, publish_artifacts, bucket_name):
    publish_artifacts(
        s3_world,
        bucket_name,
        "product_dq",
        "sentiment_analysis",
        2,
        feature_columns=["a", "b"],
        channel="latest",
    )
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
    assert sr["latest"] is None
