from fastapi.testclient import TestClient


def test_health_endpoint(make_app):
    client = TestClient(make_app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_trigger_train_returns_501(make_app):
    client = TestClient(make_app)
    r = client.post("/trigger-train")
    assert r.status_code == 501
    assert r.json()["status"] == "not_implemented"
