from fastapi.testclient import TestClient


def test_health_endpoint(make_app):
    client = TestClient(make_app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
