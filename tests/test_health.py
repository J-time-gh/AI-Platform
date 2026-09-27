from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Enterprise AI Platform"
    assert "timestamp" in body


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()
