import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_extract_empty_text(client):
    response = client.post("/extract", json={"text": ""})
    assert response.status_code in [200, 503]


def test_extract_valid_text(client):
    response = client.post("/extract", json={"text": "Gandalf viajó a Gondor."})
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "text" in data
        assert "entities" in data
