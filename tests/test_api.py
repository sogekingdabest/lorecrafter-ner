"""
Tests de la API usando TestClient de FastAPI.
Se crea una app de test con predictor mockeado para no depender del modelo en disco.
"""

from contextlib import asynccontextmanager
from typing import Annotated
from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.schemas import Entity, ExtractRequest, ExtractResponse

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """Sesion de DB falsa que no hace nada."""
    db = MagicMock()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.refresh = MagicMock()
    return db


def _build_test_app(predictor=None):
    """Construye una FastAPI app de test con el predictor dado (puede ser None)."""
    test_predictor = predictor

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        yield

    app = FastAPI(
        title="LoreCrafter NER API (test)",
        version="1.0.0",
        lifespan=test_lifespan,
    )

    @app.get("/health", summary="Health check")
    def health_check():
        return {"status": "ok", "model_loaded": test_predictor is not None}

    @app.post(
        "/extract",
        response_model=ExtractResponse,
        summary="Extraer entidades de un texto",
    )
    def extract_entities(
        request: ExtractRequest,
        db: Annotated[Session, Depends(lambda: MagicMock())],
    ):
        if test_predictor is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")

        try:
            entities = test_predictor.extract(request.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return ExtractResponse(
            text=request.text,
            entities=[Entity(**e) for e in entities],
        )

    return app


@pytest.fixture
def client_with_model(mock_db):
    """Cliente con predictor mockeado y DB mockeada."""
    fake_entities = [
        {"text": "Gandalf", "label": "PERSONAJE", "start": 0, "end": 7},
        {"text": "Gondor", "label": "LUGAR", "start": 15, "end": 21},
    ]
    fake_predictor = MagicMock()
    fake_predictor.extract.return_value = fake_entities

    app = _build_test_app(predictor=fake_predictor)

    from src.db.connection import get_db

    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_model(mock_db):
    """Cliente sin predictor (simula modelo no encontrado)."""
    app = _build_test_app(predictor=None)

    from src.db.connection import get_db

    app.dependency_overrides[get_db] = lambda: mock_db

    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


# ── Tests /health ──────────────────────────────────────────────────────────────


def test_health_check_model_loaded(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_check_model_not_loaded(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


# ── Tests /extract ─────────────────────────────────────────────────────────────


def test_extract_valid_text_returns_200(client_with_model):
    response = client_with_model.post(
        "/extract", json={"text": "Gandalf viajó a Gondor."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "entities" in data
    assert isinstance(data["entities"], list)


def test_extract_response_entity_fields(client_with_model):
    response = client_with_model.post(
        "/extract", json={"text": "Gandalf viajó a Gondor."}
    )
    entities = response.json()["entities"]
    assert len(entities) == 2
    for entity in entities:
        assert "text" in entity
        assert "label" in entity
        assert "start" in entity
        assert "end" in entity


def test_extract_empty_text_returns_422(client_with_model):
    """Texto vacio debe fallar con 422 Unprocessable Entity (validacion Pydantic)."""
    response = client_with_model.post("/extract", json={"text": ""})
    assert response.status_code == 422


def test_extract_text_too_long_returns_422(client_with_model):
    """Texto >5000 chars debe fallar con 422."""
    response = client_with_model.post("/extract", json={"text": "a" * 5001})
    assert response.status_code == 422


def test_extract_returns_503_when_model_not_loaded(client_no_model):
    response = client_no_model.post(
        "/extract", json={"text": "Gandalf viajó a Gondor."}
    )
    assert response.status_code == 503


def test_extract_missing_text_field_returns_422(client_with_model):
    response = client_with_model.post("/extract", json={})
    assert response.status_code == 422
