"""
Tests CRUD de la base de datos usando SQLite en memoria.
No necesita PostgreSQL corriendo.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.crud import (
    create_extraction_log,
    get_extraction_log_by_id,
    get_extraction_logs,
)
from src.db.models import Base

# ── Setup ──────────────────────────────────────────────────────────────────────

SQLALCHEMY_TEST_URL = "sqlite:///:memory:"

engine = create_engine(SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_db():
    """Crea las tablas antes de cada test y las elimina después."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


SAMPLE_ENTITIES = [
    {"text": "Gandalf", "label": "PERSONAJE", "start": 0, "end": 7},
    {"text": "Gondor", "label": "LUGAR", "start": 15, "end": 21},
]

# ── Tests ──────────────────────────────────────────────────────────────────────


def test_create_extraction_log(db):
    log = create_extraction_log(
        db=db,
        input_text="Gandalf viajó a Gondor.",
        entities=SAMPLE_ENTITIES,
    )

    assert log.id is not None
    assert log.input_text == "Gandalf viajó a Gondor."
    assert log.entity_count == 2
    assert log.entities_found == SAMPLE_ENTITIES
    assert log.created_at is not None


def test_get_extraction_logs_returns_list(db):
    create_extraction_log(db=db, input_text="Texto 1", entities=[])
    create_extraction_log(db=db, input_text="Texto 2", entities=SAMPLE_ENTITIES)

    logs = get_extraction_logs(db=db)

    assert isinstance(logs, list)
    assert len(logs) == 2


def test_get_extraction_logs_pagination(db):
    for i in range(5):
        create_extraction_log(db=db, input_text=f"Texto {i}", entities=[])

    page1 = get_extraction_logs(db=db, limit=2, offset=0)
    page2 = get_extraction_logs(db=db, limit=2, offset=2)

    assert len(page1) == 2
    assert len(page2) == 2
    # No deben ser los mismos registros
    assert {log.id for log in page1}.isdisjoint({log.id for log in page2})


def test_get_extraction_log_by_id(db):
    log = create_extraction_log(
        db=db,
        input_text="Aragorn empuñó Andúril.",
        entities=SAMPLE_ENTITIES,
    )

    fetched = get_extraction_log_by_id(db=db, log_id=log.id)

    assert fetched is not None
    assert fetched.id == log.id
    assert fetched.input_text == "Aragorn empuñó Andúril."


def test_get_extraction_log_by_id_not_found(db):
    result = get_extraction_log_by_id(db=db, log_id=9999)
    assert result is None


def test_entity_count_zero_for_empty_entities(db):
    log = create_extraction_log(db=db, input_text="Texto sin entidades.", entities=[])
    assert log.entity_count == 0
    assert log.entities_found == []
