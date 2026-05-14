from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class ExtractRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=5000, description="Texto narrativo a analizar"
    )


class ExtractResponse(BaseModel):
    text: str
    entities: list[Entity]


# ── Historial ──────────────────────────────────────────────────────────────────


class ExtractionLogSummary(BaseModel):
    """Resumen de un log (para listados paginados)."""

    id: int
    input_text: str
    entity_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ExtractionLogDetail(ExtractionLogSummary):
    """Detalle completo de un log, incluye snapshot de entidades."""

    entities_found: list[dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)
