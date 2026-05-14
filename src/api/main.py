from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.schemas import (
    Entity,
    ExtractionLogDetail,
    ExtractionLogSummary,
    ExtractRequest,
    ExtractResponse,
)
from src.db.connection import get_db
from src.db.crud import (
    create_extraction_log,
    get_extraction_log_by_id,
    get_extraction_logs,
)
from src.inference.predictor import LoreCrafterPredictor

predictor: LoreCrafterPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = LoreCrafterPredictor("models/lorecrafter-ner")
    except Exception:
        print("Modelo no encontrado. Ejecuta src/training/train.py primero.")
    yield


app = FastAPI(
    title="LoreCrafter NER API",
    version="1.0.0",
    description="Motor de extracción de entidades para textos de fantasía y rol.",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", summary="Health check")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post(
    "/extract", response_model=ExtractResponse, summary="Extraer entidades de un texto"
)
def extract_entities(
    request: ExtractRequest,
    db: Annotated[Session, Depends(get_db)],
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        entities = predictor.extract(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Persistir en DB
    create_extraction_log(db=db, input_text=request.text, entities=entities)

    return ExtractResponse(
        text=request.text,
        entities=[Entity(**e) for e in entities],
    )


@app.get(
    "/history",
    response_model=list[ExtractionLogSummary],
    summary="Historial de extracciones (paginado)",
)
def list_history(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return get_extraction_logs(db=db, limit=limit, offset=offset)


@app.get(
    "/history/{log_id}",
    response_model=ExtractionLogDetail,
    summary="Detalle de una extracción concreta",
)
def get_history_detail(
    log_id: int,
    db: Annotated[Session, Depends(get_db)],
):
    log = get_extraction_log_by_id(db=db, log_id=log_id)
    if log is None:
        raise HTTPException(status_code=404, detail=f"Log {log_id} no encontrado")
    return log
