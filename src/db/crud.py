from sqlalchemy.orm import Session

from src.db.models import ExtractionLog


def create_extraction_log(
    db: Session,
    input_text: str,
    entities: list[dict],
) -> ExtractionLog:
    """Persiste una extracción NER en la base de datos."""
    log = ExtractionLog(
        input_text=input_text,
        entity_count=len(entities),
        entities_found=entities,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_extraction_logs(
    db: Session,
    limit: int = 20,
    offset: int = 0,
) -> list[ExtractionLog]:
    """Lista de logs paginada, del más reciente al más antiguo."""
    return (
        db.query(ExtractionLog)
        .order_by(ExtractionLog.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_extraction_log_by_id(db: Session, log_id: int) -> ExtractionLog | None:
    """Detalle de un log concreto por su ID."""
    return db.query(ExtractionLog).filter(ExtractionLog.id == log_id).first()
