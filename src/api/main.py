from fastapi import FastAPI, HTTPException
from src.api.schemas import ExtractRequest, ExtractResponse, Entity
from src.inference.predictor import LoreCrafterPredictor

app = FastAPI(title="LoreCrafter NER API", version="1.0.0")

predictor = None


@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = LoreCrafterPredictor("models/lorecrafter-ner")
    except Exception:
        print("Modelo no encontrado. Ejecuta src/training/train.py primero.")


@app.post("/extract", response_model=ExtractResponse)
def extract_entities(request: ExtractRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    try:
        entities = predictor.extract(request.text)
        return ExtractResponse(
            text=request.text,
            entities=[Entity(text=e["text"], label=e["label"], start=e["start"], end=e["end"]) for e in entities],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}
