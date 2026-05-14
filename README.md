# LoreCrafter NER

Automatic entity extraction engine for mythology and fantasy lore. Converts free-form narrative text from fantasy novels, D&D wikis, and tabletop RPG material into a structured database of **world-building entities**.

## Extracted Entities

| Label | Description | Example |
|---|---|---|
| `PERSONAJE` | Character names | Gandalf, Aragorn |
| `FACCION` | Groups, guilds, orders | The Fellowship of the Ring, Mordor |
| `LUGAR` | Cities, kingdoms, regions | Gondor, Rivendell |
| `ARTEFACTO_MAGICO` | Magical objects | Anduril, the One Ring |
| `RAZA` | Fantasy races | Elf, Dwarf, Hobbit |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start PostgreSQL + MLflow
docker compose up -d

# 3. Train the model
PYTHONPATH=. python3 src/training/train.py

# 4. Launch the API
PYTHONPATH=. uvicorn src.api.main:app --reload
```

## Project Structure

```
├── data/              # Raw, processed, and annotated datasets
├── notebooks/         # EDA, dataset creation, error analysis
├── src/
│   ├── dataset/       # Weak supervision with spaCy + BIO preprocessing
│   ├── training/      # Fine-tuning pipeline with MLflow tracking
│   ├── api/           # FastAPI REST endpoints
│   ├── inference/     # Predictor with trained model
│   └── db/            # SQLAlchemy relational models + PostgreSQL
├── configs/           # Hyperparameters in YAML
└── tests/             # Unit tests
```

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Extract entities from text
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Gandalf traveled to Rivendell carrying Glamdring."}'
```

Swagger UI available at `http://localhost:8000/docs`.

## Relational Diagram

```
Input (raw text)
        │
        ▼
┌──────────────────┐
│   FastAPI API    │  POST /extract
└────────┬─────────┘
         │ JSON
         ▼
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│   NER Model      │────▶│  Characters  │────▶│  Factions    │
│   (BERT fine-    │     │  (id, name,  │     │  (id, name,  │
│    tuned)        │     │   race_id,   │     │   type)      │
└────────┬─────────┘     │   location_id│     └──────────────┘
         │               └──────┬───────┘
         ▼                      │
┌──────────────────┐            │
│   PostgreSQL     │◀───────────┘
│                  │
│  ┌────────────┐  │  ┌──────────────┐  ┌──────────────────┐
│  │ Locations  │  │  │   Races      │  │ MagicArtifacts   │
│  │(id, name,  │  │  │(id, name,    │  │(id, name,        │
│  │ type)      │  │  │ description) │  │ wielder_id)      │
│  └────────────┘  │  └──────────────┘  └──────────────────┘
└──────────────────┘
```

## Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for model details, metrics, and limitations.

## MLflow

Access experiment tracking at `http://localhost:5000` after running `docker compose up -d`.
For local development, MLflow data is stored in `mlflow.db` (SQLite).
