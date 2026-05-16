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

# 3. Generate dataset (choose one or combine)

# Option A: Legacy template-based generation
PYTHONPATH=. python3 src/dataset/create_dataset.py

# Option B: LLM-powered generation (requires Ollama running)
PYTHONPATH=. python3 src/dataset/llm_generator.py

# Option C: Full pipeline (scrape + LLM pre-annotate + merge)
PYTHONPATH=. python3 src/dataset/scraper.py
PYTHONPATH=. python3 src/dataset/llm_preannotator.py
PYTHONPATH=. python3 src/dataset/merge_datasets.py

# 4. Preprocess into train/val/test splits
PYTHONPATH=. python3 src/dataset/preprocess.py

# 5. Train the model
PYTHONPATH=. python3 src/training/train.py

# 6. Launch the API
PYTHONPATH=. uvicorn src.api.main:app --reload
```

## Project Structure

```
├── data/
│   ├── raw/                 # Scraped texts from wikis
│   ├── annotations/         # Labeled datasets (multiple sources)
│   │   ├── weak_labeled.json        # Legacy: spaCy template-based
│   │   ├── llm_synthetic.json       # LLM-generated narratives
│   │   ├── llm_preannotated.json    # LLM-annotated real text
│   │   └── combined.json            # Merged + validated dataset
│   └── processed/           # Train/val/test splits (BIO format)
├── notebooks/               # EDA, dataset creation, error analysis
├── src/
│   ├── dataset/             # Data generation pipeline
│   │   ├── create_dataset.py        # Legacy: weak supervision with spaCy
│   │   ├── scraper.py               # Web scraper for fantasy wikis
│   │   ├── llm_generator.py         # Synthetic data with Llama 3
│   │   ├── llm_preannotator.py      # Zero-shot NER annotation with LLM
│   │   ├── merge_datasets.py        # Merge + validate + deduplicate
│   │   └── preprocess.py            # BIO tagging + train/val/test splits
│   ├── training/            # Fine-tuning pipeline with MLflow tracking
│   ├── api/                 # FastAPI REST endpoints
│   ├── inference/           # Predictor with trained model
│   └── db/                  # SQLAlchemy relational models + PostgreSQL
├── configs/
│   ├── training.yaml        # Model hyperparameters
│   └── llm_generation.yaml  # LLM generation + scraper config
└── tests/                   # Unit tests
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
