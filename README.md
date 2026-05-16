# LoreCrafter NER

<p align="center">
  <em>Automatic entity extraction engine for mythology and fantasy lore</em>
</p>

---

### Language

[English](#-english) · [Castellano](#-castellano) · [Galego](#-galego)

---

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white&style=for-the-badge" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge" alt="PyTorch" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black&style=for-the-badge" alt="HuggingFace" />
  <img src="https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white&style=for-the-badge" alt="spaCy" />
  <img src="https://img.shields.io/badge/SQLAlchemy-D70E00?logo=sqlalchemy&logoColor=white&style=for-the-badge" alt="SQLAlchemy" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white&style=for-the-badge" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white&style=for-the-badge" alt="MLflow" />
  <img src="https://img.shields.io/badge/Groq-F5A623?logo=groq&logoColor=black&style=for-the-badge" alt="Groq" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white&style=for-the-badge" alt="Docker" />
  <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white&style=for-the-badge" alt="GitHub Actions" />
  <img src="https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white&style=for-the-badge" alt="Pydantic" />
  <img src="https://img.shields.io/badge/BeautifulSoup-4B0082?logo=python&logoColor=white&style=for-the-badge" alt="BeautifulSoup" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white&style=for-the-badge" alt="pytest" />
</p>

---

##  English

Automatic entity extraction engine for mythology and fantasy lore. Converts free-form narrative text from fantasy novels, D&D wikis, and tabletop RPG material into a structured database of **world-building entities**.

### Extracted Entities

| Label | Description | Example |
|---|---|---|
| `PERSONAJE` | Character names | Gandalf, Aragorn |
| `FACCION` | Groups, guilds, orders | The Fellowship of the Ring |
| `LUGAR` | Cities, kingdoms, regions | Gondor, Rivendell |
| `ARTEFACTO_MAGICO` | Magical objects | Anduril, the One Ring |
| `RAZA` | Fantasy races | Elf, Dwarf, Hobbit |

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Web Scraper│────▶│  LLM Generation  │────▶│  Dataset Merge   │
│  (Beautiful │     │  (Groq/Llama 3)  │     │  + Validation    │
│   Soup)     │     │  + Pre-annotate  │     │  + Deduplication │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                      │
                     ┌──────────────────┐     ┌────────▼─────────┐
                     │   MLflow         │◀────│   Fine-tuning    │
                     │   Tracking       │     │   (BERT-base)    │
                     └──────────────────┘     └────────┬─────────┘
                                                       │
                     ┌──────────────────┐     ┌────────▼─────────┐
                     │   PostgreSQL     │◀────│   FastAPI API    │
                     │   + SQLAlchemy   │     │   POST /extract  │
                     └──────────────────┘     └──────────────────┘
```

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start infrastructure (PostgreSQL + MLflow + API)
docker compose up -d

# 3. Generate dataset (choose one or combine)

# Option A: Legacy template-based generation
PYTHONPATH=. python3 src/dataset/create_dataset.py

# Option B: LLM-powered generation (requires Groq API key)
export GROQ_API_KEY=your_key
PYTHONPATH=. python3 src/dataset/llm_generator.py

# Option C: Full pipeline (scrape + LLM pre-annotate + merge)
PYTHONPATH=. python3 src/dataset/scraper.py
PYTHONPATH=. python3 src/dataset/llm_preannotator.py
PYTHONPATH=. python3 src/dataset/merge_datasets.py

# 4. Preprocess into train/val/test splits
PYTHONPATH=. python3 src/dataset/preprocess.py

# 5. Train the model (logs to MLflow at localhost:5000)
PYTHONPATH=. python3 src/training/train.py
```

> **Note:** `docker compose up -d` already starts the API at `localhost:8000`. For local development without Docker, run `PYTHONPATH=. uvicorn src.api.main:app --reload` instead.

### API Usage

```bash
# Health check
curl http://localhost:8000/health

# Extract entities from text
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Gandalf traveled to Rivendell carrying Glamdring."}'
```

Swagger UI available at `http://localhost:8000/docs`.

### Project Structure

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

### MLflow

Training runs are logged to the MLflow server started by Docker. Access the UI at `http://localhost:5000` after running `docker compose up -d`. The tracking URI is configured via the `MLFLOW_TRACKING_URI` environment variable (see `.env.example`).

### Model Card

See [MODEL_CARD.md](MODEL_CARD.md) for model details, metrics, and limitations.

---

##  Castellano

Motor de extracción automática de entidades para mitología y lore de fantasía. Convierte texto narrativo libre de novelas fantásticas, wikis de D&D y material de juegos de rol en una base de datos estructurada de **entidades de world-building**.

### Entidades Extraídas

| Etiqueta | Descripción | Ejemplo |
|---|---|---|
| `PERSONAJE` | Nombres de personajes | Gandalf, Aragorn |
| `FACCION` | Grupos, gremios, órdenes | La Comunidad del Anillo |
| `LUGAR` | Ciudades, reinos, regiones | Gondor, Rivendel |
| `ARTEFACTO_MAGICO` | Objetos mágicos | Anduril, el Anillo Único |
| `RAZA` | Razas de fantasía | Elfo, Enano, Hobbit |

### Arquitectura

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Web Scraper│────▶│  Generación LLM  │────▶│  Fusión de       │
│  (Beautiful │     │  (Groq/Llama 3)  │     │  Datasets        │
│   Soup)     │     │  + Pre-anotación │     │  + Validación    │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                      │
                     ┌──────────────────┐     ┌────────▼─────────┐
                     │   MLflow         │◀────│   Fine-tuning    │
                     │   Tracking       │     │   (BERT-base)    │
                     └──────────────────┘     └────────┬─────────┘
                                                       │
                     ┌──────────────────┐     ┌────────▼─────────┐
                     │   PostgreSQL     │◀────│   API FastAPI    │
                     │   + SQLAlchemy   │     │   POST /extract  │
                     └──────────────────┘     └──────────────────┘
```

### Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Iniciar infraestructura (PostgreSQL + MLflow + API)
docker compose up -d

# 3. Generar dataset (elige una opción o combina)

# Opción A: Generación legacy basada en plantillas
PYTHONPATH=. python3 src/dataset/create_dataset.py

# Opción B: Generación con LLM (requiere clave de Groq API)
export GROQ_API_KEY=tu_clave
PYTHONPATH=. python3 src/dataset/llm_generator.py

# Opción C: Pipeline completo (scrape + pre-anotar con LLM + fusionar)
PYTHONPATH=. python3 src/dataset/scraper.py
PYTHONPATH=. python3 src/dataset/llm_preannotator.py
PYTHONPATH=. python3 src/dataset/merge_datasets.py

# 4. Preprocesar en splits train/val/test
PYTHONPATH=. python3 src/dataset/preprocess.py

# 5. Entrenar el modelo (registra en MLflow en localhost:5000)
PYTHONPATH=. python3 src/training/train.py
```

> **Nota:** `docker compose up -d` ya levanta la API en `localhost:8000`. Para desarrollo local sin Docker, ejecuta `PYTHONPATH=. uvicorn src.api.main:app --reload`.

### Uso de la API

```bash
# Health check
curl http://localhost:8000/health

# Extraer entidades de un texto
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Gandalf viajó a Rivendel portando Glamdring."}'
```

Swagger UI disponible en `http://localhost:8000/docs`.

### Estructura del Proyecto

```
├── data/
│   ├── raw/                 # Textos extraídos de wikis
│   ├── annotations/         # Datasets etiquetados (múltiples fuentes)
│   │   ├── weak_labeled.json        # Legacy: plantillas con spaCy
│   │   ├── llm_synthetic.json       # Narrativas generadas por LLM
│   │   ├── llm_preannotated.json    # Texto real anotado por LLM
│   │   └── combined.json            # Dataset fusionado + validado
│   └── processed/           # Splits train/val/test (formato BIO)
├── notebooks/               # EDA, creación de datasets, análisis de errores
├── src/
│   ├── dataset/             # Pipeline de generación de datos
│   │   ├── create_dataset.py        # Legacy: supervisión débil con spaCy
│   │   ├── scraper.py               # Web scraper para wikis de fantasía
│   │   ├── llm_generator.py         # Datos sintéticos con Llama 3
│   │   ├── llm_preannotator.py      # Anotación NER zero-shot con LLM
│   │   ├── merge_datasets.py        # Fusionar + validar + deduplicar
│   │   └── preprocess.py            # Etiquetado BIO + splits train/val/test
│   ├── training/            # Pipeline de fine-tuning con MLflow
│   ├── api/                 # Endpoints REST con FastAPI
│   ├── inference/           # Predictor con modelo entrenado
│   └── db/                  # Modelos relacionales SQLAlchemy + PostgreSQL
├── configs/
│   ├── training.yaml        # Hiperparámetros del modelo
│   └── llm_generation.yaml  # Configuración generación LLM + scraper
└── tests/                   # Tests unitarios
```

### MLflow

Los entrenamientos se registran en el servidor MLflow que levanta Docker. Accede a la UI en `http://localhost:5000` tras ejecutar `docker compose up -d`. El URI de tracking se configura mediante la variable de entorno `MLFLOW_TRACKING_URI` (ver `.env.example`).

### Model Card

Consulta [MODEL_CARD.md](MODEL_CARD.md) para detalles del modelo, métricas y limitaciones.

---

##  Galego

Motor de extracción automática de entidades para mitoloxía e lore de fantasía. Converte texto narrativo libre de novelas fantásticas, wikis de D&D e material de xogos de rol nunha base de datos estruturada de **entidades de world-building**.

### Entidades Extraídas

| Etiqueta | Descrición | Exemplo |
|---|---|---|
| `PERSONAJE` | Nomes de personaxes | Gandalf, Aragorn |
| `FACCION` | Grupos, gremios, ordes | A Comunidade do Anel |
| `LUGAR` | Cidades, reinos, rexións | Gondor, Rivendel |
| `ARTEFACTO_MAGICO` | Obxectos máxicos | Anduril, o Anel Único |
| `RAZA` | Razas de fantasía | Elfo, Anano, Hobbit |

### Arquitectura

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Web Scraper│────▶│  Xeración LLM    │────▶│  Fusión de       │
│  (Beautiful │     │  (Groq/Llama 3)  │     │  Datasets        │
│   Soup)     │     │  + Pre-anotación │     │  + Validación    │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                      │
                     ┌──────────────────┐     ┌────────▼─────────┐
                     │   MLflow         │◀────│   Fine-tuning    │
                     │   Tracking       │     │   (BERT-base)    │
                     └──────────────────┘     └────────┬─────────┘
                                                       │
                     ┌──────────────────┐     ┌────────▼─────────┐
                     │   PostgreSQL     │◀────│   API FastAPI    │
                     │   + SQLAlchemy   │     │   POST /extract  │
                     └──────────────────┘     └──────────────────┘
```

### Inicio Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Iniciar infraestructura (PostgreSQL + MLflow + API)
docker compose up -d

# 3. Xerar dataset (escolle unha opción ou combina)

# Opción A: Xeración legacy baseada en plantillas
PYTHONPATH=. python3 src/dataset/create_dataset.py

# Opción B: Xeración con LLM (require clave de Groq API)
export GROQ_API_KEY=a_tua_clave
PYTHONPATH=. python3 src/dataset/llm_generator.py

# Opción C: Pipeline completo (scrape + pre-anotar con LLM + fusionar)
PYTHONPATH=. python3 src/dataset/scraper.py
PYTHONPATH=. python3 src/dataset/llm_preannotator.py
PYTHONPATH=. python3 src/dataset/merge_datasets.py

# 4. Preprocesar en splits train/val/test
PYTHONPATH=. python3 src/dataset/preprocess.py

# 5. Adestrar o modelo (rexistra en MLflow en localhost:5000)
PYTHONPATH=. python3 src/training/train.py
```

> **Nota:** `docker compose up -d` xa levanta a API en `localhost:8000`. Para desenvolvemento local sen Docker, executa `PYTHONPATH=. uvicorn src.api.main:app --reload`.

### Uso da API

```bash
# Health check
curl http://localhost:8000/health

# Extraer entidades dun texto
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Gandalf viaxou a Rivendel portando Glamdring."}'
```

Swagger UI dispoñible en `http://localhost:8000/docs`.

### Estrutura do Proxecto

```
├── data/
│   ├── raw/                 # Textos extraídos de wikis
│   ├── annotations/         # Datasets etiquetados (múltiples fontes)
│   │   ├── weak_labeled.json        # Legacy: plantillas con spaCy
│   │   ├── llm_synthetic.json       # Narrativas xeradas por LLM
│   │   ├── llm_preannotated.json    # Texto real anotado por LLM
│   │   └── combined.json            # Dataset fusionado + validado
│   └── processed/           # Splits train/val/test (formato BIO)
├── notebooks/               # EDA, creación de datasets, análise de erros
├── src/
│   ├── dataset/             # Pipeline de xeración de datos
│   │   ├── create_dataset.py        # Legacy: supervisión débil con spaCy
│   │   ├── scraper.py               # Web scraper para wikis de fantasía
│   │   ├── llm_generator.py         # Datos sintéticos con Llama 3
│   │   ├── llm_preannotator.py      # Anotación NER zero-shot con LLM
│   │   ├── merge_datasets.py        # Fusionar + validar + deduplicar
│   │   └── preprocess.py            # Etiquetado BIO + splits train/val/test
│   ├── training/            # Pipeline de fine-tuning con MLflow
│   ├── api/                 # Endpoints REST con FastAPI
│   ├── inference/           # Predictor con modelo adestrado
│   └── db/                  # Modelos relacionais SQLAlchemy + PostgreSQL
├── configs/
│   ├── training.yaml        # Hiperparámetros do modelo
│   └── llm_generation.yaml  # Configuración xeración LLM + scraper
└── tests/                   # Tests unitarios
```

### MLflow

Os adestramentos rexístranse no servidor MLflow que levanta Docker. Accede á UI en `http://localhost:5000` tras executar `docker compose up -d`. O URI de tracking configúrase mediante a variable de entorno `MLFLOW_TRACKING_URI` (ver `.env.example`).

### Model Card

Consulta [MODEL_CARD.md](MODEL_CARD.md) para detalles do modelo, métricas e limitacións.
