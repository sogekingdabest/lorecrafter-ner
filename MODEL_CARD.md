# Model Card: LoreCrafter NER

## Model Details

- **Base model**: `dslim/bert-base-NER` (BERT fine-tuned for general NER)
- **Fine-tuning**: Transfer learning to fantasy/role-playing domain
- **Framework**: HuggingFace Transformers + PyTorch
- **Tracking**: MLflow (SQLite backend)

## Intended Use

Extract fantasy world entities (characters, factions, locations, magical artifacts, races) from narrative texts of fantasy novels, D&D wikis, and tabletop RPG material.

## Training Data

- **Source**: Public wikis (D&D SRD, Tolkien Gateway) + synthetic text generation
- **Method**: Weak supervision with spaCy PhraseMatcher + template-based generation
- **Size**: ~960 labeled sentences (initial dataset)
- **Format**: BIO (Begin-Inside-Outside) tagging scheme

## Labels

| Label | Description | BIO Example |
|---|---|---|
| PERSONAJE | Proper names of characters | `B-PERSONAJE` Gandalf |
| FACCION | Organized groups | `B-FACCION` The Fellowship of the Ring |
| LUGAR | Geographic locations | `B-LUGAR` Minas Tirith |
| ARTEFACTO_MAGICO | Objects with magical properties | `B-ARTEFACTO_MAGICO` Glamdring |
| RAZA | Fantasy races/species | `B-RAZA` Elf |

## Metrics (Validation Set)

| Metric | PERSONAJE | FACCION | LUGAR | ARTEFACTO_MAGICO | RAZA | Overall |
|---|---|---|---|---|---|---|
| Precision | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| Recall | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| F1-Score | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |
| Accuracy | - | - | - | - | - | **0.9993** |

> *Note: Near-perfect scores on validation are expected with synthetic/template-generated data. Real-world performance on free-form fantasy text will be lower. See Limitations.*

## Limitations

- **Elves vs Cities**: The model confuses Elf names with city names because they linguistically share suffixes (e.g., "Elrond" vs "Dol Amroth"). More training on Elven language patterns is needed.
- **Generic artifacts**: Cannot distinguish between a "normal sword" and a "magic sword" without sufficient context. Phrases like "he drew his sword" without magical adjectives produce false negatives.
- **Ambiguous factions**: "The men of the north" could be a faction or a geographic description. The model tends to over-label.
- **Compound names**: "Aragorn son of Arathorn" may be fragmented into multiple entities instead of recognized as a single one.
- **Cross-domain**: Trained on Tolkien + D&D data; may fail with lore from other universes (Warhammer, Wheel of Time) without additional fine-tuning.
- **Language**: Optimized for Spanish and English texts. Degraded performance on other languages.
- **Synthetic data bias**: The training data was generated from templates, so the model performs best on sentences that follow similar patterns. Real-world prose with complex sentence structures will show lower F1 scores.

## Hyperparameters

See `configs/training.yaml` for the current configuration.

| Parameter | Value |
|---|---|
| Model | dslim/bert-base-NER |
| Max length | 128 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Epochs | 4 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |

## How to Get Started

```python
from src.inference.predictor import LoreCrafterPredictor

predictor = LoreCrafterPredictor(model_path="models/lorecrafter-ner")
entities = predictor.extract("Gandalf the Grey traveled to Rivendell carrying Glamdring.")
print(entities)
# [{'text': 'Gandalf', 'label': 'PERSONAJE'},
#  {'text': 'Rivendell', 'label': 'LUGAR'},
#  {'text': 'Glamdring', 'label': 'ARTEFACTO_MAGICO'}]
```
