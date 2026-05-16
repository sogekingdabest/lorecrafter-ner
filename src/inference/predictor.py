from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch


class LoreCrafterPredictor:
    def __init__(self, model_path="models/lorecrafter-ner"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def extract(self, text: str) -> list[dict]:
        """
        Extrae entidades nombradas del texto.

        Retorna una lista de dicts con:
          - text:  subcadena original del texto
          - label: etiqueta de entidad (PERSONAJE, LUGAR, etc.)
          - start: offset de caracter de inicio en el texto original
          - end:   offset de caracter de fin (exclusivo) en el texto original
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,  # offsets char-level reales
            truncation=True,
            max_length=512,
        )

        offset_mapping = encoding.pop("offset_mapping")[
            0
        ].tolist()  # [(start, end), ...]

        with torch.no_grad():
            outputs = self.model(**encoding)

        pred_ids = torch.argmax(outputs.logits, dim=2)[0].tolist()
        word_ids = encoding.word_ids(batch_index=0)

        entities = []
        current_entity: dict | None = None
        previous_word_idx = None

        for token_idx, (word_idx, pred_id, (char_start, char_end)) in enumerate(
            zip(word_ids, pred_ids, offset_mapping)
        ):
            # Ignorar tokens especiales ([CLS], [SEP], [PAD])
            if word_idx is None:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                previous_word_idx = None
                continue

            label = self.id2label[pred_id]

            # Solo procesar el primer subtoken de cada palabra
            if word_idx == previous_word_idx:
                # Es un subtoken: extender el end del span actual
                if current_entity:
                    current_entity["end"] = char_end
                    current_entity["text"] = text[current_entity["start"] : char_end]  # noqa: E203
                previous_word_idx = word_idx
                continue

            previous_word_idx = word_idx

            if label.startswith("B-"):
                # Cerrar entidad anterior si existia
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "text": text[char_start:char_end],
                    "label": entity_type,
                    "start": char_start,
                    "end": char_end,
                }
            elif (
                label.startswith("I-")
                and current_entity
                and current_entity["label"] == label[2:]
            ):
                # Continuar entidad existente
                current_entity["end"] = char_end
                current_entity["text"] = text[current_entity["start"] : char_end]  # noqa: E203
            else:
                # Token "O" u I- sin B- previo: cerrar entidad si habia
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Cerrar la ultima entidad si quedo abierta
        if current_entity:
            entities.append(current_entity)

        return entities
