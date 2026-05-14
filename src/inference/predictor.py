from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np


class LoreCrafterPredictor:
    def __init__(self, model_path="models/lorecrafter-ner"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def extract(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        word_predictions = []
        current_word = ""
        current_pred = None
        current_start = None
        entities = []

        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            label = self.id2label[pred_id.item()]
            is_subword = token.startswith("##")

            if is_subword and current_word:
                current_word += token[2:]
            else:
                if current_word and current_pred and not current_pred.startswith("O"):
                    entities.append({
                        "text": current_word,
                        "label": current_pred[2:] if "-" in current_pred else current_pred,
                        "start": current_start,
                        "end": current_start + len(current_word),
                    })
                current_word = token
                current_pred = label
                current_start = i

        if current_word and current_pred and not current_pred.startswith("O"):
            entities.append({
                "text": current_word,
                "label": current_pred[2:] if "-" in current_pred else current_pred,
                "start": current_start,
                "end": current_start + len(current_word),
            })

        return entities
