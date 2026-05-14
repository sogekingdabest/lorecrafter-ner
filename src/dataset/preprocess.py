import json
from pathlib import Path
from sklearn.model_selection import train_test_split


BASE_LABELS = ["PERSONAJE", "FACCION", "LUGAR", "ARTEFACTO_MAGICO", "RAZA"]
BIO_LABELS = []
for label in BASE_LABELS:
    BIO_LABELS.append(f"B-{label}")
    BIO_LABELS.append(f"I-{label}")
BIO_LABELS.append("O")

BIO2ID = {label: i for i, label in enumerate(BIO_LABELS)}


def convert_to_bio(text, entities, base_labels=BASE_LABELS):
    tokens = text.split()
    bio_tags = ["O"] * len(tokens)

    for start_char, end_char, label in entities:
        token_start = None
        token_end = None
        current_pos = 0

        for i, token in enumerate(tokens):
            token_end_pos = current_pos + len(token)
            if current_pos <= start_char < token_end_pos:
                token_start = i
            if current_pos < end_char <= token_end_pos:
                token_end = i
            current_pos = token_end_pos + 1

        if token_start is not None and token_end is not None:
            bio_tags[token_start] = f"B-{label}"
            for i in range(token_start + 1, token_end + 1):
                bio_tags[i] = f"I-{label}"

    return tokens, bio_tags


def prepare_splits(data_path, output_dir="data/processed", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    with open(data_path) as f:
        data = json.load(f)

    processed = []
    for item in data:
        tokens, bio_tags = convert_to_bio(item["text"], item["entities"])
        processed.append({"tokens": tokens, "ner_tags": bio_tags})

    train_val, test = train_test_split(processed, test_size=test_ratio, random_state=42)
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        out_path = Path(output_dir) / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"{split_name}: {len(split_data)} samples -> {out_path}")

    return train, val, test


if __name__ == "__main__":
    prepare_splits("data/annotations/weak_labeled.json")
    print(f"\nBIO labels ({len(BIO_LABELS)}): {BIO_LABELS}")
