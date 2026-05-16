import json
from pathlib import Path
import yaml


def load_config(config_path="configs/llm_generation.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("merge", {})


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_entity(text, entity):
    start, end, label = entity

    if not isinstance(start, int) or not isinstance(end, int):
        return False

    if start < 0 or end > len(text) or start >= end:
        return False

    valid_labels = ["PERSONAJE", "FACCION", "LUGAR", "ARTEFACTO_MAGICO", "RAZA"]
    if label not in valid_labels:
        return False

    return True


def normalize_entity(text, entity):
    start, end, label = entity

    while start < end and text[start] in " \t\n\r":
        start += 1

    while end > start and text[end - 1] in " \t\n\r.,;:!?\"'":
        end -= 1

    return [start, end, label]


def deduplicate_dataset(dataset):
    seen_texts = set()
    unique = []

    for item in dataset:
        text_key = item["text"].strip().lower()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique.append(item)

    removed = len(dataset) - len(unique)
    print(f"  Removed {removed} duplicate texts")
    return unique


def merge_datasets(
    paths=None,
    output_path="data/annotations/combined.json",
    deduplicate=True,
    validate_offsets=True,
):
    merge_config = load_config()

    if paths is None:
        paths = [
            "data/annotations/llm_synthetic.json",
            "data/annotations/llm_preannotated.json",
            "data/annotations/weak_labeled.json",
        ]

    output_path = merge_config.get("output_path", output_path)
    deduplicate = merge_config.get("deduplicate", deduplicate)
    validate_offsets = merge_config.get("validate_offsets", validate_offsets)

    print("Merging datasets...")

    all_items = []
    source_stats = {}

    for path in paths:
        if not Path(path).exists():
            print(f"  Skipping {path} (not found)")
            continue

        data = load_json(path)
        source_name = Path(path).stem

        valid_count = 0
        invalid_count = 0

        for item in data:
            if "text" not in item or "entities" not in item:
                invalid_count += 1
                continue

            text = item["text"]
            entities = item["entities"]

            if validate_offsets:
                validated_entities = []
                for entity in entities:
                    if validate_entity(text, entity):
                        normalized = normalize_entity(text, entity)
                        validated_entities.append(normalized)
                    else:
                        invalid_count += 1
                        continue
                item["entities"] = validated_entities

            if item["entities"]:
                item["source"] = source_name
                all_items.append(item)
                valid_count += 1

        source_stats[source_name] = {"valid": valid_count, "invalid": invalid_count}
        print(f"  {source_name}: {valid_count} valid, {invalid_count} invalid")

    print(f"\nTotal before deduplication: {len(all_items)}")

    if deduplicate:
        all_items = deduplicate_dataset(all_items)

    print(f"Total after deduplication: {len(all_items)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"\nMerged dataset saved to: {output_path}")

    label_counts = {}
    source_label_counts = {}
    for item in all_items:
        source = item.get("source", "unknown")
        if source not in source_label_counts:
            source_label_counts[source] = {}
        for _, _, label in item["entities"]:
            label_counts[label] = label_counts.get(label, 0) + 1
            source_label_counts[source][label] = (
                source_label_counts[source].get(label, 0) + 1
            )

    print("\nOverall entity distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    print("\nPer-source distribution:")
    for source, counts in source_label_counts.items():
        print(f"  {source}:")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {label}: {count}")

    return all_items


if __name__ == "__main__":
    merge_datasets()
