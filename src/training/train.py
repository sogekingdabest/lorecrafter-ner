import os
import json
import mlflow
import mlflow.transformers
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

from src.training.config import load_config
from src.training.evaluate import compute_metrics
from src.dataset.preprocess import BIO_LABELS, BIO2ID


def load_dataset(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_and_align_labels(examples, tokenizer, max_length=128):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
    )

    labels = []
    for i, tag_list in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])
        previous_word_idx = None
        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                label_ids[j] = BIO2ID.get(tag_list[word_idx], BIO2ID["O"])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train():
    config = load_config()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("LoreCrafter-NER")

    with mlflow.start_run() as run:
        mlflow.log_params(config)

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForTokenClassification.from_pretrained(
            config["model_name"],
            num_labels=len(BIO_LABELS),
            id2label={i: l for i, l in enumerate(BIO_LABELS)},
            label2id=BIO2ID,
            ignore_mismatched_sizes=True,
        )

        train_data = load_dataset("data/processed/train.json")
        val_data = load_dataset("data/processed/val.json")

        tokenized_train = train_data.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, config["max_length"]),
            batched=True,
        )
        tokenized_val = val_data.map(
            lambda x: tokenize_and_align_labels(x, tokenizer, config["max_length"]),
            batched=True,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        training_args = TrainingArguments(
            output_dir="models/lorecrafter-ner",
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["num_epochs"],
            weight_decay=config["weight_decay"],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="overall_f1",
            seed=config["seed"],
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, BIO_LABELS),
        )

        train_result = trainer.train()
        trainer.save_model("models/lorecrafter-ner")
        tokenizer.save_pretrained("models/lorecrafter-ner")

        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_epochs": config["num_epochs"],
        })

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
        )

        print(f"\nModelo guardado en models/lorecrafter-ner | Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train()
