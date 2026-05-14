import numpy as np
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


def compute_metrics(eval_pred, bio_labels):
    predictions, label_ids = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for pred, label in zip(predictions, label_ids):
        true_seq = []
        pred_seq = []
        for p, l in zip(pred, label):
            if l != -100:
                true_seq.append(bio_labels[l])
                pred_seq.append(bio_labels[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    report = classification_report(true_labels, pred_labels, mode="strict", scheme=IOB2, output_dict=True)

    result = {}
    base_labels = ["PERSONAJE", "FACCION", "LUGAR", "ARTEFACTO_MAGICO", "RAZA"]
    for label in base_labels:
        if label in report:
            result[f"{label}_precision"] = report[label]["precision"]
            result[f"{label}_recall"] = report[label]["recall"]
            result[f"{label}_f1"] = report[label]["f1-score"]

    micro = report.get("micro avg", {})
    result["overall_precision"] = micro.get("precision", 0)
    result["overall_recall"] = micro.get("recall", 0)
    result["overall_f1"] = micro.get("f1-score", 0)
    result["overall_accuracy"] = accuracy_score(true_labels, pred_labels)

    return result
