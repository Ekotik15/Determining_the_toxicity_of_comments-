from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1_toxic: float
    f1_macro: float
    pr_auc: float
    roc_auc: float

    def as_record(self, model: str, split: str, threshold: float) -> dict[str, float | str]:
        record: dict[str, float | str] = {
            "model": model,
            "split": split,
            "threshold": threshold,
        }
        record.update(asdict(self))
        return record


def select_threshold(labels: pd.Series, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return 0.5
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def calculate_metrics(
    labels: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> Metrics:
    predictions = (scores >= threshold).astype(int)
    return Metrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1_toxic=float(f1_score(labels, predictions, zero_division=0)),
        f1_macro=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        pr_auc=float(average_precision_score(labels, scores)),
        roc_auc=float(roc_auc_score(labels, scores)),
    )
