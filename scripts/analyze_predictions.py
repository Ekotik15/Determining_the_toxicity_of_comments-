import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create uncertainty estimates and an error-analysis sample."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/test_predictions.parquet"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/model_metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _metric_values(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1 toxic": float(f1_score(labels, predictions, zero_division=0)),
        "PR-AUC": float(average_precision_score(labels, scores)),
        "ROC-AUC": float(roc_auc_score(labels, scores)),
    }


def _bootstrap_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    iterations: int,
    random_state: int,
) -> pd.DataFrame:
    if iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")

    rng = np.random.default_rng(random_state)
    class_zero = np.flatnonzero(labels == 0)
    class_one = np.flatnonzero(labels == 1)
    observed = _metric_values(labels, scores, predictions)
    samples = {name: [] for name in observed}

    for _ in range(iterations):
        sampled = np.concatenate(
            [
                rng.choice(class_zero, size=len(class_zero), replace=True),
                rng.choice(class_one, size=len(class_one), replace=True),
            ]
        )
        values = _metric_values(labels[sampled], scores[sampled], predictions[sampled])
        for name, value in values.items():
            samples[name].append(value)

    rows = []
    for name, point in observed.items():
        lower, upper = np.quantile(samples[name], [0.025, 0.975])
        rows.append(
            {
                "metric": name,
                "point_estimate": point,
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "bootstrap_iterations": iterations,
            }
        )
    return pd.DataFrame(rows)


def _compact_text(value: object, limit: int = 180) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _error_sample(predictions: pd.DataFrame) -> pd.DataFrame:
    false_positives = (
        predictions[(predictions["label"] == 0) & (predictions["prediction"] == 1)]
        .sort_values("score", ascending=False)
        .head(3)
        .copy()
    )
    false_positives["error_type"] = "False positive"

    false_negatives = (
        predictions[(predictions["label"] == 1) & (predictions["prediction"] == 0)]
        .sort_values("score", ascending=True)
        .head(3)
        .copy()
    )
    false_negatives["error_type"] = "False negative"

    sample = pd.concat([false_positives, false_negatives], ignore_index=True)
    sample["text"] = sample["text"].map(_compact_text)
    return sample[["error_type", "text", "label", "score", "prediction"]]


def _threshold_comparison(
    predictions: pd.DataFrame,
    metadata_path: Path,
) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_threshold = float(metadata["threshold"])
    labels = predictions["label"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)

    rows = []
    for name, threshold in [("Default", 0.5), ("Validation-selected", selected_threshold)]:
        predicted = (scores >= threshold).astype(int)
        values = _metric_values(labels, scores, predicted)
        rows.append(
            {
                "threshold_policy": name,
                "threshold": threshold,
                "accuracy": float((predicted == labels).mean()),
                "precision": values["Precision"],
                "recall": values["Recall"],
                "f1_toxic": values["F1 toxic"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    required = {"text", "label", "score", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Predictions must contain columns: {sorted(required)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = predictions["label"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    predicted = predictions["prediction"].to_numpy(dtype=int)

    intervals = _bootstrap_intervals(
        labels,
        scores,
        predicted,
        iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )
    intervals.to_csv(args.output_dir / "bootstrap_ci.csv", index=False)
    _error_sample(predictions).to_csv(args.output_dir / "error_analysis.csv", index=False)
    _threshold_comparison(predictions, args.metadata).to_csv(
        args.output_dir / "threshold_comparison.csv",
        index=False,
    )

    tn = int(((labels == 0) & (predicted == 0)).sum())
    fp = int(((labels == 0) & (predicted == 1)).sum())
    fn = int(((labels == 1) & (predicted == 0)).sum())
    tp = int(((labels == 1) & (predicted == 1)).sum())
    summary = {
        "rows": int(len(predictions)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "false_positive_rate": fp / (fp + tn),
        "false_negative_rate": fn / (fn + tp),
        "bootstrap_method": "stratified non-parametric bootstrap of the fixed test predictions",
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_state": args.random_state,
    }
    (args.output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(intervals.to_string(index=False))
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
