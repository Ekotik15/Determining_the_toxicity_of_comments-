import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from toxic_comments.data import DatasetSplits
from toxic_comments.evaluation import calculate_metrics, select_threshold

CLASSICAL_MODEL_NAMES = {
    "Dummy prior",
    "TF-IDF + Logistic Regression",
    "Word TF-IDF + Linear SVM",
    "Word+char TF-IDF + Linear SVM",
}


@dataclass(frozen=True)
class ExperimentResult:
    metrics: pd.DataFrame
    best_model: str
    threshold: float
    model_path: Path


def _word_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=max_features,
        sublinear_tf=True,
    )


def _word_char_features(max_features: int) -> FeatureUnion:
    word_features = _word_vectorizer(max_features)
    char_features = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=3,
        max_features=max_features,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word_features), ("char", char_features)])


def _logistic_pipeline(max_features: int, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("features", _word_vectorizer(max_features)),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _linear_svc_pipeline(
    max_features: int,
    random_state: int,
    include_char_features: bool,
) -> Pipeline:
    classifier = CalibratedClassifierCV(
        estimator=LinearSVC(C=1.5, class_weight="balanced", random_state=random_state),
        method="sigmoid",
        cv=3,
        n_jobs=-1,
    )
    features = (
        _word_char_features(max_features)
        if include_char_features
        else _word_vectorizer(max_features)
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def _save_dataset_summary(splits: DatasetSplits, output_path: Path) -> None:
    summary = {}
    for split_name, frame in [
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ]:
        summary[split_name] = {
            "rows": len(frame),
            "toxic_rows": int(frame["label"].sum()),
            "toxic_share": float(frame["label"].mean()),
            "median_characters": float(frame["text"].str.len().median()),
        }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _save_pr_curves(
    curves: dict[str, tuple[pd.Series, np.ndarray]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    for model_name, (labels, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.plot(recall, precision, label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curves on the test split")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_classical_experiment(
    splits: DatasetSplits,
    artifacts_dir: Path,
    max_features: int = 40_000,
    random_state: int = 42,
) -> ExperimentResult:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _save_dataset_summary(splits, artifacts_dir / "dataset_summary.json")

    train_texts = splits.train["text"]
    train_labels = splits.train["label"]
    validation_texts = splits.validation["text"]
    validation_labels = splits.validation["label"]
    test_texts = splits.test["text"]
    test_labels = splits.test["label"]

    rows: list[dict[str, float | str]] = []
    test_curves: dict[str, tuple[pd.Series, np.ndarray]] = {}
    fitted_models: dict[str, Pipeline | DummyClassifier] = {}
    thresholds: dict[str, float] = {}

    dummy = DummyClassifier(strategy="prior", random_state=random_state)
    dummy.fit(np.zeros((len(train_labels), 1)), train_labels)
    dummy_validation_scores = dummy.predict_proba(np.zeros((len(validation_labels), 1)))[:, 1]
    dummy_test_scores = dummy.predict_proba(np.zeros((len(test_labels), 1)))[:, 1]
    dummy_threshold = 0.5
    rows.append(
        calculate_metrics(
            validation_labels,
            dummy_validation_scores,
            dummy_threshold,
        ).as_record("Dummy prior", "validation", dummy_threshold)
    )
    rows.append(
        calculate_metrics(test_labels, dummy_test_scores, dummy_threshold).as_record(
            "Dummy prior", "test", dummy_threshold
        )
    )
    test_curves["Dummy prior"] = (test_labels, dummy_test_scores)
    fitted_models["Dummy prior"] = dummy
    thresholds["Dummy prior"] = dummy_threshold

    models = {
        "TF-IDF + Logistic Regression": _logistic_pipeline(max_features, random_state),
        "Word TF-IDF + Linear SVM": _linear_svc_pipeline(
            max_features,
            random_state,
            include_char_features=False,
        ),
        "Word+char TF-IDF + Linear SVM": _linear_svc_pipeline(
            max_features,
            random_state,
            include_char_features=True,
        ),
    }
    for model_name, model in models.items():
        model.fit(train_texts, train_labels)
        validation_scores = model.predict_proba(validation_texts)[:, 1]
        threshold = select_threshold(validation_labels, validation_scores)
        test_scores = model.predict_proba(test_texts)[:, 1]

        rows.append(
            calculate_metrics(validation_labels, validation_scores, threshold).as_record(
                model_name, "validation", threshold
            )
        )
        rows.append(
            calculate_metrics(test_labels, test_scores, threshold).as_record(
                model_name, "test", threshold
            )
        )
        test_curves[model_name] = (test_labels, test_scores)
        fitted_models[model_name] = model
        thresholds[model_name] = threshold

    classical_metrics = pd.DataFrame(rows)
    validation_metrics = classical_metrics[classical_metrics["split"] == "validation"]
    best_row = validation_metrics.loc[validation_metrics["f1_toxic"].idxmax()]
    best_model_name = str(best_row["model"])
    best_threshold = float(best_row["threshold"])
    best_model = fitted_models[best_model_name]

    model_path = artifacts_dir / "toxic_classifier.joblib"
    joblib.dump(best_model, model_path)
    metadata = {
        "model": best_model_name,
        "threshold": best_threshold,
        "selection_metric": "validation f1_toxic",
        "random_state": random_state,
    }
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    metrics_path = artifacts_dir / "metrics.csv"
    if metrics_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        preserved_metrics = existing_metrics[~existing_metrics["model"].isin(CLASSICAL_MODEL_NAMES)]
        metrics = pd.concat([classical_metrics, preserved_metrics], ignore_index=True)
    else:
        metrics = classical_metrics
    metrics.to_csv(metrics_path, index=False)

    best_test_scores = test_curves[best_model_name][1]
    best_test_predictions = (best_test_scores >= best_threshold).astype(int)
    predictions = splits.test.copy()
    predictions["score"] = best_test_scores
    predictions["prediction"] = best_test_predictions
    predictions.to_parquet(artifacts_dir / "test_predictions.parquet", index=False)

    ConfusionMatrixDisplay.from_predictions(
        test_labels,
        best_test_predictions,
        display_labels=["non-toxic", "toxic"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(f"Test confusion matrix: {best_model_name}")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    _save_pr_curves(test_curves, artifacts_dir / "precision_recall_curves.png")

    return ExperimentResult(
        metrics=metrics,
        best_model=best_model_name,
        threshold=best_threshold,
        model_path=model_path,
    )
