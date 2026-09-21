import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_recall_curve
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from toxic_comments.data import load_splits
from toxic_comments.evaluation import calculate_metrics, select_threshold

MODEL_LABEL = "DistilBERT (fine-tuned)"


class TokenizedTextDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series,
        labels: pd.Series,
        tokenizer: object,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            max_length=max_length,
        )
        self.labels = labels.to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {name: values[index] for name, values in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, object]:
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate a Transformer on the fixed project splits."
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("transformer-output"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _positive_scores(prediction_output: object) -> np.ndarray:
    logits = np.asarray(prediction_output.predictions)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def _trainer_metrics(eval_prediction: object) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.asarray(logits).argmax(axis=1)
    return {"f1_toxic": float(f1_score(labels, predictions, zero_division=0))}


def _update_metrics(
    metrics_path: Path,
    validation_record: dict[str, float | str],
    test_record: dict[str, float | str],
) -> pd.DataFrame:
    new_rows = pd.DataFrame([validation_record, test_record])
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics = metrics[metrics["model"] != MODEL_LABEL]
        metrics = pd.concat([metrics, new_rows], ignore_index=True)
    else:
        metrics = new_rows
    metrics.to_csv(metrics_path, index=False)
    return metrics


def _save_comparison_plots(
    labels: pd.Series,
    transformer_scores: np.ndarray,
    threshold: float,
    artifacts_dir: Path,
) -> None:
    transformer_predictions = (transformer_scores >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        labels,
        transformer_predictions,
        display_labels=["non-toxic", "toxic"],
        cmap="Blues",
        colorbar=False,
    )
    plt.title(f"Test confusion matrix: {MODEL_LABEL}")
    plt.tight_layout()
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    classical_path = artifacts_dir / "test_predictions.parquet"
    if classical_path.exists():
        classical = pd.read_parquet(classical_path)
        precision, recall, _ = precision_recall_curve(classical["label"], classical["score"])
        plt.plot(recall, precision, label="Word+char TF-IDF + Linear SVM")
    precision, recall, _ = precision_recall_curve(labels, transformer_scores)
    plt.plot(recall, precision, label=MODEL_LABEL)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curves on the test split")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "precision_recall_curves.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(
        cache_dir=args.cache_dir,
        random_state=args.random_state,
        sample_size=args.sample_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = TokenizedTextDataset(
        splits.train["text"],
        splits.train["label"],
        tokenizer,
        args.max_length,
    )
    validation_dataset = TokenizedTextDataset(
        splits.validation["text"],
        splits.validation["label"],
        tokenizer,
        args.max_length,
    )
    test_dataset = TokenizedTextDataset(
        splits.test["text"],
        splits.test["label"],
        tokenizer,
        args.max_length,
    )

    class_counts = np.bincount(splits.train["label"].to_numpy(dtype=int), minlength=2)
    class_weights = torch.tensor(
        [1.0, class_counts[0] / class_counts[1]],
        dtype=torch.float32,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    has_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_toxic",
        greater_is_better=True,
        save_total_limit=1,
        fp16=has_cuda,
        dataloader_num_workers=2,
        dataloader_pin_memory=has_cuda,
        report_to="none",
        seed=args.random_state,
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()

    validation_scores = _positive_scores(trainer.predict(validation_dataset))
    threshold = select_threshold(splits.validation["label"], validation_scores)
    test_scores = _positive_scores(trainer.predict(test_dataset))
    validation_metrics = calculate_metrics(
        splits.validation["label"],
        validation_scores,
        threshold,
    )
    test_metrics = calculate_metrics(splits.test["label"], test_scores, threshold)
    metrics = _update_metrics(
        args.artifacts_dir / "metrics.csv",
        validation_metrics.as_record(MODEL_LABEL, "validation", threshold),
        test_metrics.as_record(MODEL_LABEL, "test", threshold),
    )

    transformer_model_dir = args.artifacts_dir / "transformer_model"
    trainer.save_model(transformer_model_dir)
    tokenizer.save_pretrained(transformer_model_dir)
    metadata = {
        "model": MODEL_LABEL,
        "backbone": args.model_name,
        "threshold": threshold,
        "selection_metric": "validation f1_toxic",
        "random_state": args.random_state,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "positive_class_weight": class_weights[1].item(),
        "device": "cuda" if has_cuda else "cpu",
    }
    (args.artifacts_dir / "transformer_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    predictions = splits.test.copy()
    predictions["score"] = test_scores
    predictions["prediction"] = (test_scores >= threshold).astype(int)
    predictions.to_parquet(
        args.artifacts_dir / "transformer_test_predictions.parquet",
        index=False,
    )
    _save_comparison_plots(
        splits.test["label"],
        test_scores,
        threshold,
        args.artifacts_dir,
    )

    print(metrics.to_string(index=False))
    print(f"\nValidation-selected threshold: {threshold:.4f}")
    print(f"Saved Transformer artifacts to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
