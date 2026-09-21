import argparse
from pathlib import Path

from toxic_comments import load_splits, run_classical_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the classical-model experiment.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--max-features", type=int, default=40_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = load_splits(
        cache_dir=args.cache_dir,
        random_state=args.random_state,
        sample_size=args.sample_size,
    )
    result = run_classical_experiment(
        splits=splits,
        artifacts_dir=args.artifacts_dir,
        max_features=args.max_features,
        random_state=args.random_state,
    )
    print(result.metrics.to_string(index=False))
    print(f"\nSelected model: {result.best_model}")
    print(f"Validation threshold: {result.threshold:.4f}")
    print(f"Saved model: {result.model_path}")


if __name__ == "__main__":
    main()
