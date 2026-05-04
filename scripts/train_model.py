"""
train_model.py — train baseline PLH detection models.

Usage:
  python scripts/train_model.py --data data/features --model lr
  python scripts/train_model.py --data data/features --model lr --multiclass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.extractor import extract_dataset, ALL_FEATURE_NAMES
from src.logging.logger import InteractionLogger
from src.models.baselines.logistic_regression import PLHLogisticRegression


def load_split(data_dir: Path, split: str):
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    return InteractionLogger.read_all(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/features", help="Splits directory")
    p.add_argument("--model", default="lr", choices=["lr"],
                   help="Model type to train")
    p.add_argument("--output", default="data/models",
                   help="Directory to save trained model")
    p.add_argument("--multiclass", action="store_true",
                   help="Train 16-class model instead of binary")
    args = p.parse_args()

    data_dir = Path(args.data)
    binary = not args.multiclass

    print(f"Loading splits from {data_dir} ...")
    train_entries = load_split(data_dir, "train")
    val_entries   = load_split(data_dir, "val")
    test_entries  = load_split(data_dir, "test")

    if not train_entries:
        print("No training data found. Run run_pipeline.py first, then dataset_builder.")
        sys.exit(1)

    X_train, y_train = extract_dataset(train_entries, binary=binary)
    X_val, y_val     = extract_dataset(val_entries,   binary=binary)
    X_test, y_test   = extract_dataset(test_entries,  binary=binary)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"Class distribution (train): { {int(k): int(v) for k,v in zip(*_counts(y_train))} }")

    if args.model == "lr":
        model = PLHLogisticRegression(binary=binary)

    print("\nTraining ...")
    model.fit(X_train, y_train)

    print("\n--- Validation Metrics ---")
    val_metrics = model.evaluate(X_val, y_val)
    _print_metrics(val_metrics)

    print("\n--- Test Metrics ---")
    test_metrics = model.evaluate(X_test, y_test)
    _print_metrics(test_metrics)

    if hasattr(model, "feature_importance"):
        print("\n--- Feature Importance (top 10) ---")
        for name, imp in model.feature_importance(ALL_FEATURE_NAMES)[:10]:
            print(f"  {name:35s} {imp:.4f}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "binary" if binary else "multiclass"
    save_path = out_dir / f"{args.model}_{mode_tag}.pkl"
    model.save(save_path)
    print(f"\nModel saved → {save_path}")

    # Save metrics
    results = {"val": val_metrics, "test": test_metrics}
    metrics_path = out_dir / f"{args.model}_{mode_tag}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Metrics saved → {metrics_path}")


def _counts(y):
    import numpy as np
    unique, counts = np.unique(y, return_counts=True)
    return unique, counts


def _print_metrics(metrics: dict):
    for k, v in metrics.items():
        if k == "classification_report":
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
