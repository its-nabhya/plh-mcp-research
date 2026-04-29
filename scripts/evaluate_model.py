"""
evaluate_model.py — full evaluation including calibration, PHR, and scaling.

Usage:
  python scripts/evaluate_model.py --model data/models/lr_binary.pkl --data data/features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.evaluation.calibration import calibration_report
from src.evaluation.metrics import (
    compute_binary_metrics, phr_by_complexity, protocol_hallucination_rate,
)
from src.features.extractor import extract_dataset
from src.logging.logger import InteractionLogger
from src.models.baselines.logistic_regression import PLHLogisticRegression


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to saved .pkl model")
    p.add_argument("--data", default="data/features",
                   help="Directory containing test.jsonl")
    p.add_argument("--output", default=None,
                   help="Optional path to save JSON report")
    args = p.parse_args()

    # Load
    model = PLHLogisticRegression.load(args.model)
    test_path = Path(args.data) / "test.jsonl"
    if not test_path.exists():
        print(f"test.jsonl not found at {test_path}")
        sys.exit(1)

    entries = InteractionLogger.read_all(test_path)
    if not entries:
        print("Test set is empty.")
        sys.exit(1)

    X, y = extract_dataset(entries, binary=True)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Extract agent confidence scores for calibration comparison
    conf_scores = np.array(
        [e["agent_reasoning"]["confidence_signals"]["score"] for e in entries],
        dtype=float,
    )

    print(f"\n=== Test Set: {len(entries)} entries ===")
    print(f"PHR (ground truth): {protocol_hallucination_rate(entries):.4f}")

    print("\n--- Detection Metrics ---")
    det = compute_binary_metrics(y, y_pred, y_proba.reshape(-1, 1))
    for k, v in det.items():
        print(f"  {k}: {v}")

    print("\n--- Calibration Report ---")
    cal = calibration_report(y, y_proba, y_pred, conf_scores)
    for k, v in cal.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n--- PHR by World ---")
    by_world = phr_by_complexity(entries)
    for world, phr in by_world.items():
        print(f"  {world}: {phr:.4f}")

    report = {
        "total_test": len(entries),
        "phr_ground_truth": protocol_hallucination_rate(entries),
        "detection_metrics": det,
        "calibration": cal,
        "phr_by_world": by_world,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
