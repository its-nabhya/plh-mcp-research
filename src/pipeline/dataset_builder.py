"""
DatasetBuilder — aggregates raw JSONL logs into train/val/test splits.
All field access is defensive (.get with defaults) to handle partial entries
written during failed or partial Ollama/Gemini runs.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.features.extractor import extract_dataset, ALL_FEATURE_NAMES
from src.logging.logger import InteractionLogger, summarise_log
from src.labeling.labeler import label_quality_report

import numpy as np


class DatasetBuilder:
    def __init__(self, logs_dir="data/logs"):
        self.logs_dir = Path(logs_dir)

    def load_all(self, glob_pattern: str = "*.jsonl") -> List[Dict[str, Any]]:
        entries = []
        for p in sorted(self.logs_dir.glob(glob_pattern)):
            entries.extend(InteractionLogger.read_all(p))
        print(f"Loaded {len(entries)} entries from {self.logs_dir}")
        return entries

    def filter(
        self,
        entries: List[Dict[str, Any]],
        exclude_exploratory: bool = True,
        exclude_api_errors: bool = True,
        min_label_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        filtered = []
        for e in entries:
            label      = e.get("label") or {}
            tool_call  = e.get("tool_call") or {}
            tool_name  = tool_call.get("name", "")

            if exclude_exploratory and label.get("is_exploratory"):
                continue
            if exclude_api_errors and tool_name == "__api_error__":
                continue
            if label.get("confidence", 1.0) < min_label_confidence:
                continue
            filtered.append(e)

        print(f"After filtering: {len(filtered)} entries (was {len(entries)})")
        return filtered

    def split(
        self,
        entries: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratify: bool = True,
    ) -> Tuple[List, List, List]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        random.seed(seed)

        if not stratify:
            shuffled = entries[:]
            random.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            return (shuffled[:n_train],
                    shuffled[n_train:n_train + n_val],
                    shuffled[n_train + n_val:])

        # Stratify by primary label
        buckets: Dict[str, List] = defaultdict(list)
        for e in entries:
            lab = (e.get("label") or {}).get("primary") or "valid"
            buckets[lab].append(e)

        train, val, test = [], [], []
        for lab, items in buckets.items():
            random.shuffle(items)
            n    = len(items)
            n_tr = max(1, int(n * train_ratio))
            n_va = max(0, int(n * val_ratio))
            train.extend(items[:n_tr])
            val.extend(items[n_tr:n_tr + n_va])
            test.extend(items[n_tr + n_va:])

        return train, val, test

    def save_splits(
        self,
        train: List, val: List, test: List,
        output_dir="data/features",
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, subset in [("train", train), ("val", val), ("test", test)]:
            path = out / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as fh:
                for e in subset:
                    fh.write(json.dumps(e, default=str) + "\n")
            print(f"Saved {len(subset)} → {path}")

    def build_numpy_dataset(
        self,
        entries: List[Dict[str, Any]],
        binary: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return extract_dataset(entries, binary=binary)

    def summary_report(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = summarise_log(entries)
        quality = label_quality_report(entries)
        return {**summary, "label_quality": quality,
                "feature_names": ALL_FEATURE_NAMES}