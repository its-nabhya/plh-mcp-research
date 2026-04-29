"""
Tests for the feature extractor and label encoder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.features.extractor import (
    FeatureExtractor, LabelEncoder, extract_dataset,
    ALL_FEATURE_NAMES,
)


def _make_entry(tool_exists=True, schema_valid=True, permission=True,
                state=None, primary=None, conf_score=0.5,
                high_words=None, low_words=None, discovery=False,
                turn=1, tool_name="add_numbers"):
    return {
        "tool_call": {"name": tool_name, "parameters": {}},
        "validation": {
            "tool_exists": tool_exists,
            "schema_valid": schema_valid,
            "permission_granted": permission,
            "state_valid": state,
            "violations": [],
        },
        "agent_reasoning": {
            "confidence_signals": {
                "score": conf_score,
                "high_conf_words": high_words or [],
                "low_conf_words": low_words or [],
                "discovery_before_invocation": discovery,
                "tools_queried": ["list_tools"] if discovery else [],
            },
        },
        "conversation_turn": turn,
        "label": {"primary": primary, "is_hallucination": primary is not None},
    }


class TestFeatureExtractor:
    fe = FeatureExtractor()

    def test_output_shape(self):
        entry = _make_entry()
        vec = self.fe.extract(entry)
        assert vec.shape == (len(ALL_FEATURE_NAMES),)

    def test_tool_exists_flag(self):
        e_valid = _make_entry(tool_exists=True)
        e_invalid = _make_entry(tool_exists=False)
        v = self.fe.extract(e_valid)
        iv = self.fe.extract(e_invalid)
        idx = ALL_FEATURE_NAMES.index("tool_exists")
        assert v[idx] == 1.0
        assert iv[idx] == 0.0

    def test_confidence_score_propagation(self):
        e = _make_entry(conf_score=0.9)
        vec = self.fe.extract(e)
        idx = ALL_FEATURE_NAMES.index("confidence_score")
        assert abs(vec[idx] - 0.9) < 1e-5

    def test_unknown_permission_is_half(self):
        e = _make_entry(permission=None)
        vec = self.fe.extract(e)
        idx = ALL_FEATURE_NAMES.index("permission_granted")
        assert abs(vec[idx] - 0.5) < 1e-5

    def test_discovery_flag(self):
        e_disc = _make_entry(discovery=True)
        e_no   = _make_entry(discovery=False)
        vec_d = self.fe.extract(e_disc)
        vec_n = self.fe.extract(e_no)
        idx = ALL_FEATURE_NAMES.index("used_discovery_tools")
        assert vec_d[idx] == 1.0
        assert vec_n[idx] == 0.0

    def test_batch_shape(self):
        entries = [_make_entry() for _ in range(5)]
        X = self.fe.extract_batch(entries)
        assert X.shape == (5, len(ALL_FEATURE_NAMES))


class TestLabelEncoder:
    enc = LabelEncoder()

    def test_valid_label_encodes(self):
        idx = self.enc.encode(None)
        assert self.enc.decode(idx) == "valid"

    def test_roundtrip(self):
        from src.labeling.taxonomy import ALL_KEYS
        for key in ALL_KEYS:
            idx = self.enc.encode(key)
            assert self.enc.decode(idx) == key

    def test_binary_encode(self):
        assert self.enc.binary_encode(None) == 0
        assert self.enc.binary_encode("tool_existence_hallucination") == 1


class TestExtractDataset:
    def test_binary_output(self):
        entries = [
            _make_entry(primary="tool_existence_hallucination"),
            _make_entry(primary=None),
            _make_entry(primary="parameter_schema_hallucination"),
        ]
        X, y = extract_dataset(entries, binary=True)
        assert X.shape[0] == 3
        assert list(y) == [1, 0, 1]

    def test_multiclass_output(self):
        entries = [_make_entry(primary=None), _make_entry(primary="tool_existence_hallucination")]
        X, y = extract_dataset(entries, binary=False)
        assert X.shape[0] == 2
        assert y[0] != y[1]
