"""Test examples/_shared/log_schema.py."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from examples._shared.log_schema import EpochLog, FinalLog, RunLog, dump_log, load_log


def test_dump_then_load_roundtrips(tmp_path):
    log: RunLog = {
        "impl": "pytorch",
        "example": "har_classifier",
        "config": {
            "epochs": 2, "batch": 64, "lr": 0.01, "momentum": 0.9,
            "seed": 42, "shuffle_seed": 42,
        },
        "epochs": [
            {"epoch": 0, "step_losses": [1.5, 1.2, 1.0], "train_loss": 1.23,
             "val_loss": 0.85, "val_acc": 0.71, "wall_s": 1.4},
            {"epoch": 1, "step_losses": [0.8, 0.6, 0.5], "train_loss": 0.62,
             "val_loss": 0.42, "val_acc": 0.88, "wall_s": 1.3},
        ],
        "final": {"test_loss": 0.4, "test_acc": 0.9, "test_auc": None},
    }
    path = tmp_path / "log.json"
    dump_log(path, log)
    loaded = load_log(path)
    assert loaded == log


def test_load_validates_required_top_level_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"impl": "pytorch"}))  # missing fields
    try:
        load_log(path)
    except KeyError as e:
        assert "config" in str(e) or "epochs" in str(e)
    else:
        raise AssertionError("expected KeyError on missing required top-level key")
