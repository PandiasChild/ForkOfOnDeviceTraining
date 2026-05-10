"""JSON log schema shared by train_pytorch.py and train_c.c.

The C side writes this format by hand using printf, so the schema here
is also the contract C must follow. Keep keys ASCII and avoid nested
constructs that complicate hand-written JSON emission.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class TrainConfig(TypedDict):
    epochs: int
    batch: int
    lr: float
    momentum: float
    seed: int
    shuffle_seed: int


class EpochLog(TypedDict):
    epoch: int
    step_losses: list[float]
    train_loss: float
    val_loss: float
    val_acc: float | None
    wall_s: float


class FinalLog(TypedDict):
    test_loss: float
    test_acc: float | None
    test_auc: float | None


class RunLog(TypedDict):
    impl: str  # "pytorch" or "c"
    example: str
    config: TrainConfig
    epochs: list[EpochLog]
    final: FinalLog


_REQUIRED_TOP = ("impl", "config", "epochs", "final", "example")


def dump_log(path: Path | str, log: RunLog) -> None:
    Path(path).write_text(json.dumps(log, indent=2))


def load_log(path: Path | str) -> RunLog:
    data = json.loads(Path(path).read_text())
    for key in _REQUIRED_TOP:
        if key not in data:
            raise KeyError(f"log file {path}: missing required key {key!r}")
    return data  # type: ignore[return-value]
