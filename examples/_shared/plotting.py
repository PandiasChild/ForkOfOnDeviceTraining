"""Matplotlib helpers for compare.py across all four examples.

Convention: PyTorch curves are solid lines; C curves are dashed lines.
Two colors per chart, one per split (train/val).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # no GUI in CI / headless dev shells
import matplotlib.pyplot as plt
import numpy as np

from examples._shared.log_schema import RunLog


def plot_loss_curves(out_path: Path | str, pt_log: RunLog, c_log: RunLog) -> None:
    pt_train = [e["train_loss"] for e in pt_log["epochs"]]
    pt_val   = [e["val_loss"]   for e in pt_log["epochs"]]
    c_train  = [e["train_loss"] for e in c_log["epochs"]]
    c_val    = [e["val_loss"]   for e in c_log["epochs"]]
    n = max(len(pt_train), len(c_train))
    x = list(range(n))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x[:len(pt_train)], pt_train, "-",  color="#1f77b4", label="PyTorch train")
    ax.plot(x[:len(pt_val)],   pt_val,   "-",  color="#ff7f0e", label="PyTorch val")
    ax.plot(x[:len(c_train)],  c_train,  "--", color="#1f77b4", label="C train")
    ax.plot(x[:len(c_val)],    c_val,    "--", color="#ff7f0e", label="C val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{pt_log['example']} — train/val loss (PyTorch solid, C dashed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_accuracy_curves(out_path: Path | str, pt_log: RunLog, c_log: RunLog) -> None:
    pt_acc = [e["val_acc"] for e in pt_log["epochs"]]
    c_acc  = [e["val_acc"] for e in c_log["epochs"]]
    n = max(len(pt_acc), len(c_acc))
    x = list(range(n))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x[:len(pt_acc)], pt_acc, "-",  color="#2ca02c", label="PyTorch val acc")
    ax.plot(x[:len(c_acc)],  c_acc,  "--", color="#2ca02c", label="C val acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.set_title(f"{pt_log['example']} — validation accuracy (PyTorch solid, C dashed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    out_path: Path | str,
    cm: np.ndarray,  # shape [num_classes, num_classes]; cm[i, j] = count predicted=i, actual=j
    class_names: Iterable[str],
    title: str,
) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    names = list(class_names)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
