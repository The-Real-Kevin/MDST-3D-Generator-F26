"""
Loads the synthetic (sentence -> parameters) CSV and converts it into the
tensors the training loop needs: class indices for categorical parameters,
[0,1]-normalized floats for continuous parameters (matching the sigmoid
scaling model.decode_outputs expects), and 0/1 floats for binary parameters.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_SCHEMA, PARAM_BY_NAME  # noqa: E402


def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_targets(rows: list[dict]) -> dict:
    """
    Returns {param_name: np.ndarray}, one array per parameter:
        categorical -> int64 class indices, shape (N,)
        continuous  -> float32 in [0,1] (normalized), shape (N,)
        binary      -> float32 0.0/1.0, shape (N,)
    Also returns the raw (unnormalized) continuous values under
    "<name>__raw" keys, handy for reporting human-readable metrics
    (e.g. density MAE in actual 0-1 generator units, not normalized units).
    """
    targets = {}
    for spec in PARAM_SCHEMA:
        col = [r[spec.name] for r in rows]
        if spec.kind == "categorical":
            targets[spec.name] = np.array([spec.choices.index(v) for v in col], dtype=np.int64)
        elif spec.kind == "continuous":
            raw = np.array([float(v) for v in col], dtype=np.float32)
            low, high = spec.value_range
            targets[spec.name] = ((raw - low) / (high - low)).astype(np.float32)
            targets[f"{spec.name}__raw"] = raw
        elif spec.kind == "binary":
            targets[spec.name] = np.array([1.0 if v == "True" else 0.0 for v in col], dtype=np.float32)
    return targets


def train_val_split(n: int, val_fraction: float = 0.15, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    return idx[n_val:], idx[:n_val]  # train_idx, val_idx
