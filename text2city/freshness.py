"""
Staleness detection for checkpoints.

This exists because of a recurring, costly confusion during development:
editing generate_dataset.py, schema.py, or dataset.py does NOTHING to an
already-trained checkpoint sitting on disk -- you have to explicitly
re-run generate_dataset.py and train.py for changes to take effect. It's
easy to forget this and spend a long time debugging "why isn't my fix
working" when the real answer is "you're still running the old model."

fingerprint_sources() hashes the actual dataset CSV plus the source of the
files that decide what's IN that CSV. train.py stores this fingerprint in
the checkpoint; infer.py and evaluate.py recompute it fresh and warn
loudly if it doesn't match, rather than silently running on stale data.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Files whose content changes should invalidate a checkpoint if edited
# without retraining.
SOURCE_FILES = ["schema.py", "generate_dataset.py", "dataset.py", "model.py"]


def fingerprint_sources(dataset_path: str) -> str:
    hasher = hashlib.sha256()

    dataset_file = Path(dataset_path)
    if dataset_file.exists():
        hasher.update(dataset_file.read_bytes())
    else:
        hasher.update(b"MISSING_DATASET")

    this_dir = Path(__file__).resolve().parent
    for name in SOURCE_FILES:
        path = this_dir / name
        if path.exists():
            hasher.update(path.read_bytes())

    return hasher.hexdigest()[:16]


def check_staleness(checkpoint_fingerprint: str, dataset_path: str) -> None:
    """Prints a loud warning (does not raise) if the checkpoint looks stale."""
    current = fingerprint_sources(dataset_path)
    if checkpoint_fingerprint is None:
        print("WARNING: this checkpoint has no fingerprint recorded (trained with an "
              "older version of train.py) -- can't check staleness. Consider retraining.")
        return
    if current != checkpoint_fingerprint:
        print("=" * 70)
        print("WARNING: this checkpoint appears STALE.")
        print("The dataset CSV and/or schema.py / generate_dataset.py / dataset.py /")
        print("model.py have changed since this checkpoint was trained. Predictions")
        print("below reflect the OLD code, not what's currently on disk.")
        print("Fix: re-run generate_dataset.py, then train.py, then try again.")
        print("=" * 70)
