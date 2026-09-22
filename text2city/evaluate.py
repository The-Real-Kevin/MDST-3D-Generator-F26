#!/usr/bin/env python3
"""
Step 5: evaluate a trained checkpoint.

Two things this checks, deliberately kept separate:
  1. Held-out synthetic examples (same templates/phrase banks as training,
     just examples the model didn't train on) -- tells you if the model
     learned the mapping at all.
  2. Hand-written sentences that NO template in generate_dataset.py could
     have produced -- tells you if it generalizes to real phrasing, which
     is what actually matters once a real person is typing into this.
     This is the more important number. A model that aces (1) but fails
     (2) has memorized the templates, not learned the task -- and this is
     exactly where the tfidf embedder is expected to struggle (it only
     "knows" words it saw while fitting) versus a real sentence-transformer
     (which should handle a paraphrase like "surrounded by peaks" even
     though only "surrounded by mountains" was in the training data).

Usage:
    python text2city/evaluate.py --checkpoint text2city/checkpoint
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_SCHEMA
from text2city.dataset import load_rows, build_targets, train_val_split
from text2city.embedders import load_embedder
from text2city.model import TextToCityParams, decode_batch
from text2city.freshness import check_staleness


# Hand-written sentences -- deliberately phrased differently from every
# template/phrase-bank entry in generate_dataset.py, to test generalization
# rather than memorization. Expected values are approximate (this is a
# qualitative check, not a strict pass/fail).
HANDWRITTEN_EXAMPLES = [
    "a huge sprawling metropolis packed with skyscrapers",
    "a quiet little town in the mountains with hardly any buildings",
    "a city built along a big river, with lots of parks on both banks",
    "an ultra futuristic city with several separate business hubs",
    "a small fishing village by the coast",
    "a flat desert city with almost no water and very few trees",
]


def load_checkpoint(checkpoint_dir: str, dataset_path: str = "data/text2city_dataset.csv"):
    ckpt_dir = Path(checkpoint_dir)
    ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    check_staleness(ckpt.get("fingerprint"), dataset_path)
    model = TextToCityParams(embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    embedder_ext = "pkl" if ckpt["embedder_kind"] == "tfidf" else "txt"
    embedder = load_embedder(ckpt["embedder_kind"], str(ckpt_dir / f"embedder.{embedder_ext}"))
    return model, embedder


def predict(model, embedder, texts: list[str]) -> list[dict]:
    embeddings = torch.tensor(embedder.encode(texts))
    with torch.no_grad():
        raw = model(embeddings)
    return [decode_batch(raw, i) for i in range(len(texts))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="text2city/checkpoint")
    parser.add_argument("--dataset", type=str, default="data/text2city_dataset.csv")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, embedder = load_checkpoint(args.checkpoint, args.dataset)

    # --- 1. held-out synthetic examples ---
    rows = load_rows(args.dataset)
    texts = [r["text"] for r in rows]
    targets = build_targets(rows)
    _, val_idx = train_val_split(len(rows), args.val_fraction, seed=args.seed)

    val_texts = [texts[i] for i in val_idx]
    predictions = predict(model, embedder, val_texts)

    print("=== 1. held-out synthetic examples ===")
    for spec in PARAM_SCHEMA:
        if spec.kind == "categorical":
            correct = sum(1 for i, p in zip(val_idx, predictions) if p[spec.name] == rows[i][spec.name])
            print(f"  {spec.name:20s} accuracy = {correct/len(val_idx):.3f}")
        elif spec.kind == "binary":
            correct = sum(1 for i, p in zip(val_idx, predictions) if str(p[spec.name]) == rows[i][spec.name])
            print(f"  {spec.name:20s} accuracy = {correct/len(val_idx):.3f}")
        else:
            preds = np.array([p[spec.name] for p in predictions])
            actual = targets[f"{spec.name}__raw"][val_idx]
            print(f"  {spec.name:20s} MAE = {np.mean(np.abs(preds - actual)):.4f}")

    # --- 2. hand-written, never-templated sentences (qualitative) ---
    print("\n=== 2. hand-written sentences (not from any template -- generalization check) ===")
    hw_predictions = predict(model, embedder, HANDWRITTEN_EXAMPLES)
    for text, pred in zip(HANDWRITTEN_EXAMPLES, hw_predictions):
        print(f"\n  \"{text}\"")
        for spec in PARAM_SCHEMA:
            val = pred[spec.name]
            if isinstance(val, float):
                val = round(val, 3)
            print(f"      {spec.name:20s} = {val}")


if __name__ == "__main__":
    main()
