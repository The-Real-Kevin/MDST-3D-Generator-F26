#!/usr/bin/env python3
"""
Step 4: train the text-to-city-parameter model.

Example (recommended, real language understanding, needs internet once):
    python text2city/train.py --embedder sentence-transformer

Example (fully offline, no downloads, for dev/testing):
    python text2city/train.py --embedder tfidf

Both save a checkpoint to --out (default text2city/checkpoint/) containing
the trained trunk+heads weights and the fitted embedder, so evaluate.py and
infer.py can load everything from one place.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_SCHEMA
from text2city.dataset import load_rows, build_targets, train_val_split
from text2city.embedders import get_embedder
from text2city.model import TextToCityParams
from text2city.freshness import fingerprint_sources


def compute_class_weights(targets: dict, train_idx) -> dict:
    """
    Inverse-frequency class weights for each categorical parameter, computed
    from the training split. Needed because some categorical parameters have
    a schema default that ALSO happens to be one of the valid choices (e.g.
    city_archetype defaults to "monocentric", which is also a real choice) --
    so unmentioned examples inflate that class's frequency on top of its own
    share of explicit mentions. Left unweighted, cross-entropy loss will
    happily lean on that majority-class prior whenever a sentence's wording
    only weakly overlaps the training vocabulary, instead of learning to
    actually discriminate the minority classes. Weighting by inverse
    frequency removes that shortcut.
    """
    weights = {}
    for spec in PARAM_SCHEMA:
        if spec.kind != "categorical":
            continue
        train_targets = targets[spec.name][train_idx]
        counts = np.array([np.sum(train_targets == i) for i in range(len(spec.choices))], dtype=np.float32)
        counts = np.maximum(counts, 1)  # avoid divide-by-zero for an unseen class
        inv_freq = 1.0 / counts
        weights[spec.name] = torch.tensor(inv_freq / inv_freq.sum() * len(spec.choices), dtype=torch.float32)
    return weights


def compute_loss(raw_outputs: dict, targets: dict, device, class_weights: dict) -> tuple:
    """Returns (total_loss, dict of per-parameter loss values for logging)."""
    total = torch.tensor(0.0, device=device)
    per_param = {}
    for spec in PARAM_SCHEMA:
        pred = raw_outputs[spec.name]
        target = torch.tensor(targets[spec.name], device=device)
        if spec.kind == "categorical":
            loss = nn.functional.cross_entropy(pred, target, weight=class_weights[spec.name].to(device))
        elif spec.kind == "continuous":
            loss = nn.functional.mse_loss(torch.sigmoid(pred.squeeze(-1)), target)
        elif spec.kind == "binary":
            loss = nn.functional.binary_cross_entropy_with_logits(pred.squeeze(-1), target)
        total = total + loss
        per_param[spec.name] = loss.item()
    return total, per_param


@torch.no_grad()
def evaluate_split(model, embeddings, targets, idx, device) -> dict:
    model.eval()
    batch = torch.tensor(embeddings[idx], device=device)
    raw = model(batch)
    metrics = {}
    for spec in PARAM_SCHEMA:
        target = targets[spec.name][idx]
        if spec.kind == "categorical":
            pred_idx = torch.argmax(raw[spec.name], dim=1).cpu().numpy()
            metrics[spec.name] = float((pred_idx == target).mean())  # accuracy
        elif spec.kind == "continuous":
            low, high = spec.value_range
            pred_frac = torch.sigmoid(raw[spec.name].squeeze(-1)).cpu().numpy()
            pred_raw = low + pred_frac * (high - low)
            target_raw = targets[f"{spec.name}__raw"][idx]
            metrics[spec.name] = float(np.mean(np.abs(pred_raw - target_raw)))  # MAE in real units
        elif spec.kind == "binary":
            pred = (torch.sigmoid(raw[spec.name].squeeze(-1)) > 0.5).cpu().numpy().astype(np.float32)
            metrics[spec.name] = float((pred == target).mean())  # accuracy
    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/text2city_dataset.csv")
    parser.add_argument("--embedder", choices=["sentence-transformer", "tfidf"], default="sentence-transformer")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="text2city/checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")  # this model is intentionally small -- CPU is fine, no GPU needed

    print(f"loading {args.dataset} ...")
    rows = load_rows(args.dataset)
    texts = [r["text"] for r in rows]
    targets = build_targets(rows)
    n = len(rows)
    train_idx, val_idx = train_val_split(n, args.val_fraction, seed=args.seed)
    print(f"{n} examples ({len(train_idx)} train / {len(val_idx)} val)")

    class_weights = compute_class_weights(targets, train_idx)
    for name, w in class_weights.items():
        print(f"  class weights for {name}: {[round(x, 2) for x in w.tolist()]}")

    print(f"building '{args.embedder}' embedder ...")
    t0 = time.time()
    embedder = get_embedder(args.embedder)
    embedder.fit([texts[i] for i in train_idx])
    embeddings = embedder.encode(texts)
    print(f"embedded {n} sentences into dim={embeddings.shape[1]} in {time.time() - t0:.1f}s")

    model = TextToCityParams(embedding_dim=embeddings.shape[1], hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\ntraining for {args.epochs} epochs ...")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        perm = rng.permutation(train_idx)
        epoch_loss = 0.0
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start:start + args.batch_size]
            batch_emb = torch.tensor(embeddings[batch_idx], device=device)
            batch_targets = {k: v[batch_idx] for k, v in targets.items()}

            raw = model(batch_emb)
            loss, _ = compute_loss(raw, batch_targets, device, class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = evaluate_split(model, embeddings, targets, val_idx, device)
            metric_str = "  ".join(f"{k}={v:.3f}" for k, v in list(val_metrics.items())[:4])
            print(f"  epoch {epoch:3d}  train_loss={epoch_loss/len(train_idx):.4f}  val: {metric_str} ...")

    elapsed = time.time() - t0
    print(f"\ntraining done in {elapsed:.1f}s")

    print("\nfinal validation metrics (accuracy for categorical/binary, MAE in real units for continuous):")
    final_metrics = evaluate_split(model, embeddings, targets, val_idx, device)
    for spec in PARAM_SCHEMA:
        kind_label = "acc" if spec.kind in ("categorical", "binary") else "MAE"
        print(f"  {spec.name:20s} {kind_label}={final_metrics[spec.name]:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "embedding_dim": embeddings.shape[1],
        "hidden_dim": args.hidden_dim,
        "embedder_kind": args.embedder,
        "fingerprint": fingerprint_sources(args.dataset),
    }, out_dir / "model.pt")
    embedder.save(str(out_dir / f"embedder.{('pkl' if args.embedder == 'tfidf' else 'txt')}"))
    print(f"\nsaved checkpoint to {out_dir}/")


if __name__ == "__main__":
    main()
