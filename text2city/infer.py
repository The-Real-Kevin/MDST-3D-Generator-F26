#!/usr/bin/env python3
"""
Step 6: text -> predicted parameters -> real generated city.

Wires the trained model directly into the existing generate_city.py /
export_to_blender.py pipeline -- no changes needed to either of those.

Example:
    python text2city/infer.py "a seaside city with a dense center and lots of green areas" \\
        --checkpoint text2city/checkpoint --out data/from_text.npz --preview

Then, same as always:
    blender --background --python scripts/export_to_blender.py -- data/from_text.npz
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from text2city.schema import to_generator_kwargs  # noqa: E402
from text2city.embedders import load_embedder  # noqa: E402
from text2city.model import TextToCityParams, decode_batch  # noqa: E402
from generate_city import build_city  # noqa: E402


def load_checkpoint(checkpoint_dir: str):
    ckpt_dir = Path(checkpoint_dir)
    ckpt = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    model = TextToCityParams(embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    embedder_ext = "pkl" if ckpt["embedder_kind"] == "tfidf" else "txt"
    embedder = load_embedder(ckpt["embedder_kind"], str(ckpt_dir / f"embedder.{embedder_ext}"))
    return model, embedder


def predict_params(model, embedder, text: str) -> dict:
    embedding = torch.tensor(embedder.encode([text]))
    with torch.no_grad():
        raw = model(embedding)
    return decode_batch(raw, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=str, help="natural-language city description")
    parser.add_argument("--checkpoint", type=str, default="text2city/checkpoint")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/from_text.npz")
    parser.add_argument("--preview", action="store_true", help="also save a 2D preview PNG")
    args = parser.parse_args()

    model, embedder = load_checkpoint(args.checkpoint)
    params = predict_params(model, embedder, args.text)

    print(f'"{args.text}"\n')
    print("predicted parameters:")
    for k, v in params.items():
        print(f"  {k:20s} = {round(v, 3) if isinstance(v, float) else v}")

    kwargs = to_generator_kwargs(params)
    city = build_city(size=args.size, seed=args.seed, **kwargs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    city.save(str(out_path))
    print(f"\nsaved {out_path}")
    print("stats:", city.stats())

    if args.preview:
        preview_path = out_path.with_suffix(".png")
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "visualize_2d.py"),
            str(out_path), "--out", str(preview_path),
        ], check=True)


if __name__ == "__main__":
    main()
