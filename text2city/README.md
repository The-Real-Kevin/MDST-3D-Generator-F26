# text2city — text-to-parameter model

Full pipeline: synthetic dataset → train → evaluate → text-to-city inference.
Everything below was actually run, not just written — see "What I found"
for two real issues this surfaced (one fixed, one worth knowing about).

## Files

- `schema.py` — the 10-parameter source of truth + `to_generator_kwargs()`
  wiring into the existing generator.
- `generate_dataset.py` — builds the synthetic (sentence → parameters) CSV.
- `dataset.py` — loads that CSV into training-ready tensors.
- `embedders.py` — two interchangeable embedding backends (see below).
- `model.py` — the trunk + per-parameter-heads architecture.
- `train.py` — Step 4.
- `evaluate.py` — Step 5.
- `infer.py` — Step 6: text in, real generated city out.

## Two embedding backends

- **`sentence-transformer`** (recommended for what you actually ship) — a
  real pretrained language model. Needs `pip install sentence-transformers`
  and a one-time internet download (~80MB, then fully offline/cached).
  Should generalize to phrasing it never saw in training.
- **`tfidf`** (fully offline fallback, no downloads) — classical bag-of-words.
  Only "knows" exact words it saw while fitting on the training set — no
  general language understanding. **This is what all the results below
  use**, since this sandbox's network can't reach the model download host.
  Run everything again with `--embedder sentence-transformer` on your own
  machine to get the real version.

## How to run it, in order

```bash
# 1. generate the dataset (already done, data/text2city_dataset.csv exists)
python text2city/generate_dataset.py --n 6000 --seed 0 --out data/text2city_dataset.csv

# 2. train
python text2city/train.py --embedder sentence-transformer   # recommended, needs internet once
# or, fully offline:
python text2city/train.py --embedder tfidf

# 3. evaluate
python text2city/evaluate.py --checkpoint text2city/checkpoint

# 4. generate a city from text
python text2city/infer.py "a seaside city with a dense center and lots of green space" --preview
blender --background --python scripts/export_to_blender.py -- data/from_text.npz
```

**If `torch` errors on import with a NumPy ABI warning** (`_ARRAY_API not
found`, or `Could not infer dtype of numpy.float32`): an old `torch`
(built before NumPy 2.x support existed) is now sitting next to NumPy 2.x.
Fix: `pip install "numpy<2"`. Don't chase a torch upgrade first —
**on Intel Macs, PyTorch only ships wheels up through 2.2.2** (dropped
support after that), so torch 2.2.2 may simply be the ceiling on that
machine, full stop, and `numpy<2` is the actual fix, not a workaround.

**If you also see `[transformers] Disabling PyTorch because PyTorch >= 2.5
is required but found ...` or a crash with `NameError: name 'nn' is not
defined` inside `transformers/integrations/accelerate.py`**: recent
`transformers`/`sentence-transformers` releases check the installed torch
version and refuse to enable their PyTorch backend below a rising minimum
(2.4, then 2.5, then higher) -- impossible to satisfy on an Intel Mac
capped at torch 2.2.2, and this mismatch has caused real import crashes,
not just warnings. Fix: pin an older, confirmed-stable trio instead of
chasing "latest" --
```bash
pip uninstall -y sentence-transformers transformers huggingface-hub tokenizers safetensors
pip install "sentence-transformers==2.2.2" "transformers==4.30.2" "huggingface-hub==0.16.4"
```
Already pinned this way in `text2city/requirements.txt`.

**Training:** 6,000 examples, 8.4 seconds on CPU — comfortably inside your
1-2 minute target with a lot of room to spare (even the real
sentence-transformer version, which only adds a one-time embedding pass,
should stay well under 5 minutes).

**Held-out synthetic accuracy:** terrain 100%, city_archetype 100%,
green_override 100%, and MAE of 0.02–0.13 on the continuous parameters
(all in their real generator units, e.g. density MAE of 0.043 on a 0-1
scale). This says the model learned the mapping.

**Hand-written sentences the templates never produced** (the real test —
does it generalize, not just memorize):
- `"a huge sprawling metropolis packed with skyscrapers"` → density
  correctly jumped to 0.975 (from a 0.5 default) — "packed" was recognized.
- `"a quiet little town in the mountains with hardly any buildings"` →
  terrain correctly identified as `mountains`, but density came out 0.636
  (should have gone down for "hardly any buildings", not up) — a real miss.
- `"a city built along a big river, with lots of parks on both banks"` →
  terrain correctly `river`, green_amount correctly rose to 0.39.
- `"a flat desert city with almost no water and very few trees"` →
  terrain correctly `flat`, green_amount correctly dropped to 0.035.

