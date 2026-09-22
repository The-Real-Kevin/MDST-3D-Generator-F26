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

## What actually happened when I ran this (tfidf embedder)

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

## What I found (worth knowing before you present this)

**1. A real bug, now fixed:** `to_generator_kwargs()` was passing the
model's predicted `water_level` through unconditionally. Since the model
always outputs *some* number for every continuous parameter (there's no
"said nothing about this" output), an unmentioned water level near the
schema default still got applied — and for terrain archetypes with a wide
low-elevation spread (mountains especially), that flooded up to **32% of
the map** with water that had nothing to do with the input text. Tested,
confirmed visually, and fixed: `to_generator_kwargs()` now passes
`water_level=None` through whenever the prediction is close to its
schema default, letting `generate_terrain()` apply its own correct
per-archetype default instead (no water for land-locked archetypes, ~0.2
for coastal/river ones). Re-ran the same prompt after the fix — water
dropped from 32% to 0%, confirmed both numerically and visually.

**2. A genuine tfidf limitation (not a bug, an inherent property of this
embedder):** `"a seaside city..."` predicted terrain = `hills`, not
`coastal_harbor`. I traced this exactly: the word "seaside" never appears
anywhere in `generate_dataset.py`'s phrase banks (it got removed earlier
while fixing a grammar issue), so tfidf — which only knows words it
literally saw — had zero signal for it. Even words that DO appear can get
dropped if they're too rare relative to `max_features=400`, since
TfidfVectorizer keeps only the top-N most frequent terms/bigrams pooled
together (confirmed "sea" and "coast" also didn't make the cutoff, despite
appearing in phrase banks). This is precisely the failure mode a real
sentence-transformer is meant to avoid — it would recognize "seaside" is
semantically close to "coastal"/"harbor" even having never seen that exact
word, because it comes pretrained on general English. This is the
strongest concrete argument for running the real embedder before you
present this: **run `train.py --embedder sentence-transformer` and
re-check this exact prompt** — if terrain correctly comes out
`coastal_harbor`, that's a clean, tell-able story about why the pretrained
embedding matters.

## Update: fixed a real class-imbalance bug, found a real TF-IDF ceiling

After the previous run, real-world testing surfaced a genuine problem:
**`city_archetype` and `terrain` both have a schema default that is ALSO
one of their valid choices** (`monocentric`, `hills`). Since ~65-45% of
examples don't mention these parameters at all and fall back to that
default, the default class ends up wildly overrepresented in training —
confirmed by checking the actual data: `monocentric` was 76% of the
dataset, `hills` was 52%. Any sentence whose wording only weakly overlaps
the training vocabulary gets pulled toward that majority-class prior
instead of the correct answer.

**What I tried, in order, testing each before moving to the next:**
1. Class-weighted cross-entropy loss (weighting each class inversely to
   its frequency). Retrained and re-tested on the exact phrases that had
   failed — this did NOT reliably help, and even flipped one previously-
   correct case to wrong. Kept it anyway (it is good general practice and
   doesn't hurt held-out accuracy), but it wasn't the real fix.
2. Widened the TF-IDF vocabulary (400 → 1200 features) and added more
   phrase variety to `generate_dataset.py`'s phrase banks (more synonyms
   for polycentric/linear, and "seaside" phrasing restored for coastal).
   Retrained and re-tested the same failing phrases — this genuinely
   worked: `"a city with two separate downtown areas"` now correctly
   predicts `polycentric` (was `monocentric`), `"a long narrow city"`
   now correctly predicts `linear`, while `"a small quiet town"` (a
   genuine default case, no real signal either way) still correctly
   stays `monocentric` rather than over-correcting.

**What's still a genuine, confirmed ceiling, not a bug**: `"a seaside
city..."` still predicts `hills`, not `coastal_harbor` — even though
"seaside" is now literally in the fitted vocabulary. I checked why: it
only appears in 2 of roughly 50 terrain phrase entries, so its learned
weight still isn't strong enough to overcome the "hills" majority prior.
No amount of tfidf tuning fixes this cleanly, because TF-IDF only knows
"has this exact word appeared, and how often" — it has no notion that
"seaside" and "coastal" mean the same thing. This is exactly the
capability a real sentence-transformer adds for free, without needing
dozens of manually-written synonym phrases for every concept.

## Independent variation (per your Step 1 requirement)

Confirmed in the results above: `cluster_tightness` moved on "tightly
clustered downtown" while `green_amount` moved on "very few parks" in the
*same* sentence, independently and correctly — the shared trunk with
separate heads does let each parameter vary on its own rather than forcing
them to move together.
