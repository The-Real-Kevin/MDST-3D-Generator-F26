"""
Builds a synthetic (sentence -> parameter vector) dataset to train the
text-to-parameter model on.

Why synthetic instead of a real dataset: there's no existing dataset of
"city description text" paired with these exact 10 generator parameters --
so, same trick as the procedural city generator itself, we write the rules
that produce (sentence, parameters) pairs and sample from them thousands
of times. See PARAM_SCHEMA in schema.py for the single source of truth
this pulls from.

Key design choice -- PARTIAL mentions, not full combinatorial coverage:
    A real prompt like "a seaside city with a dense center and lots of
    green space" only describes ~3 of the 10 parameters. It does NOT mean
    "use the default for everything else" is wrong to predict -- it's
    exactly right. So for every synthetic example, we randomly choose a
    SUBSET of parameters to actually mention (per-parameter mention_prob
    from the schema), phrase those, and leave every unmentioned parameter
    at its schema default. This teaches the model the behavior we
    actually want: confidently use what's said, fall back to sensible
    defaults for what isn't -- rather than trying to guess at values for
    things the sentence never touches.

    This is also *why* we don't need combinatorial coverage of all 10
    parameters' value ranges: each output head only has to learn its own
    phrase-to-value mapping largely independently of what the other
    heads see in the same example, so a few thousand randomly sampled
    (not gridded) examples is enough for the phrase/value associations to
    be learned and to recombine freely at inference time.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text2city.schema import PARAM_BY_NAME, PARAM_NAMES, default_vector  # noqa: E402

# ---------------------------------------------------------------------------
# Phrase banks. Each parameter maps to a list of "levels" -- a level is
# (numeric_or_class_value_range, [phrase fragments]). Every phrase fragment
# must read naturally after "a city that is ..." / "... a city that is" so
# they can be freely combined into one sentence regardless of which subset
# of parameters got picked for a given example.
# ---------------------------------------------------------------------------

TERRAIN_PHRASES = {
    "flat": ["on flat land", "built on completely flat terrain", "on level ground"],
    "hills": ["set in rolling hills", "spread across gentle hills", "on hilly terrain"],
    "mountains": ["surrounded by mountains", "nestled in rugged mountains", "high in the mountains"],
    "valley": ["nestled in a deep valley", "tucked into a valley", "set in a river valley"],
    "coastal_harbor": ["built around a harbor", "hugging the coastline", "by the sea", "on the coast", "a seaside setting", "along the seaside"],
    "peninsula": ["built on a peninsula", "surrounded by water on three sides", "on a narrow peninsula"],
    "river": ["split by a winding river", "built along a river", "with a river cutting through it"],
}

CITY_ARCHETYPE_PHRASES = {
    "monocentric": ["with a single downtown core", "built around one central business district", "with one clear city center"],
    "polycentric": ["with several distinct business districts", "made up of multiple downtown clusters", "with scattered urban centers", "with multiple business hubs", "with two separate downtown areas", "made up of several separate city centers"],
    "linear": ["stretched out in a long corridor", "elongated along a single axis", "laid out in a line", "long and narrow", "thin and elongated in shape"],
}

# (low, high) numeric range, [phrases]
NUM_CLUSTERS_LEVELS = [
    ((1, 2), ["with just one or two hubs", "with a couple of centers"]),
    ((3, 4), ["with a handful of clusters", "with several separate hubs"]),
    ((5, 6), ["with many scattered clusters", "with a large number of distinct hubs"]),
]

DENSITY_LEVELS = [
    ((0.0, 0.15), ["barely built up", "almost entirely empty", "sparsely inhabited"]),
    ((0.15, 0.35), ["lightly built up", "fairly sparse", "not very built up"]),
    ((0.35, 0.6), ["moderately built up", "with medium density", "reasonably built up"]),
    ((0.6, 0.8), ["densely populated", "densely built", "quite crowded with buildings"]),
    ((0.8, 1.0), ["extremely dense", "packed with buildings", "ultra-dense and crowded"]),
]

CLUSTER_TIGHTNESS_LEVELS = [
    ((0.0, 0.3), ["with buildings spread loosely across the city", "with a sprawling, spread-out layout"]),
    ((0.3, 0.6), ["with a moderately clustered downtown", "with buildings loosely grouped together"]),
    ((0.6, 1.0), ["with a tightly clustered downtown core", "with a compact, concentrated skyline"]),
]

RELIEF_LEVELS = [
    ((0.0, 0.5), ["with almost no elevation change", "on very flat, gentle ground"]),
    ((0.5, 1.0), ["with gently varying elevation", "with mild rolling terrain"]),
    ((1.2, 2.0), ["with dramatic peaks and valleys", "with extreme, jagged elevation changes"]),
]

WATER_LEVEL_LEVELS = [
    ((0.0, 0.1), ["with barely any water", "with just a trickle of water"]),
    ((0.1, 0.25), ["with a modest amount of water", "with a small river or bay"]),
    ((0.25, 0.45), ["with a fair amount of water", "with a sizeable harbor"]),
    ((0.45, 0.6), ["with a large body of water", "with an expansive bay"]),
]

GREEN_AMOUNT_LEVELS = [
    ((0.0, 0.15), ["with very few parks", "with almost no greenery"]),
    ((0.15, 0.4), ["with some green spaces", "with a handful of parks"]),
    ((0.4, 0.7), ["with lots of green areas", "with many parks scattered around", "full of parks and trees"]),
    ((0.7, 1.0), ["covered almost entirely in greenery", "mostly parks and forest"]),
]

ROAD_CONTINUITY_LEVELS = [
    ((0.0, 0.3), ["with a broken, patchy street network", "with disconnected, fragmented roads"]),
    ((0.3, 0.7), ["with a fairly normal street layout", "with an ordinary road network"]),
    ((0.7, 1.0), ["with a fully connected grid of streets", "with a dense, well-connected road network"]),
]

GREEN_OVERRIDE_PHRASES = [
    "reclaimed almost entirely by nature",
    "overgrown, with greenery taking over the streets",
]

SENTENCE_STARTERS = [
    "A city {items}.",
    "Generate a city {items}.",
    "I'd like a city {items}.",
    "Make me a city {items}.",
    "Design a city {items}.",
]


def _sample_continuous(rng: random.Random, param_name: str, levels: list):
    (low, high), phrases = rng.choice(levels)
    value = rng.uniform(low, high)
    phrase = rng.choice(phrases)
    return value, phrase


def generate_example(rng: random.Random) -> dict:
    params = default_vector()
    mentioned_phrases = []

    for spec in PARAM_BY_NAME.values():
        if rng.random() > spec.mention_prob:
            continue  # not mentioned in this example -> stays at schema default

        if spec.name == "terrain":
            value = rng.choice(spec.choices)
            phrase = rng.choice(TERRAIN_PHRASES[value])
            params["terrain"] = value

        elif spec.name == "city_archetype":
            value = rng.choice(spec.choices)
            phrase = rng.choice(CITY_ARCHETYPE_PHRASES[value])
            params["city_archetype"] = value

        elif spec.name == "num_clusters":
            value, phrase = _sample_continuous(rng, spec.name, NUM_CLUSTERS_LEVELS)
            params["num_clusters"] = value

        elif spec.name == "density":
            value, phrase = _sample_continuous(rng, spec.name, DENSITY_LEVELS)
            params["density"] = value

        elif spec.name == "cluster_tightness":
            value, phrase = _sample_continuous(rng, spec.name, CLUSTER_TIGHTNESS_LEVELS)
            params["cluster_tightness"] = value

        elif spec.name == "relief":
            value, phrase = _sample_continuous(rng, spec.name, RELIEF_LEVELS)
            params["relief"] = value

        elif spec.name == "water_level":
            value, phrase = _sample_continuous(rng, spec.name, WATER_LEVEL_LEVELS)
            params["water_level"] = value

        elif spec.name == "green_amount":
            value, phrase = _sample_continuous(rng, spec.name, GREEN_AMOUNT_LEVELS)
            params["green_amount"] = value

        elif spec.name == "green_override":
            params["green_override"] = True
            phrase = rng.choice(GREEN_OVERRIDE_PHRASES)

        elif spec.name == "road_continuity":
            value, phrase = _sample_continuous(rng, spec.name, ROAD_CONTINUITY_LEVELS)
            params["road_continuity"] = value

        else:
            continue

        mentioned_phrases.append(phrase)

    if not mentioned_phrases:
        # guarantee at least one mentioned parameter so the sentence isn't empty
        phrase = rng.choice(DENSITY_LEVELS[2][1])
        mentioned_phrases.append(phrase)
        params["density"] = rng.uniform(*DENSITY_LEVELS[2][0])

    rng.shuffle(mentioned_phrases)
    if len(mentioned_phrases) == 1:
        items = mentioned_phrases[0]
    else:
        items = ", ".join(mentioned_phrases[:-1]) + ", and " + mentioned_phrases[-1]

    starter = rng.choice(SENTENCE_STARTERS)
    sentence = starter.format(items=items)

    return {"text": sentence, **params}


def generate_dataset(n: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    return [generate_example(rng) for _ in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=6000, help="number of examples to generate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="data/text2city_dataset.csv")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    args = parser.parse_args()

    rows = generate_dataset(args.n, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        fieldnames = ["text"] + PARAM_NAMES
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"wrote {len(rows)} examples to {out_path}")
    print("\nsample rows:")
    for row in rows[:8]:
        print(f"  {row['text']}")
        print(f"    -> {({k: v for k, v in row.items() if k != 'text'})}")


if __name__ == "__main__":
    main()
