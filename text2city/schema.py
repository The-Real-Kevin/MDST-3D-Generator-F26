"""
Single source of truth for the 10 city-generator parameters the
text-to-parameter model will predict.

Every other piece (dataset generation, model architecture, and eventually
the inference script that calls generate_city.py) imports PARAM_SCHEMA
from here instead of hardcoding parameter names/ranges. If the underlying
city generator's parameters ever change, this is the one file to update.

Each parameter is one of three kinds:
    categorical -- a fixed set of named choices (e.g. terrain archetype)
    continuous  -- a float in [low, high]
    binary      -- True/False

`default` is what the model should learn to predict when a parameter is
NOT mentioned in the input text at all (see generate_dataset.py) --
this is what keeps the model's behavior predictable on short, partial
prompts instead of guessing wildly for anything the user didn't specify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParamSpec:
    name: str
    kind: str  # "categorical" | "continuous" | "binary"
    default: object
    choices: Optional[list] = None      # required for "categorical"
    value_range: Optional[tuple] = None  # required for "continuous", (low, high)
    # how often a synthetic training sentence should mention this parameter
    # at all -- tuned to roughly reflect how often a real person would
    # actually describe this aspect of a city in casual text
    mention_prob: float = 0.35


PARAM_SCHEMA: list[ParamSpec] = [
    ParamSpec(
        name="terrain", kind="categorical", default="hills",
        choices=["flat", "hills", "mountains", "valley", "coastal_harbor", "peninsula", "river"],
        mention_prob=0.55,
    ),
    ParamSpec(
        name="city_archetype", kind="categorical", default="monocentric",
        choices=["monocentric", "polycentric", "linear"],
        mention_prob=0.35,
    ),
    ParamSpec(
        name="num_clusters", kind="continuous", default=1.0,
        value_range=(1.0, 6.0),
        mention_prob=0.25,
    ),
    ParamSpec(
        name="density", kind="continuous", default=0.5,
        value_range=(0.0, 1.0),
        mention_prob=0.55,
    ),
    ParamSpec(
        name="cluster_tightness", kind="continuous", default=0.5,
        value_range=(0.0, 1.0),
        mention_prob=0.3,
    ),
    ParamSpec(
        name="relief", kind="continuous", default=1.0,
        value_range=(0.0, 2.0),
        mention_prob=0.3,
    ),
    ParamSpec(
        name="water_level", kind="continuous", default=0.2,
        value_range=(0.0, 0.6),
        mention_prob=0.3,
    ),
    ParamSpec(
        name="green_amount", kind="continuous", default=0.15,
        value_range=(0.0, 1.0),
        mention_prob=0.45,
    ),
    ParamSpec(
        name="green_override", kind="binary", default=False,
        mention_prob=0.1,
    ),
    ParamSpec(
        name="road_continuity", kind="continuous", default=0.7,
        value_range=(0.0, 1.0),
        mention_prob=0.25,
    ),
]

PARAM_NAMES = [p.name for p in PARAM_SCHEMA]
PARAM_BY_NAME = {p.name: p for p in PARAM_SCHEMA}

CATEGORICAL_PARAMS = [p for p in PARAM_SCHEMA if p.kind == "categorical"]
CONTINUOUS_PARAMS = [p for p in PARAM_SCHEMA if p.kind == "continuous"]
BINARY_PARAMS = [p for p in PARAM_SCHEMA if p.kind == "binary"]


def default_vector() -> dict:
    """The full parameter dict when nothing at all is mentioned."""
    return {p.name: p.default for p in PARAM_SCHEMA}


def to_generator_kwargs(params: dict) -> dict:
    """
    Map this schema's parameter names to generate_city.py's build_city()
    keyword arguments. Kept in one place so a naming mismatch is a single
    fix, not a hunt through the inference script later.

    water_level needs special handling: the model always predicts SOME
    numeric value for it (it's a continuous head, there's no "didn't say
    anything" output), but generate_terrain()'s own per-archetype default
    is genuinely terrain-dependent -- land-locked archetypes (mountains,
    hills, flat, valley) default to no water at all, while water-capable
    ones (coastal_harbor, peninsula, river) default to ~0.2. A single
    fixed schema default for water_level can't represent that, so if the
    predicted value is close to the schema default (meaning the text
    likely said nothing specific about water), we pass water_level=None
    through instead and let generate_terrain() apply the correct
    archetype-specific default -- otherwise a "mountains" prediction with
    an untouched water_level near its default could flood a third of the
    map, since mountains terrain has a lot of area at low elevation.
    """
    water_level = params["water_level"]
    water_spec = PARAM_BY_NAME["water_level"]
    water_level_kwarg = None if abs(water_level - water_spec.default) < 0.03 else water_level

    return {
        "terrain_archetype": params["terrain"],
        "city_archetype": params["city_archetype"],
        "clusters": int(round(params["num_clusters"])),
        "density": params["density"],
        "cluster_tightness": params["cluster_tightness"],
        "relief": params["relief"],
        "water_level": water_level_kwarg,
        "green_amount": params["green_amount"],
        "green_override": bool(params["green_override"]),
        "road_continuity": params["road_continuity"],
    }
