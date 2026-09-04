"""
Procedural terrain generation.

This is the "rule-based generator" standing in for the future Terrain Model.
It produces a smooth heightmap using multi-octave blurred noise, then shapes
it into a named archetype (flat, hills, mountains, valley, coastal harbor,
peninsula, river). Swapping this module out for a trained VAE later should
require no changes anywhere else in the pipeline -- it just needs to keep
returning (height, is_water) arrays of the same shape.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

TERRAIN_ARCHETYPES = (
    "flat",
    "hills",
    "mountains",
    "valley",
    "coastal_harbor",
    "peninsula",
    "river",
)


def _fractal_noise(size: int, rng: np.random.Generator, base_scale: float,
                    octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """Sum of several blurred-noise layers at decreasing scale/amplitude."""
    height = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for octave in range(octaves):
        sigma = max(base_scale / (2 ** octave), 1.0)
        layer = rng.normal(size=(size, size)).astype(np.float32)
        layer = gaussian_filter(layer, sigma=sigma)
        height += amplitude * layer
        total_amp += amplitude
        amplitude *= persistence
    height /= total_amp
    return _normalize(height)


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    peak = arr.max()
    return arr / peak if peak > 1e-8 else arr


def generate_terrain(
    size: int,
    archetype: str = "hills",
    seed: int | None = None,
    relief: float = 1.0,
    water_level: float | None = None,
):
    """
    Returns (terrain_height: float32[size,size], is_water: bool[size,size]).

    relief: how much the terrain varies. 1.0 = as designed per archetype.
        Lower (e.g. 0.3) flattens toward gentle rolling ground; higher
        (e.g. 1.5-2.0) exaggerates peaks and valleys. Does not affect
        how much water there is -- that's controlled separately by
        water_level, so the two knobs are independent.
    water_level: 0-1 threshold below which ground counts as water,
        measured on the *unscaled* normalized heightmap (before relief
        is applied). Higher = more water. If None, uses each archetype's
        default (0.2 for water-capable archetypes, none for the rest).
    """
    if archetype not in TERRAIN_ARCHETYPES:
        raise ValueError(f"unknown terrain archetype {archetype!r}, choose from {TERRAIN_ARCHETYPES}")

    rng = np.random.default_rng(seed)
    base = _fractal_noise(size, rng, base_scale=size / 6)
    yy, xx = np.mgrid[0:size, 0:size]
    has_water = False

    if archetype == "flat":
        height = base * 0.1

    elif archetype == "hills":
        height = base * 0.4

    elif archetype == "mountains":
        height = base ** 1.6 * 1.0

    elif archetype == "valley":
        height = base * 0.5
        center_dist = np.abs(yy - size / 2) / (size / 2)
        height -= (1 - center_dist) * 0.3

    elif archetype == "coastal_harbor":
        height = base * 0.5
        height -= np.clip(1 - xx / (size * 0.35), 0, 1) * 0.9
        has_water = True

    elif archetype == "peninsula":
        height = base * 0.5
        cx, cy = size * 0.3, size / 2
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / size
        height -= np.clip(dist - 0.15, 0, None) * 1.5
        has_water = True

    elif archetype == "river":
        height = base * 0.4
        wiggle = rng.uniform(0, 2 * np.pi)
        river_amplitude = size / 5      # was size / 8 — smaller = straighter
        river_frequency = 2 * np.pi    # was 3 * np.pi — smaller = fewer/gentler bends
        river_half_width = size * 0.25   # was size * 0.04 — larger = wider river
        river_path = size / 2 + river_amplitude * np.sin(yy / size * river_frequency + wiggle)
        river_dist = np.abs(xx - river_path)
        height -= np.clip(1 - river_dist / river_half_width, 0, 1) * 0.5
        has_water = True

    height = _normalize(height)

    # water is decided on the un-relief-adjusted map, so "how much water"
    # and "how much the terrain varies" stay independently controllable
    default_water_level = 0.2 if has_water else -0.01
    level = default_water_level if water_level is None else water_level
    is_water = height <= level

    # relief: push values away from (relief > 1) or toward (relief < 1)
    # the mid-point, i.e. contrast-adjust the normalized heightmap
    relief = max(relief, 0.0)
    height = np.clip(0.5 + (height - 0.5) * relief, 0.0, 1.0)

    return height.astype(np.float32), is_water
