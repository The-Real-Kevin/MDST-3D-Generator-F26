"""
Procedural building placement, conditioned on terrain.

This stands in for the future Building Model. It's conditioned on the
terrain output (water + slope) so buildings never end up in a lake or on a
cliff -- the same coherence constraint the trained model should eventually
learn on its own. Swap this module for a CVAE later without touching
terrain.py or the CityGrid format.
"""
from __future__ import annotations

import numpy as np

from .city_grid import LAND_BUILDING, LAND_EMPTY, LAND_GREEN, LAND_ROAD

CITY_ARCHETYPES = ("monocentric", "polycentric", "linear")


def _cluster_centers(archetype: str, size: int, num_clusters: int,
                      rng: np.random.Generator):
    if archetype == "monocentric":
        return [(size / 2, size / 2)]

    if archetype == "polycentric":
        return [
            (rng.uniform(size * 0.15, size * 0.85), rng.uniform(size * 0.15, size * 0.85))
            for _ in range(max(num_clusters, 1))
        ]

    if archetype == "linear":
        n = max(num_clusters, 3)
        angle = rng.uniform(0, np.pi)
        centers = []
        for t in np.linspace(0.15, 0.85, n):
            cx = size / 2 + (t - 0.5) * size * np.cos(angle)
            cy = size / 2 + (t - 0.5) * size * np.sin(angle)
            centers.append((cx, cy))
        return centers

    raise ValueError(f"unknown city archetype {archetype!r}, choose from {CITY_ARCHETYPES}")


def generate_buildings(
    terrain_height: np.ndarray,
    is_water: np.ndarray,
    archetype: str = "monocentric",
    num_clusters: int = 1,
    density: float = 0.5,
    seed: int | None = None,
):
    """Returns (building_height: float32[size,size], land_use: uint8[size,size])."""
    size = terrain_height.shape[0]
    rng = np.random.default_rng(seed)
    density = float(np.clip(density, 0.05, 1.0))

    centers = _cluster_centers(archetype, size, num_clusters, rng)

    yy, xx = np.mgrid[0:size, 0:size]
    intensity = np.zeros((size, size), dtype=np.float32)
    for cx, cy in centers:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / size
        falloff = np.exp(-dist * (3.5 - 2.5 * density))
        intensity = np.maximum(intensity, falloff)

    gy, gx = np.gradient(terrain_height)
    slope_mag = np.sqrt(gx ** 2 + gy ** 2)
    buildable = (~is_water) & (slope_mag < 0.08)

    is_building = buildable & (rng.uniform(size=(size, size)) < intensity * density)

    height_noise = rng.normal(loc=1.0, scale=0.4, size=(size, size))
    raw_height = np.clip(intensity * 8.0 * height_noise, 0.3, None)
    building_height = np.where(is_building, raw_height, 0.0).astype(np.float32)

    land_use = np.full((size, size), LAND_EMPTY, dtype=np.uint8)
    land_use[is_building] = LAND_BUILDING

    green_prob = (1 - intensity) * 0.15
    is_green = buildable & (~is_building) & (rng.uniform(size=(size, size)) < green_prob)
    land_use[is_green] = LAND_GREEN

    # simple orthogonal road grid, only on buildable ground that isn't already assigned
    spacing = max(size // 16, 4)
    for axis_slice in (np.s_[::spacing, :], np.s_[:, ::spacing]):
        mask = buildable[axis_slice] & (land_use[axis_slice] == LAND_EMPTY)
        sub = land_use[axis_slice]
        sub[mask] = LAND_ROAD
        land_use[axis_slice] = sub

    return building_height, land_use
