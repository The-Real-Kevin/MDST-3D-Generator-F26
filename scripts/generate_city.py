#!/usr/bin/env python3
"""
Generate one synthetic city and save it as a .npz CityGrid.

Example:
    python scripts/generate_city.py \\
        --size 128 --terrain coastal_harbor --city polycentric \\
        --clusters 3 --density 0.6 --seed 42 --out data/city_0001.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from citygen import CityGrid, CITY_ARCHETYPES, TERRAIN_ARCHETYPES, generate_buildings, generate_terrain


def build_city(size, terrain_archetype, city_archetype, clusters, density, seed,
                relief=1.0, water_level=None, cluster_tightness=0.5):
    terrain_height, is_water = generate_terrain(
        size, archetype=terrain_archetype, seed=seed,
        relief=relief, water_level=water_level,
    )
    building_height, land_use = generate_buildings(
        terrain_height, is_water,
        archetype=city_archetype, num_clusters=clusters, density=density,
        cluster_tightness=cluster_tightness,
        seed=None if seed is None else seed + 1,
    )
    return CityGrid(
        size=size,
        terrain_height=terrain_height,
        is_water=is_water,
        building_height=building_height,
        land_use=land_use,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--terrain", choices=TERRAIN_ARCHETYPES, default="hills")
    parser.add_argument("--city", choices=CITY_ARCHETYPES, default="monocentric")
    parser.add_argument("--clusters", type=int, default=1)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--cluster-tightness", type=float, default=0.5,
                         help="0-1, how tightly buildings hug cluster centers (higher = more compact)")
    parser.add_argument("--relief", type=float, default=1.0,
                         help="how much terrain height varies (1.0 = default, lower = flatter, higher = more dramatic)")
    parser.add_argument("--water-level", type=float, default=None,
                         help="0-1 threshold for water coverage (higher = more water); default depends on terrain archetype")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/city.npz")
    args = parser.parse_args()

    city = build_city(
        args.size, args.terrain, args.city, args.clusters, args.density, args.seed,
        relief=args.relief, water_level=args.water_level, cluster_tightness=args.cluster_tightness,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    city.save(args.out)

    print(f"saved {args.out}")
    for key, value in city.stats().items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
