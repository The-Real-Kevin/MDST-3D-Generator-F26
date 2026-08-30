# citygen-starter

First-steps scaffold for the procedural/generative cyberpunk city project.
This is the rule-based generator standing in for the future trained models
(terrain model, building model) — same data format throughout, so the ML
teams can swap their model in later without changing anything downstream.

## Structure

```
citygen/
    city_grid.py    # CityGrid: the shared data structure (terrain, water,
                     # building height, land use), plus save/load to .npz
    terrain.py       # generate_terrain(): procedural heightmap + water,
                     # by named archetype (hills, mountains, coastal_harbor, ...)
    buildings.py     # generate_buildings(): procedural building placement,
                     # conditioned on terrain, by named city archetype
scripts/
    generate_city.py     # CLI: generate one city, save as .npz
    visualize_2d.py       # quick matplotlib preview of a saved city
    export_to_blender.py  # run INSIDE Blender to build real 3D geometry
data/                     # generated .npz cities land here by default
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 1. Generate a city

```bash
python scripts/generate_city.py \
    --size 128 --terrain coastal_harbor --city polycentric \
    --clusters 3 --density 0.6 --seed 42 --out data/city_harbor.npz
```

- `--terrain`: flat, hills, mountains, valley, coastal_harbor, peninsula, river
- `--city`: monocentric, polycentric, linear
- `--clusters`: number of building-density hotspots (ignored for monocentric)
- `--density`: 0–1, how built-up the city is overall
- `--seed`: fix for reproducible output, omit for random each run

Prints summary stats (% water, % building, tallest building, etc.) so you
can sanity-check a city without opening anything.

## 2. Preview it in 2D (fast iteration, no Blender needed)

```bash
python scripts/visualize_2d.py data/city_harbor.npz --out data/city_harbor_preview.png
# or drop --out to open an interactive window
```

Shows terrain height, water mask, building height, and land use side by side.
Use this to tune archetype parameters quickly before touching Blender at all.

## 3. Build it in Blender

This step needs Blender's own bundled Python, not your virtualenv:

```bash
blender --background --python scripts/export_to_blender.py -- data/city_harbor.npz
```

Drop `--background` if you want the Blender UI to open with the result
already loaded. This builds:
- a **Terrain** mesh (displaced grid, water cells left flat)
- a **Buildings** mesh (one box per building cell, merged into a single mesh)

and saves a `.blend` file next to your `.npz`.

**Known limitation, by design:** this script builds plain boxes directly
with `bmesh` — no Geometry Nodes yet. That's intentional for a first
end-to-end pass. The Geometry Nodes subgroup's job is to replace the
building-box loop with a proper node-based system that reads the same
`CityGrid` arrays as input and can do things a simple loop can't easily do
(varied footprints, roof detail, road meshes, materials by land use).

## Data format reference

Every city is a `CityGrid` (see `citygen/city_grid.py`), four arrays all
shaped `(size, size)`:

| channel            | dtype   | meaning                                  |
|--------------------|---------|-------------------------------------------|
| `terrain_height`   | float32 | ground elevation                          |
| `is_water`         | bool    | True where the cell is water              |
| `building_height`  | float32 | 0.0 where there's no building             |
| `land_use`         | uint8   | 0=empty, 1=building, 2=road, 3=green space |

This is the contract every part of the pipeline (procedural generator now,
trained models later, Blender importer, Streamlit UI) reads and writes.
Keep it stable — it's what lets subgroups work in parallel without
blocking on each other.

## Known tuning TODOs (not bugs, just first-pass defaults)

- Coastal/river water coverage is intentionally modest — tune the
  subtraction strength and `sea_level` in `terrain.py` to taste.
- Road placement is a naive fixed grid, not connected to the city
  archetype's actual center — fine for now, worth revisiting once
  Geometry Nodes needs proper road meshes.
- No footprint/shape mask yet (square grid only) — circular/elliptical/
  custom-shape grids are a planned addition, applied as a boundary mask
  on top of `CityGrid` rather than a change to its structure.
