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
                     # green space, and roads, conditioned on terrain
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
    --clusters 3 --density 0.6 --cluster-tightness 0.6 \
    --green-amount 0.25 --road-continuity 0.7 \
    --seed 42 --out data/city_harbor.npz
```

- `--terrain`: flat, hills, mountains, valley, coastal_harbor, peninsula, river
- `--city`: monocentric, polycentric, linear
- `--clusters`: number of building-density hotspots (ignored for monocentric)
- `--density`: 0–1, how built-up the city is overall
- `--cluster-tightness`: 0–1, how tightly buildings hug their cluster centers.
  Low = clusters blend into a broad built-up area with soft edges; high =
  sharp, compact downtown cores. Independent of `--density`.
- `--relief`: how much terrain height varies (1.0 = default, lower = flatter,
  higher = more dramatic). Independent of water coverage.
- `--water-level`: 0–1 threshold for how much water there is (higher = more).
- `--green-amount`: 0–1, how much green space (parks/forest) to add.
- `--green-override`: if set, green space can replace buildings/roads too
  (everything except water) — at `--green-amount 1.0` this makes the whole
  board green except water. If *not* set (default), green only fills
  ground that's otherwise empty — at `--green-amount 1.0` all leftover
  empty ground goes green, but existing buildings/roads are untouched.
- `--road-continuity`: 0–1, how complete the road grid is. Lower values
  leave more gaps/missing segments so it reads as real streets rather
  than a perfectly ordered grid. Useful range is roughly 0.4–0.9 — below
  about 0.3 the road network mostly disappears rather than looking
  "broken but present."
- `--seed`: fix for reproducible output, omit for random each run

Prints summary stats (% water, % building, % green, tallest building,
etc.) so you can sanity-check a city without opening anything.

## 2. Preview it in 2D (fast iteration, no Blender needed)

```bash
python scripts/visualize_2d.py data/city_harbor.npz --out data/city_harbor_preview.png
# or drop --out to open an interactive window
```

Shows terrain height, water mask, building height, and land use side by side.
Use this to tune parameters quickly before touching Blender at all.

## 3. Build it in Blender

This step needs Blender's own bundled Python, not your virtualenv:

```bash
blender --background --python scripts/export_to_blender.py -- data/city_harbor.npz
```

Drop `--background` if you want the Blender UI to open with the result
already loaded. This builds:
- a **Terrain** mesh, colored by land use (tan ground, blue water, green
  parks, dark asphalt roads). Green cells get a small height bump so
  parks/forest read as slightly raised, not flat.
- a **Buildings** mesh:
  - short buildings (roughly the bottom 40% of that city's height range)
    that are adjacent to another short building of similar height are
    merged into one continuous block with no gap or wall between them —
    like a row of connected townhouses. A 1×3 strip or a 2×2 cluster of
    short buildings renders as a single solid block.
  - mid/tall buildings stay as individual boxes with a small gap between
    them, as before.
  - Both are colored by height: grey (short) / tan-yellow (mid) /
    blue-grey (tall), with cutoffs computed per-city from that city's own
    height percentiles.

and saves a `.blend` file next to your `.npz`.

**To actually see the colors** when you open the `.blend` file: Blender's
default "Solid" viewport shading mode shows flat grey unless you tell it
to use material colors. Switch the viewport shading dropdown (top-right
of the 3D viewport) to **Material Preview** or **Rendered**.

To change the palette, edit the `COLORS` dict near the top of
`export_to_blender.py`. To change how buildings are classified as
short/mid/tall or how aggressively short buildings merge, see the
`HEIGHT_GROUP_STEP` constant and the percentile cutoffs in `build_buildings()`.

**Known limitation, by design:** this script builds plain boxes directly
with `bmesh` — no Geometry Nodes yet. That's intentional for a first
end-to-end pass. The Geometry Nodes subgroup's job is to replace this with
a proper node-based system that reads the same `CityGrid` arrays as input
and can do things a simple loop can't easily do (varied footprints, roof
detail, road meshes with actual width/geometry, non-square city shapes).

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
blocking on each other. Everything added in this update (green space,
road gaps, building merging) works entirely within this existing schema —
no new channels were needed.

## Known tuning TODOs (not bugs, just first-pass defaults)

- `water_level`/`relief` interact loosely with archetype-specific shaping
  (e.g. very high relief on `coastal_harbor` can push the coastline
  around a bit) — fine for now, just don't expect them to be perfectly
  orthogonal at extreme values.
- Road placement is still a fixed-spacing grid underneath the gap logic,
  not routed around the city archetype's actual center — worth revisiting
  once Geometry Nodes needs proper connected road meshes.
- No footprint/shape mask yet (square grid only) — circular/elliptical/
  custom-shape grids are a planned addition, applied as a boundary mask
  on top of `CityGrid` rather than a change to its structure.
- Building merging only applies to "short" buildings by design (real
  row-house/townhouse rows are typically low-rise); mid/tall buildings
  are intentionally never merged.
