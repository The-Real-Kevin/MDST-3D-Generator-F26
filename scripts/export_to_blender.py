"""
Import a saved CityGrid into Blender and build real 3D geometry from it.

This must be run WITH Blender's own Python interpreter (which has `bpy`
built in) -- not with your regular python3/venv. Run it like this from a
terminal, no need to open the Blender UI first:

    blender --background --python scripts/export_to_blender.py -- data/city_harbor.npz

Or, to open the result in the Blender UI immediately after building it:

    blender --python scripts/export_to_blender.py -- data/city_harbor.npz

What it builds:
    - a "Terrain" mesh: a subdivided plane displaced by terrain_height,
      with water cells left flat/low so you can add an ocean plane later
    - a "Buildings" mesh: one box per building cell, merged into a single
      mesh object (fast even at 128x128 -- thousands of boxes is fine for
      a static mesh, just don't turn each into a separate Blender object)

This is a deliberately simple, direct approach to get something real
into Blender fast. The Geometry Nodes subgroup's job later in the
timeline is to replace the building-box loop below with a proper
node-based generator (varied footprints, rooftop detail, road meshes,
etc.) that reads the same CityGrid arrays as input.
"""
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np

# When Blender runs this file, sys.argv contains Blender's own args followed
# by a "--" separator and then our args.
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not argv:
    raise SystemExit("usage: blender --background --python export_to_blender.py -- <city.npz>")

CITY_PATH = argv[0]
CELL_SIZE = 1.0          # world-space size of one grid cell
HEIGHT_SCALE = 8.0       # vertical exaggeration for terrain
BUILDING_HEIGHT_SCALE = 1.0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citygen import CityGrid, LAND_BUILDING  # noqa: E402


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def build_terrain(city: CityGrid):
    size = city.size
    mesh = bpy.data.meshes.new("TerrainMesh")
    bm = bmesh.new()

    verts = np.empty((size, size), dtype=object)
    for y in range(size):
        for x in range(size):
            z = 0.0 if city.is_water[y, x] else float(city.terrain_height[y, x]) * HEIGHT_SCALE
            verts[y, x] = bm.verts.new((x * CELL_SIZE, y * CELL_SIZE, z))

    for y in range(size - 1):
        for x in range(size - 1):
            bm.faces.new((verts[y, x], verts[y, x + 1], verts[y + 1, x + 1], verts[y + 1, x]))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def build_buildings(city: CityGrid):
    size = city.size
    mesh = bpy.data.meshes.new("BuildingsMesh")
    bm = bmesh.new()

    building_cells = np.argwhere(city.land_use == LAND_BUILDING)
    for y, x in building_cells:
        base_z = float(city.terrain_height[y, x]) * HEIGHT_SCALE
        h = float(city.building_height[y, x]) * BUILDING_HEIGHT_SCALE
        if h <= 0:
            continue
        cx, cy = x * CELL_SIZE, y * CELL_SIZE
        half = CELL_SIZE * 0.4  # small gap between adjacent buildings

        bottom = [
            bm.verts.new((cx - half, cy - half, base_z)),
            bm.verts.new((cx + half, cy - half, base_z)),
            bm.verts.new((cx + half, cy + half, base_z)),
            bm.verts.new((cx - half, cy + half, base_z)),
        ]
        top = [
            bm.verts.new((cx - half, cy - half, base_z + h)),
            bm.verts.new((cx + half, cy - half, base_z + h)),
            bm.verts.new((cx + half, cy + half, base_z + h)),
            bm.verts.new((cx - half, cy + half, base_z + h)),
        ]
        bm.faces.new(bottom)
        bm.faces.new(top[::-1])
        for i in range(4):
            j = (i + 1) % 4
            bm.faces.new((bottom[i], bottom[j], top[j], top[i]))

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new("Buildings", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def main():
    city = CityGrid.load(CITY_PATH)
    print(f"loaded {CITY_PATH}: {city.stats()}")

    clear_scene()
    build_terrain(city)
    build_buildings(city)

    #out_path = str(Path(CITY_PATH).with_suffix(".blend"))
    out_path = str(Path(CITY_PATH).with_suffix(".blend").resolve())
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
