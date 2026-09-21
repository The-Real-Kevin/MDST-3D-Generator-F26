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
      colored by land use (tan ground, blue water, green parks, dark
      asphalt roads). Green cells get a small height bump so parks/
      forest read as slightly raised, not flat.
    - a "Buildings" mesh:
        - short buildings (below the ~40th percentile of this city's
          building heights) that are adjacent to another short building
          of similar height are merged into one continuous block with no
          gap or interior wall between them -- like row houses.
        - mid/tall buildings are kept as individual boxes with a small
          gap between them, as before.
      All colored by height (grey short / tan-yellow mid / blue-grey tall).

This is a deliberately direct approach (bmesh, no Geometry Nodes yet) to
get something real into Blender fast. The Geometry Nodes subgroup's job
later in the timeline is to replace this with a proper node-based
generator that reads the same CityGrid arrays as input.

Colors live in the COLORS dict below -- edit those RGB tuples directly to
change the palette, no need to touch the mesh-building logic.
"""
import sys
from pathlib import Path

import bpy
import bmesh
import numpy as np
from scipy.ndimage import label

# When Blender runs this file, sys.argv contains Blender's own args followed
# by a "--" separator and then our args.
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if not argv:
    raise SystemExit("usage: blender --background --python export_to_blender.py -- <city.npz>")

CITY_PATH = argv[0]
CELL_SIZE = 1.0            # world-space size of one grid cell
HEIGHT_SCALE = 8.0         # vertical exaggeration for terrain
BUILDING_HEIGHT_SCALE = 1.0
GREEN_BUMP_HEIGHT = 0.4    # how much parks/forest are raised above bare ground
HEIGHT_GROUP_STEP = 1.0    # short buildings within this height step of each other can merge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citygen import CityGrid, LAND_BUILDING, LAND_GREEN, LAND_ROAD  # noqa: E402

# name -> (R, G, B), 0-1 range
COLORS = {
    "ground": (0.78, 0.72, 0.55),          # light tan
    "green": (0.30, 0.55, 0.25),           # park/forest green
    "road": (0.15, 0.15, 0.16),            # asphalt
    "water": (0.10, 0.35, 0.55),           # blue
    "building_short": (0.55, 0.55, 0.55),  # light grey
    "building_mid": (0.70, 0.62, 0.45),    # light tan/yellow
    "building_tall": (0.35, 0.42, 0.52),   # blue-grey
}


def get_or_create_material(name: str):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b = COLORS[name]
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    mat.diffuse_color = (r, g, b, 1.0)  # also set for viewport solid shading
    return mat


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
            if city.is_water[y, x]:
                z = 0.0
            else:
                z = float(city.terrain_height[y, x]) * HEIGHT_SCALE
                if city.land_use[y, x] == LAND_GREEN:
                    z += GREEN_BUMP_HEIGHT
            verts[y, x] = bm.verts.new((x * CELL_SIZE, y * CELL_SIZE, z))

    slot_names = ["ground", "water", "green", "road"]
    for y in range(size - 1):
        for x in range(size - 1):
            face = bm.faces.new((verts[y, x], verts[y, x + 1], verts[y + 1, x + 1], verts[y + 1, x]))
            if city.is_water[y, x]:
                face.material_index = slot_names.index("water")
            elif city.land_use[y, x] == LAND_GREEN:
                face.material_index = slot_names.index("green")
            elif city.land_use[y, x] == LAND_ROAD:
                face.material_index = slot_names.index("road")
            else:
                face.material_index = slot_names.index("ground")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    for name in slot_names:
        mesh.materials.append(get_or_create_material(name))

    obj = bpy.data.objects.new("Terrain", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _short_building_groups(is_short: np.ndarray, building_height: np.ndarray,
                            height_step: float = HEIGHT_GROUP_STEP) -> np.ndarray:
    """
    Groups adjacent short-building cells with similar height into blocks
    that get rendered as one continuous merged building (row houses).
    Returns an int array, 0 = not part of any group.
    """
    size = is_short.shape[0]
    bucket = np.where(is_short, np.round(building_height / height_step).astype(np.int32), -1)
    group_id = np.zeros((size, size), dtype=np.int32)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4-connectivity
    next_id = 1
    for value in np.unique(bucket):
        if value < 0:
            continue
        mask = bucket == value
        labeled, n = label(mask, structure=structure)
        if n == 0:
            continue
        offset = np.where(labeled > 0, labeled + (next_id - 1), 0)
        group_id[mask] = offset[mask]
        next_id += n
    return group_id


def build_buildings(city: CityGrid):
    size = city.size
    mesh = bpy.data.meshes.new("BuildingsMesh")
    bm = bmesh.new()
    slot_names = ["building_short", "building_mid", "building_tall"]

    is_building = city.land_use == LAND_BUILDING
    heights_present = city.building_height[is_building]
    heights_present = heights_present[heights_present > 0]
    if len(heights_present) > 0:
        low_cut, high_cut = np.percentile(heights_present, [40, 75])
    else:
        low_cut, high_cut = 0.0, 0.0

    is_short = is_building & (city.building_height > 0) & (city.building_height < low_cut)
    is_tallish = is_building & (city.building_height >= low_cut)

    # --- short buildings: merged into continuous blocks, no internal walls ---
    group_id = _short_building_groups(is_short, city.building_height)
    max_group = int(group_id.max())
    for gid in range(1, max_group + 1):
        cells = np.argwhere(group_id == gid)
        if len(cells) == 0:
            continue
        group_height = float(city.building_height[cells[:, 0], cells[:, 1]].mean())

        for y, x in cells:
            base_z = float(city.terrain_height[y, x]) * HEIGHT_SCALE
            top_z = base_z + group_height
            cx, cy = x * CELL_SIZE, y * CELL_SIZE
            half = CELL_SIZE * 0.5  # flush -- no gap within a merged block

            b_sw = bm.verts.new((cx - half, cy - half, base_z))
            b_se = bm.verts.new((cx + half, cy - half, base_z))
            b_ne = bm.verts.new((cx + half, cy + half, base_z))
            b_nw = bm.verts.new((cx - half, cy + half, base_z))
            t_sw = bm.verts.new((cx - half, cy - half, top_z))
            t_se = bm.verts.new((cx + half, cy - half, top_z))
            t_ne = bm.verts.new((cx + half, cy + half, top_z))
            t_nw = bm.verts.new((cx - half, cy + half, top_z))

            new_faces = [
                bm.faces.new((b_sw, b_se, b_ne, b_nw)),
                bm.faces.new((t_nw, t_ne, t_se, t_sw)),
            ]
            # only add a side wall where the neighbor cell isn't part of
            # this same merged group -- this is what removes the gap/wall
            # between adjacent cells of one row-house block
            neighbor_walls = (
                (y - 1, x, (b_sw, b_se, t_se, t_sw)),  # south
                (y + 1, x, (b_ne, b_nw, t_nw, t_ne)),  # north
                (y, x - 1, (b_nw, b_sw, t_sw, t_nw)),  # west
                (y, x + 1, (b_se, b_ne, t_ne, t_se)),  # east
            )
            for ny, nx, wall_verts in neighbor_walls:
                same_group = 0 <= ny < size and 0 <= nx < size and group_id[ny, nx] == gid
                if not same_group:
                    new_faces.append(bm.faces.new(wall_verts))

            for face in new_faces:
                face.material_index = slot_names.index("building_short")

    # --- mid/tall buildings: standalone boxes with a small gap, as before ---
    for y, x in np.argwhere(is_tallish):
        base_z = float(city.terrain_height[y, x]) * HEIGHT_SCALE
        h = float(city.building_height[y, x]) * BUILDING_HEIGHT_SCALE
        if h <= 0:
            continue
        cx, cy = x * CELL_SIZE, y * CELL_SIZE
        half = CELL_SIZE * 0.4  # small gap between adjacent buildings
        mat_idx = slot_names.index("building_mid") if h < high_cut else slot_names.index("building_tall")

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
        new_faces = [bm.faces.new(bottom), bm.faces.new(top[::-1])]
        for i in range(4):
            j = (i + 1) % 4
            new_faces.append(bm.faces.new((bottom[i], bottom[j], top[j], top[i])))
        for face in new_faces:
            face.material_index = mat_idx

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    for name in slot_names:
        mesh.materials.append(get_or_create_material(name))

    obj = bpy.data.objects.new("Buildings", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def main():
    city = CityGrid.load(CITY_PATH)
    print(f"loaded {CITY_PATH}: {city.stats()}")

    clear_scene()
    build_terrain(city)
    build_buildings(city)

    out_path = str(Path(CITY_PATH).with_suffix(".blend").resolve())
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
