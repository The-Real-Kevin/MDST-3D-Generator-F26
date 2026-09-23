"""
MDST 3D Generative Pipeline — Week 1 SOLUTIONS

This is the fully filled-in version of week1_city_generator.ipynb, as one
plain Python script. Same structure, same functions, same variable names --
just with every TODO completed, so it runs top to bottom on its own.

Intended for whoever's running the session (to compare against, or to
unblock someone who's stuck) -- not meant to be handed to students ahead of
the exercise itself.

Run it directly:
    python week1_solutions.py
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SIZE = 128

LAND_EMPTY = 0
LAND_BUILDING = 1
LAND_GREEN = 2


# ---------------------------------------------------------------------------
# Step 1: Terrain (flat)
# ---------------------------------------------------------------------------

def generate_flat_terrain(size, seed=None):
    """
    Returns a (size, size) float32 array of terrain heights, roughly flat
    but with a little gentle random texture.
    """
    rng = np.random.default_rng(seed)

    # small random noise centered near zero -- keeps the terrain close to
    # flat while still having a little natural-looking texture
    terrain_height = rng.normal(size=(size, size)) * 0.05

    return terrain_height.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: Building placement
# ---------------------------------------------------------------------------

def generate_buildings(size, density=0.5, seed=None):
    """
    Returns (building_height, land_use), each a (size, size) array.
    building_height is 0.0 where there's no building.
    land_use uses LAND_EMPTY / LAND_BUILDING from above.

    One downtown center: buildings are more likely the closer a cell is to
    the middle of the grid.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    # the single downtown center, right in the middle of the grid
    cx, cy = size / 2, size / 2

    # distance from every cell to the center, normalized so it's roughly 0-1
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / size

    # intensity: high near the center (dist=0), decaying toward 0 far away.
    # k=5 controls how fast the downtown shrinks as you move outward --
    # bigger k = a smaller, more concentrated downtown.
    intensity = np.exp(-dist * 5)

    # a cell becomes a building when a random roll is less than
    # (intensity * density) -- so higher density or higher intensity both
    # increase the chance of a building at that cell
    is_building = rng.uniform(size=(size, size)) < (intensity * density)

    # building heights: random between 1 and 8 wherever there's a building
    building_height = np.where(is_building, rng.uniform(1, 8, size=(size, size)), 0.0)
    land_use = np.where(is_building, LAND_BUILDING, LAND_EMPTY)

    return building_height.astype(np.float32), land_use.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 3: Green space
# ---------------------------------------------------------------------------

def add_green_space(land_use, green_amount=0.2, seed=None):
    """
    Takes an existing land_use array and returns a NEW one with some empty
    cells turned into green space. Does not touch existing buildings.
    """
    rng = np.random.default_rng(seed)
    size = land_use.shape[0]

    # cells that are currently empty (not a building)
    empty_mask = land_use == LAND_EMPTY

    # of the empty cells, randomly select roughly `green_amount` fraction
    # of them to become green space
    green_mask = empty_mask & (rng.uniform(size=(size, size)) < green_amount)

    land_use = land_use.copy()
    land_use[green_mask] = LAND_GREEN
    return land_use


# ---------------------------------------------------------------------------
# Step 4: Put it all together
# ---------------------------------------------------------------------------

def generate_city(size=SIZE, density=0.6, green_amount=0.25, seed=42):
    """Runs the full Week 1 pipeline and returns (terrain_height, building_height, land_use)."""
    terrain_height = generate_flat_terrain(size, seed=seed)
    building_height, land_use = generate_buildings(size, density=density, seed=seed + 1)
    land_use = add_green_space(land_use, green_amount=green_amount, seed=seed + 2)
    return terrain_height, building_height, land_use


# ---------------------------------------------------------------------------
# Step 5: Visualize
# ---------------------------------------------------------------------------

LAND_USE_COLORS = {
    LAND_EMPTY: (0.85, 0.85, 0.8),     # light tan
    LAND_BUILDING: (0.75, 0.2, 0.2),   # red
    LAND_GREEN: (0.25, 0.6, 0.25),     # green
}


def land_use_rgb(land_use):
    rgb = np.zeros((*land_use.shape, 3), dtype=np.float32)
    for code_val, color in LAND_USE_COLORS.items():
        rgb[land_use == code_val] = color
    return rgb


def plot_city(terrain_height, building_height, land_use, size=SIZE):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(terrain_height, cmap="terrain")
    axes[0].set_title("Terrain height")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(building_height, cmap="magma")
    axes[1].set_title("Building height")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].imshow(land_use_rgb(land_use))
    axes[2].set_title("Land use\n(red=building, green=park, tan=empty)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Week 1 city preview ({size}x{size})")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    terrain_height, building_height, land_use = generate_city()

    print("Generated a city!")
    print(f"  buildings: {(land_use == LAND_BUILDING).mean()*100:.1f}%")
    print(f"  green space: {(land_use == LAND_GREEN).mean()*100:.1f}%")
    print(f"  empty ground: {(land_use == LAND_EMPTY).mean()*100:.1f}%")

    plot_city(terrain_height, building_height, land_use)
