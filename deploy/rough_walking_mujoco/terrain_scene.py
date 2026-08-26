"""Generate training-compatible rough height fields and insert them into MuJoCo."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from deploy.astar_mpc_walking.mujoco_scene import build_augmented_scene_xml


TERRAIN_NAMES = (
    "flat",
    "random_rough",
    "up_slope",
    "down_slope",
    "up_stairs",
    "down_stairs",
    "discrete_obstacles",
    "mixed",
)


def difficulty_parameters(level: int) -> Dict[str, float]:
    """Mirror the level-dependent terrain parameters used by the play suite."""
    if not 0 <= level <= 9:
        raise ValueError("terrain level must be in [0, 9]")
    difficulty = level / 10.0
    return {
        "difficulty": difficulty,
        "rough_amplitude_m": 0.01 + 0.045 * difficulty,
        "slope_gradient": 0.04 + 0.20 * difficulty,
        "step_height_m": 0.025 + 0.11 * difficulty,
        "obstacle_height_m": 0.02 + 0.09 * difficulty,
    }


@dataclass(frozen=True)
class TerrainMap:
    name: str
    heights: np.ndarray
    x_bounds: Tuple[float, float]
    y_bounds: Tuple[float, float]
    parameters: Dict[str, float]

    def height_at(self, x: float, y: float) -> float:
        """Bilinearly interpolate the surface height at a world point."""
        ny, nx = self.heights.shape
        fx = np.clip((x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]), 0.0, 1.0)
        fy = np.clip((y - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0]), 0.0, 1.0)
        ix = fx * (nx - 1)
        iy = fy * (ny - 1)
        x0, y0 = int(np.floor(ix)), int(np.floor(iy))
        x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
        wx, wy = ix - x0, iy - y0
        return float(
            (1.0 - wy) * ((1.0 - wx) * self.heights[y0, x0] + wx * self.heights[y0, x1])
            + wy * ((1.0 - wx) * self.heights[y1, x0] + wx * self.heights[y1, x1])
        )


def _smoothed_noise(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    noise = rng.uniform(-1.0, 1.0, size=shape)
    # Isaac Gym first samples a coarser random field and interpolates it.  This
    # compact low-pass filter creates the same spatial character at 0.1 m cells.
    for _ in range(2):
        padded = np.pad(noise, 1, mode="edge")
        noise = (
            4.0 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
        ) / 8.0
    maximum = max(float(np.max(np.abs(noise))), 1.0e-6)
    return noise / maximum


def generate_terrain(name: str, level: int, seed: int = 33, size: float = 8.0) -> TerrainMap:
    if name not in TERRAIN_NAMES:
        raise ValueError(f"Unknown terrain {name!r}; choose from {TERRAIN_NAMES}")
    params = difficulty_parameters(level)
    resolution = 0.10
    count = int(round(size / resolution)) + 1
    axis = np.linspace(-0.5 * size, 0.5 * size, count)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")
    radius = np.maximum(np.abs(x_grid), np.abs(y_grid))
    outside = np.maximum(radius - 1.0, 0.0)
    central_platform = radius <= 1.0
    rng = np.random.default_rng(seed)
    heights = np.zeros_like(x_grid)

    if name == "random_rough":
        heights = params["rough_amplitude_m"] * _smoothed_noise(heights.shape, rng)
    elif name in ("up_slope", "down_slope"):
        sign = 1.0 if name == "up_slope" else -1.0
        heights = sign * params["slope_gradient"] * outside
    elif name in ("up_stairs", "down_stairs"):
        sign = 1.0 if name == "up_stairs" else -1.0
        step_index = np.floor(outside / 0.35)
        heights = sign * params["step_height_m"] * step_index
    elif name == "discrete_obstacles":
        obstacle_height = params["obstacle_height_m"]
        for _ in range(24):
            cx, cy = rng.uniform(-3.8, 3.8, size=2)
            if max(abs(cx), abs(cy)) < 1.15:
                continue
            sx, sy = rng.uniform(0.225, 0.60, size=2)
            mask = (np.abs(x_grid - cx) <= sx) & (np.abs(y_grid - cy) <= sy)
            heights[mask] = rng.choice((-1.0, -0.5, 0.5, 1.0)) * obstacle_height
        heights[central_platform] = 0.0
    elif name == "mixed":
        noise = params["rough_amplitude_m"] * _smoothed_noise(heights.shape, rng)
        rough_mask = (x_grid > 0.8) & (x_grid <= 1.8)
        heights[rough_mask] = noise[rough_mask]
        ramp_mask = (x_grid > 1.8) & (x_grid <= 2.8)
        heights[ramp_mask] = params["slope_gradient"] * (x_grid[ramp_mask] - 1.8)
        stair_mask = x_grid > 2.8
        heights[stair_mask] = (
            params["slope_gradient"]
            + params["step_height_m"] * np.floor((x_grid[stair_mask] - 2.8) / 0.35)
        )

    return TerrainMap(
        name=name,
        heights=heights.astype(np.float64),
        x_bounds=(-0.5 * size, 0.5 * size),
        y_bounds=(-0.5 * size, 0.5 * size),
        parameters=params,
    )


def build_rough_scene_xml(scene_path: str, terrain: TerrainMap) -> Tuple[str, np.ndarray]:
    """Return an inlined robot scene plus a zero-data hfield and its normalized samples."""
    root = ET.fromstring(build_augmented_scene_xml(scene_path, ()))
    worldbody = root.find("worldbody")
    asset = root.find("asset")
    if worldbody is None or asset is None:
        raise ValueError("Scene must contain asset and worldbody sections")

    for geom in tuple(worldbody.findall("geom")):
        if geom.attrib.get("name") == "floor" or geom.attrib.get("type") == "plane":
            worldbody.remove(geom)

    minimum = float(np.min(terrain.heights))
    maximum = float(np.max(terrain.heights))
    elevation_range = max(maximum - minimum, 1.0e-4)
    normalized = np.clip((terrain.heights - minimum) / elevation_range, 0.0, 1.0)
    ny, nx = terrain.heights.shape
    half_x = 0.5 * (terrain.x_bounds[1] - terrain.x_bounds[0])
    half_y = 0.5 * (terrain.y_bounds[1] - terrain.y_bounds[0])
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "rough_terrain_hfield",
            "nrow": str(ny),
            "ncol": str(nx),
            "size": f"{half_x} {half_y} {elevation_range} 0.15",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "rough_terrain",
            "type": "hfield",
            "hfield": "rough_terrain_hfield",
            "pos": f"0 0 {minimum}",
            "rgba": "0.34 0.28 0.20 1",
            "friction": "1.0 0.005 0.0001",
            "group": "3",
            "condim": "3",
            "contype": "1",
            "conaffinity": "1",
        },
    )
    return ET.tostring(root, encoding="unicode"), normalized.ravel()
