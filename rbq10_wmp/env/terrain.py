# Terrain layout ported from bytedance/WMP ``legged_gym/utils/terrain.py`` (BSD-3-Clause,
# ETH Zurich / NVIDIA, with Bytedance's modifications) and the Isaac Gym
# ``terrain_utils`` height-field primitives it calls.
"""WMP terrains (Slope, Stair, Gap, Climb, Tilt, Crawl) as one height field + boxes.

Pure NumPy/trimesh: the class doubles as an Isaac Lab ``TerrainGenerator``
(``__init__(cfg, device)``, ``terrain_mesh``, ``terrain_origins``,
``flat_patches``), so Isaac Lab imports exactly this mesh, while the height field
and the x-edge mask stay available for the ``feet_edge`` reward, as in WMP.

RBQ10 port: obstacle dimensions are multiplied by ``cfg.length_scale`` (sagittal
lengths and heights) and ``cfg.lateral_scale`` (tilt passage width) because
RBQ10 legs are 1.65x longer than A1 legs.  ``length_scale = lateral_scale = 1``
reproduces the original A1 terrains exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation

TERRAIN_NAMES = ("wave", "slope", "stairs_up", "stairs_down", "discrete", "gap", "climb", "tilt", "crawl", "rough_flat")


@dataclass
class WMPTerrainParams:
    """Values of ``A1AMPCfg.terrain`` (bytedance/WMP ``a1_amp_config.py``)."""

    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 25.0
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int = 10  # difficulty levels
    num_cols: int = 20  # terrain types
    terrain_proportions: tuple = (0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05)
    slope_threshold: float = 0.75
    curriculum: bool = True
    length_scale: float = 1.0
    lateral_scale: float = 1.0
    seed: int | None = None
    edge_half_width: int = 1


class SubTerrain:
    """Minimal clone of ``isaacgym.terrain_utils.SubTerrain``."""

    def __init__(self, width, length, vertical_scale, horizontal_scale):
        self.width, self.length = width, length
        self.vertical_scale, self.horizontal_scale = vertical_scale, horizontal_scale
        self.height_field_raw = np.zeros((width, length), dtype=np.int16)


# ---------------------------------------------------------------- terrain_utils
def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None, rng=np.random):
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)
    heights_range = np.arange(min_height, max_height + step, step)
    hf_down = rng.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale),
                                         int(terrain.length * terrain.horizontal_scale / downsampled_scale)))
    x = np.linspace(0, terrain.width * terrain.horizontal_scale, hf_down.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, hf_down.shape[1])
    f = RegularGridInterpolator((x, y), hf_down.astype(np.float64), method="linear")
    xu = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    yu = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    xx, yy = np.meshgrid(xu, yu, indexing="ij")
    z = np.rint(f(np.stack([xx, yy], -1)))
    terrain.height_field_raw += z.astype(np.int16)
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.0):
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x, center_y = int(terrain.width / 2), int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1, x2 = terrain.width // 2 - platform_size, terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height = 0
    start_x, stop_x, start_y, stop_y = 0, terrain.width, 0, terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
    return terrain


# ------------------------------------------------------------- WMP terrains
def gap_terrain(terrain, gap_size, platform_size=1.0, rng=np.random, width_scale=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    center_x, center_y = terrain.length // 2, terrain.width // 2
    x1 = int(center_x - 1 / terrain.horizontal_scale)
    x2 = int(center_x + 2 / terrain.horizontal_scale)
    x3, x4 = x1 - gap_size, x2 + gap_size
    half_width = width_scale * (1 + 1.0 * rng.random()) / 2
    y1 = int(center_y - half_width / terrain.horizontal_scale)
    y2 = int(center_y + half_width / terrain.horizontal_scale)
    x5 = gap_size
    terrain.height_field_raw[:, :] = -1000
    terrain.height_field_raw[x5:x3, y1:y2] = 0
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    terrain.height_field_raw[x4:, y1:y2] = 0
    return terrain


def climb_terrain(terrain, depth, platform_size=1.0, rng=np.random, length_scale=1.0):
    depth = int(depth / terrain.vertical_scale)
    x1 = int(1 / terrain.horizontal_scale)
    length = length_scale * (1.0 + 0.2 * rng.random())
    x2 = int((1 + length) / terrain.horizontal_scale)
    x3 = int(6 / terrain.horizontal_scale)
    length = length_scale * (1.0 + 0.2 * rng.random())
    x4 = int((6 + length) / terrain.horizontal_scale)
    terrain.height_field_raw[x1:x2, :] = depth
    terrain.height_field_raw[x3:x4, :] = depth
    return terrain


def box_trimesh(size, center):
    """Axis-aligned box as (vertices, triangles) (legged_gym/utils/trimesh.py)."""
    sx, sy, sz = np.asarray(size, dtype=np.float32) / 2
    c = np.asarray(center, dtype=np.float32)
    v = np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                  [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]], dtype=np.float32) + c
    t = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.uint32)
    return v, t


def combine_trimeshes(a, b):
    if a is None:
        return b
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1] + len(a[0])]).astype(np.uint32)


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """Isaac Gym/legged_gym conversion; also returns the x-direction vertical-edge mask."""
    hf = height_field_raw
    num_rows, num_cols = hf.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    move_x = np.zeros((num_rows, num_cols))
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows - 1, :] += (hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols - 1] += (hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows - 1, :num_cols - 1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    # Vectorised version of the original per-row triangle loop (same ordering).
    i = np.arange(num_rows - 1)[:, None]
    ind0 = (np.arange(num_cols - 1)[None, :] + i * num_cols).reshape(-1)
    ind1, ind2 = ind0 + 1, ind0 + num_cols
    ind3 = ind2 + 1
    triangles = np.empty((2 * len(ind0), 3), dtype=np.uint32)
    triangles[0::2] = np.stack([ind0, ind3, ind1], -1)
    triangles[1::2] = np.stack([ind0, ind2, ind3], -1)
    return vertices, triangles, move_x != 0


class WMPTerrain:
    """Port of WMP ``Terrain`` (curriculum mode: rows = difficulty, cols = type)."""

    def __init__(self, params: WMPTerrainParams):
        self.cfg = params
        self.rng = np.random.RandomState(params.seed)
        self.env_length, self.env_width = params.terrain_length, params.terrain_width
        self.proportions = [np.sum(params.terrain_proportions[:i + 1]) for i in range(len(params.terrain_proportions))]
        self.env_origins = np.zeros((params.num_rows, params.num_cols, 3))
        self.width_per_env_pixels = int(self.env_width / params.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / params.horizontal_scale)
        self.border = int(params.border_size / params.horizontal_scale)
        self.tot_cols = int(params.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(params.num_rows * self.length_per_env_pixels) + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.added_trimesh = None
        self.col_types = []
        for j in range(params.num_cols):
            for i in range(params.num_rows):
                difficulty = i / params.num_rows if params.curriculum else self.rng.choice([0.5, 0.75, 0.9])
                choice = j / params.num_cols + 0.001
                terrain, name = self.make_terrain(choice, difficulty, i, j)
                self.add_terrain_to_map(terrain, i, j)
            self.col_types.append(name)
        self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(
            self.height_field_raw, params.horizontal_scale, params.vertical_scale, params.slope_threshold)
        # Height-field vertices only; boxes (tilt/crawl) are appended afterwards.
        self.num_heightfield_vertices = len(self.vertices)
        if self.added_trimesh is not None:
            self.vertices, self.triangles = combine_trimeshes((self.vertices, self.triangles), self.added_trimesh)
        structure = np.ones((params.edge_half_width * 2 + 1, 1))
        self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)

    def type_of_choice(self, choice):
        for k, p in enumerate(self.proportions):
            if choice < p:
                return TERRAIN_NAMES[k]
        return TERRAIN_NAMES[-1]

    def make_terrain(self, choice, difficulty, i=0, j=0):
        p, s, w, rng = self.cfg, self.cfg.length_scale, self.cfg.lateral_scale, self.rng
        terrain = SubTerrain(self.width_per_env_pixels, self.width_per_env_pixels, p.vertical_scale, p.horizontal_scale)
        slope = difficulty * 0.4
        step_height = s * (0.05 + 0.18 * difficulty)
        gap_size = s * 1.0 * difficulty
        pit_depth = s * 0.6 * difficulty
        tilt_width = w * (0.32 - 0.04 * difficulty)
        stair_step_width = s * (0.30 + rng.random() * 0.04)
        prop = self.proportions
        name = self.type_of_choice(choice)
        if name in ("wave", "discrete"):
            raise NotImplementedError("WMP uses zero proportion for wave/discrete terrains")
        if name == "slope":
            if choice < (prop[0] + prop[1]) / 2:
                slope *= -1
            pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            random_uniform_terrain(terrain, -0.05, 0.05, step=0.005, downsampled_scale=0.2, rng=rng)
        elif name in ("stairs_up", "stairs_down"):
            if choice < prop[2]:
                step_height *= -1
            pyramid_stairs_terrain(terrain, step_width=stair_step_width, step_height=step_height, platform_size=3.0)
        elif name == "gap":
            gap_terrain(terrain, gap_size=gap_size, platform_size=4.0, rng=rng, width_scale=w)
            random_uniform_terrain(terrain, -0.05, 0.05, step=0.005, downsampled_scale=0.2, rng=rng)
        elif name == "climb":
            climb_terrain(terrain, depth=pit_depth, platform_size=4.0, rng=rng)
            random_uniform_terrain(terrain, -0.05, 0.05, step=0.005, downsampled_scale=0.2, rng=rng)
        elif name == "tilt":
            ox, oy = (i + 0.5) * self.env_length, (j + 0.5) * self.env_width
            box_z = 1.0
            box_x = s * (0.4 + 0.4 * rng.random())
            side = (self.env_width - tilt_width) / 2
            b = p.border_size
            for x_sign in (1, -1):  # front and back pairs of walls
                cx = ox + b + x_sign * (2 + box_x / 2)
                for y_sign in (-1, 1):
                    cy = oy + b + y_sign * (tilt_width / 2 + side / 2)
                    self.added_trimesh = combine_trimeshes(
                        self.added_trimesh, box_trimesh([box_x, side, box_z], [cx, cy, box_z / 2]))
        elif name == "crawl":
            crawl_height = s * (0.35 - 0.15 * difficulty)
            ox, oy = (i + 0.5) * self.env_length, (j + 0.5) * self.env_width
            box_x = s * (0.2 + 0.2 * rng.random())
            box_z = 1.0
            # The original adds the same bar twice ("front"/"back"); one copy is equivalent.
            self.added_trimesh = combine_trimeshes(self.added_trimesh, box_trimesh(
                [box_x, self.env_width, box_z],
                [ox + p.border_size + 2 + box_x / 2, oy + p.border_size, crawl_height + box_z / 2]))
        else:  # rough_flat
            random_uniform_terrain(terrain, -0.05, 0.05, step=0.005, downsampled_scale=0.2, rng=rng)
        return terrain, name

    def add_terrain_to_map(self, terrain, row, col):
        start_x = self.border + row * self.length_per_env_pixels
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = start_y + self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw
        env_origin_x = (row + 0.5) * self.env_length
        env_origin_y = (col + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[row, col] = [env_origin_x, env_origin_y, env_origin_z]

    def type_env_ranges(self, num_envs):
        """Env index range [start, end) of every terrain type, as WMP's *_start_idx."""
        ranges, total = {}, 0.0
        for name, prop in zip(TERRAIN_NAMES, self.cfg.terrain_proportions):
            start = math.ceil(num_envs * total)
            total += prop
            ranges[name] = (start, math.ceil(num_envs * total))
        return ranges


class WMPTerrainGenerator:
    """Isaac Lab ``TerrainGenerator`` stand-in that serves the WMP mesh.

    Mesh coordinates are shifted by ``-border_size`` exactly like legged_gym's
    ``tm_params.transform``, so world xy = terrain-map xy; height samples are at
    ``(world_xy + border_size) / horizontal_scale``.
    """

    last_instance: "WMPTerrainGenerator | None" = None

    def __init__(self, cfg, device: str = "cpu"):
        import trimesh

        keys = WMPTerrainParams.__dataclass_fields__.keys()
        params = WMPTerrainParams(**{k: getattr(cfg, k) for k in keys if hasattr(cfg, k)})
        self.cfg, self.device = cfg, device
        self.terrain = WMPTerrain(params)
        vertices = self.terrain.vertices.copy()
        vertices[:, :2] -= params.border_size
        self.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=self.terrain.triangles, process=False)
        self.terrain_origins = self.terrain.env_origins.copy()
        self.flat_patches = {}
        WMPTerrainGenerator.last_instance = self
