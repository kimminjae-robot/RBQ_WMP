"""Render the six WMP training terrains (RBQ10 scale) at several curriculum levels.

    python3 tools/render_wmp_terrains.py --output docs/wmp/terrains.png
"""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rbq10_wmp.env.terrain import WMPTerrain, WMPTerrainParams  # noqa: E402

SHOW = [("slope", 0), ("stairs_down", 4), ("gap", 7), ("climb", 12), ("tilt", 17), ("crawl", 18)]
TITLES = {"slope": "Slope", "stairs_down": "Stair", "gap": "Gap", "climb": "Climb", "tilt": "Tilt", "crawl": "Crawl"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "docs/wmp/terrains.png")
    parser.add_argument("--length-scale", type=float, default=1.65)
    parser.add_argument("--lateral-scale", type=float, default=1.45)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 5, 9])
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import LightSource

    p = WMPTerrainParams(seed=1, length_scale=args.length_scale, lateral_scale=args.lateral_scale)
    t = WMPTerrain(p)
    hs, vs, b = p.horizontal_scale, p.vertical_scale, t.border
    n = t.length_per_env_pixels
    verts = t.vertices[t.num_heightfield_vertices:]
    tris = t.triangles[len(t.triangles) - (len(verts) // 8) * 12:] - t.num_heightfield_vertices if len(verts) else None

    fig = plt.figure(figsize=(4.0 * len(SHOW), 3.6 * len(args.levels)), dpi=100)
    ls = LightSource(azdeg=315, altdeg=45)
    for r, level in enumerate(args.levels):
        difficulty = level / p.num_rows
        for c, (name, col) in enumerate(SHOW):
            ax = fig.add_subplot(len(args.levels), len(SHOW), r * len(SHOW) + c + 1, projection="3d")
            x0, y0 = b + level * n, b + col * n
            hf = t.height_field_raw[x0:x0 + n, y0:y0 + n].astype(float) * vs
            hf = np.maximum(hf, -0.8)  # gap pits are 5 m deep; clip for display
            xs = np.arange(n) * hs
            X, Y = np.meshgrid(xs, xs, indexing="ij")
            boxes = []
            if tris is not None and name in ("tilt", "crawl"):
                ox, oy = level * p.terrain_length, col * p.terrain_width
                for k in range(0, len(tris), 12):
                    box = verts[tris[k:k + 12]] - np.array([p.border_size + ox, p.border_size + oy, 0])
                    lo, hi = box.reshape(-1, 3).min(0), box.reshape(-1, 3).max(0)
                    if lo[0] < -0.1 or hi[0] > n * hs + 0.1 or lo[1] < -0.1 or hi[1] > n * hs + 0.1:
                        continue
                    boxes.append((lo, hi))
            colors = None
            for lo, hi in boxes:
                if lo[2] < 0.05:  # tilt walls stand on the ground: draw as raised cells
                    m = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
                    hf = np.where(m, hi[2], hf)
            rgb = ls.shade(hf, cmap=plt.cm.terrain, vert_exag=2, blend_mode="soft", vmin=-0.8, vmax=1.2)
            for lo, hi in boxes:
                m = (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
                rgb[m] = (0.89, 0.38, 0.17, 1.0)
            ax.plot_surface(X, Y, hf, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
            for lo, hi in boxes:
                if lo[2] >= 0.05:  # crawl bar floats: draw its edges and the clearance
                    x = [lo[0], hi[0], hi[0], lo[0], lo[0]]
                    for yy in (lo[1], hi[1]):
                        ax.plot(x, [yy] * 5, [lo[2], lo[2], min(hi[2], 1.6), min(hi[2], 1.6), lo[2]], color="#e3622b", lw=2.2, zorder=10)
                    for xx in (lo[0], hi[0]):
                        for zz in (lo[2], min(hi[2], 1.6)):
                            ax.plot([xx, xx], [lo[1], hi[1]], [zz, zz], color="#e3622b", lw=2.2, zorder=10)
                    ax.text(hi[0] + 0.2, 0.3, lo[2], f"{lo[2]:.2f} m", color="#7a3010", fontsize=9, zorder=11)
            # Robot spawn centre and a 0.8 m x 0.5 m RBQ10 footprint for scale.
            ax.plot([3.6, 4.4, 4.4, 3.6, 3.6], [3.75, 3.75, 4.25, 4.25, 3.75], [hf[40, 40] + 0.5] * 5, color="#1f3fbf", lw=2.5, zorder=12)
            ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_zlim(-0.8, 1.6)
            ax.view_init(elev=32, azim=-120)
            ax.set_box_aspect((1, 1, 0.3))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(f"{TITLES[name]}  level {level} (d={difficulty:.1f})", fontsize=11)
    fig.suptitle(f"WMP terrains for RBQ10 (length x{args.length_scale}, lateral x{args.lateral_scale}); blue = RBQ10 trunk footprint",
                 fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
