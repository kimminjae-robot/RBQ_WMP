"""Retarget the WMP A1 AMP reference clips to RBQ10 and write a visual report.

    python3 tools/retarget_a1_mocap.py            # datasets/a1_mocap -> datasets/rbq10_mocap
    python3 tools/retarget_a1_mocap.py --plot docs/wmp/retarget.png

Pure NumPy; no simulator is needed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rbq10_wmp.retarget import a1_to_rbq10 as rt  # noqa: E402

# RBQ10 URDF limits in RBQ10_LEG_ORDER x (R, P, K).
ROLL_LIMITS = {"HR": (-0.7330, 0.5411), "HL": (-0.5411, 0.7330), "FR": (-0.7330, 0.5411), "FL": (-0.5411, 0.7330)}
KNEE_LIMITS = (-2.8274, -0.4189)


def limit_report(frames):
    q = frames[:, rt.I_JPOS:rt.I_JPOS + 12].reshape(-1, 4, 3)
    worst = {}
    for k, leg in enumerate(rt.RBQ10_LEG_ORDER):
        lo, hi = ROLL_LIMITS[leg]
        worst[f"{leg}R"] = float(max(0.0, lo - q[:, k, 0].min(), q[:, k, 0].max() - hi))
        worst[f"{leg}K"] = float(max(0.0, KNEE_LIMITS[0] - q[:, k, 2].min(), q[:, k, 2].max() - KNEE_LIMITS[1]))
    return worst


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig = plt.figure(figsize=(15, 4.2 * len(names)), dpi=110)
    grid = fig.add_gridspec(len(names), 3, width_ratios=[1.25, 1.25, 1.6])
    colors = {"FL": "#2a6fdb", "FR": "#e3622b", "HL": "#2aa876", "HR": "#a64ac9",
              "RL": "#2aa876", "RR": "#a64ac9"}
    for row, name in enumerate(names):
        src, out, dt_src, dt_out = results[name]
        # (1) side view stick figure at a mid-stance frame, both robots.
        ax = fig.add_subplot(grid[row, 0])
        f = len(src) // 3
        for frames, geo, order, src_map, style, label in (
            (src, rt.A1, rt.A1_LEG_ORDER, None, "--", "A1 (mocap)"),
            (out, rt.RBQ10, rt.RBQ10_LEG_ORDER, None, "-", "RBQ10 (retargeted)"),
        ):
            z0 = frames[f, 2]
            q = frames[f, rt.I_JPOS:rt.I_JPOS + 12].reshape(4, 3)
            hip_xs = []
            for k, leg in enumerate(order):
                sx, sy = rt.LEG_SIGNS[leg]
                if sy < 0:
                    continue  # draw left legs only for a clean side view
                hip = np.array([sx * geo.hip_x, sy * geo.hip_y, 0.0])
                q1, q2, q3 = q[k]
                knee = hip + np.array([-geo.thigh * np.sin(q2), 0, -geo.thigh * np.cos(q2)])
                foot = rt.leg_fk(q[k], geo, sx, sy)
                pts = np.stack([hip, knee, foot]) + np.array([0, 0, z0])
                ax.plot(pts[:, 0], pts[:, 2], style, color=colors[leg], lw=2.4, marker="o", ms=4)
                hip_xs.append(hip[0])
            ax.plot([min(hip_xs), max(hip_xs)], [z0, z0], style, color="#333", lw=4, label=label)
        ax.axhline(0, color="#999", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"{name}: side view, frame {f}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        # (2) front-left foot trajectory in base frame.
        ax = fig.add_subplot(grid[row, 1])
        a = src[:, rt.I_FPOS:rt.I_FPOS + 12].reshape(-1, 4, 3)[:, rt.A1_LEG_ORDER.index("FL")]
        b = out[:, rt.I_FPOS:rt.I_FPOS + 12].reshape(-1, 4, 3)[:, rt.RBQ10_LEG_ORDER.index("FL")]
        ax.plot(a[:, 0] - rt.A1.hip_x, a[:, 2], "--", color="#888", lw=1.4, label="A1 FL foot")
        ax.plot(b[:, 0] - rt.RBQ10.hip_x, b[:, 2], "-", color=colors["FL"], lw=1.6, label="RBQ10 FL foot")
        ax.set_aspect("equal")
        ax.set_title("FL foot path relative to hip (base frame)")
        ax.set_xlabel("x - hip_x [m]"); ax.set_ylabel("z [m]")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

        # (3) joint angles of the FL leg over time.
        ax = fig.add_subplot(grid[row, 2])
        ta = np.arange(len(src)) * dt_src
        tb = np.arange(len(out)) * dt_out
        qa = src[:, rt.I_JPOS:rt.I_JPOS + 12].reshape(-1, 4, 3)[:, rt.A1_LEG_ORDER.index("FL")]
        qb = out[:, rt.I_JPOS:rt.I_JPOS + 12].reshape(-1, 4, 3)[:, rt.RBQ10_LEG_ORDER.index("FL")]
        for j, (jn, c) in enumerate(zip(("roll", "pitch", "knee"), ("#2a6fdb", "#e3622b", "#2aa876"))):
            ax.plot(ta, qa[:, j], "--", color=c, lw=1.0, alpha=0.6)
            ax.plot(tb, qb[:, j], "-", color=c, lw=1.5, label=f"FL {jn}")
        ax.set_xlim(0, 3.0)
        ax.set_title("FL joint angles (dashed A1, solid RBQ10, Froude time scale)")
        ax.set_xlabel("time [s]"); ax.set_ylabel("rad")
        ax.legend(fontsize=8, frameon=False, ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/a1_mocap")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/rbq10_mocap")
    parser.add_argument("--scale", type=float, default=None, help="Length scale; default is leg-length ratio 0.66/0.40")
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()

    ik = rt.check_ik_roundtrip()
    report = {"ik_roundtrip_max_error_m": ik, "clips": {}}
    results = {}
    for src in sorted(args.source.glob("*.txt")):
        dst = args.output / src.name
        info = rt.retarget_file(src, dst, args.scale)
        out = np.asarray(json.loads(dst.read_text())["Frames"])
        info["joint_limit_violation_rad"] = limit_report(out)
        report["clips"][src.stem] = info
        source = json.loads(src.read_text())
        results[src.stem] = (np.asarray(source["Frames"]), out, float(source["FrameDuration"]), info["frame_duration"])
        print(f"{src.name}: {info['num_frames']} frames, dt {info['frame_duration']:.4f}s, "
              f"A1 FK mean err {1000 * info['a1_fk_mean_error_m']:.1f} mm, "
              f"max limit violation {max(info['joint_limit_violation_rad'].values()):.3f} rad")
    (args.output / "retarget_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        plot(results, args.plot)
        print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
