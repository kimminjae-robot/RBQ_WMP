"""Figures for RBQ10 WMP runs.

    python3 tools/plot_wmp.py architecture --out docs/wmp/architecture.png
    python3 tools/plot_wmp.py curves --log logs/wmp/<run>/progress.jsonl --out docs/wmp/training.png
    python3 tools/plot_wmp.py depth --mock-checkpoint logs/wmp_mock/model_150.pt --out docs/wmp/wm_depth.png
    python3 tools/plot_wmp.py depth --npz depth.npz --out docs/wmp/wm_depth.png   # from play_wmp.py --save_depth
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlecolor": INK,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": "white"})
    return plt


# --------------------------------------------------------------- architecture
def architecture(out: Path):
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, sub="", color=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=color + "1f", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h / 2 + (0.14 if sub else 0), title, ha="center", va="center", fontsize=10.5,
                weight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8.5, color=INK2)

    def arrow(a, b, text="", color=INK2, style="-|>", ls="-", off=(0, 0.13)):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, lw=1.3, ls=ls))
        if text:
            ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", fontsize=8.5,
                    color=INK2)

    # Region labels
    ax.text(0.2, 5.95, "World model (RSSM) - updated every k = 5 policy steps (0.1 s)", fontsize=11, weight="bold",
            color=BLUE)
    ax.text(7.0, 2.55, "Policy (50 Hz): asymmetric actor-critic + AMP-PPO", fontsize=11, weight="bold", color=ORANGE)

    # World model row
    box(0.2, 4.2, 2.0, 1.2, "Depth 64x64", "RBQ10 front camera", AQUA)
    box(0.2, 3.0, 2.0, 0.9, "Proprioception", "33-d", AQUA)
    box(2.9, 3.6, 1.9, 1.3, "Encoder", "CNN + MLP")
    box(5.5, 3.6, 2.2, 1.3, "Posterior z_t", "32 x 32 categorical")
    box(5.5, 5.05, 2.2, 0.75, "Recurrent h_t", "GRU, 512-d")
    box(8.5, 4.4, 2.0, 1.2, "Decoder", "depth + prop recon.")
    box(8.5, 3.0, 2.0, 1.0, "Prior p(z_t | h_t)", "KL (Eq. 5)")
    box(11.0, 4.4, 1.8, 1.2, "Loss", "recon + beta KL")
    arrow((2.2, 4.8), (2.9, 4.4))
    arrow((2.2, 3.45), (2.9, 3.9))
    arrow((4.8, 4.25), (5.5, 4.25), "e_t")
    arrow((6.6, 5.05), (6.6, 4.9))
    arrow((7.7, 4.4), (8.5, 4.9))
    arrow((7.7, 5.4), (8.5, 3.6), "")
    arrow((10.5, 5.0), (11.0, 5.0))
    arrow((10.5, 3.5), (11.0, 4.6))
    ax.text(2.9, 5.55, "a_{t-k:t} (5 x 12 actions)", fontsize=8.5, color=INK2)
    arrow((2.9, 5.43), (5.5, 5.43))

    # Policy row
    box(0.2, 0.4, 2.0, 1.4, "Prop. history", "5 steps + command", AQUA)
    box(2.9, 1.25, 2.2, 0.95, "sg(h_t)", "stop-gradient, 512-d", BLUE)
    box(2.9, 0.15, 2.2, 0.85, "Privileged + heights", "critic only (53 + 187)", VIOLET)
    box(5.8, 1.15, 2.0, 1.1, "Actor", "MLP 256-128-64", ORANGE)
    box(5.8, 0.05, 2.0, 0.9, "Critic", "MLP 512-256-128", ORANGE)
    box(8.5, 1.15, 2.0, 1.1, "Action a_t", "12 joint targets", ORANGE)
    box(8.5, 0.05, 2.0, 0.9, "AMP discriminator", "style reward (Eq. 10-11)", VIOLET)
    box(11.0, 0.4, 1.8, 1.4, "RBQ10", "Isaac Lab, 6 terrains", AQUA)
    arrow((6.6, 3.6), (4.0, 2.2), "h_t", color=BLUE, ls="--", off=(0.45, 0.05))
    arrow((2.2, 1.12), (5.8, 1.22))
    arrow((5.1, 1.7), (5.8, 1.7))
    arrow((5.1, 1.5), (5.8, 0.6))
    arrow((5.1, 0.55), (5.8, 0.5))
    arrow((7.8, 1.7), (8.5, 1.7))
    arrow((10.5, 1.7), (11.0, 1.4))
    arrow((11.0, 0.7), (10.5, 0.5), "s, s'")
    fig.suptitle("WMP on RBQ10: architecture (paper Fig. 2) with RBQ10 dimensions",
                 fontsize=13, weight="bold", color=INK, x=0.015, ha="left")
    fig.savefig(out, bbox_inches="tight")
    print(out)


# --------------------------------------------------------------------- curves
PANELS = [
    ("Train/mean_reward", "Mean episode return (incl. AMP)", BLUE),
    ("Train/mean_episode_length", "Mean episode length [steps]", BLUE),
    ("Episode/terrain_level", "Terrain level (curriculum)", BLUE),
    ("World_model/image_loss", "World model: depth recon. loss", ORANGE),
    ("World_model/prop_loss", "World model: prop. recon. loss", ORANGE),
    ("World_model/kl", "World model: KL(post || prior)", ORANGE),
    ("Loss/AMP", "AMP discriminator loss", VIOLET),
    ("Loss/vel_predict", "Velocity estimator loss", VIOLET),
    ("Policy/mean_noise_std", "Policy action std", VIOLET),
]


def _smooth(y, k):
    if k <= 1 or len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0.0))
    s = (c[k:] - c[:-k]) / k
    return np.concatenate([y[: k - 1] * np.nan, s])


def curves(log: Path, out: Path, title: str | None, smooth: int):
    plt = _plt()
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    panels = [p for p in PANELS if any(p[0] in r for r in rows)]
    extra = sorted({k for r in rows for k in r if k.startswith("Episode/") and k not in dict((p[0], 0) for p in PANELS)
                    and ("crash" in k or "term_" in k)})
    panels += [(k, k.split("/", 1)[1].replace("_", " "), AQUA) for k in extra[:3]]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.9 * nrow), dpi=150, squeeze=False)
    for ax, (key, label, color) in zip(axes.flat, panels):
        pts = [(r["iteration"], r[key]) for r in rows if key in r]
        x, y = np.array(pts, dtype=float).T
        if smooth > 1 and len(y) > smooth:
            ax.plot(x, y, color=color, lw=1.0, alpha=0.25)
            ax.plot(x, _smooth(y, smooth), color=color, lw=2.0)
        else:
            ax.plot(x, y, color=color, lw=2.0)
        if key.startswith("World_model/") and np.nanmax(y) / max(np.nanmin(y), 1e-9) > 20:
            ax.set_yscale("log")
            label += " (log)"
        ax.set_title(label, loc="left")
        ax.set_xlabel("iteration")
        ax.annotate(f"{y[-1]:.3g}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=9,
                    color=INK, va="center")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    last = rows[-1]
    fig.suptitle(title or f"WMP training - {log.parent.name} ({last['iteration'] + 1} iterations, "
                          f"{last['total_timesteps']:,} steps, {last['total_time_s'] / 60:.1f} min)",
                 x=0.01, ha="left", fontsize=13, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


# ---------------------------------------------------------------------- depth
def _mock_depth(checkpoint: Path, steps: int = 120):
    import torch
    from rbq10_wmp.agents import train_cfg
    from rbq10_wmp.inference import WMPPolicy
    from rbq10_wmp.mock_env import MockWMPEnv
    from rbq10_wmp.runner import WMPRunner

    torch.manual_seed(3)
    env = MockWMPEnv(num_envs=4, episode_length=400, seed=7)
    cfg = train_cfg(amp_num_preload_transitions=1000, wm_dataset_envs=1)
    runner = WMPRunner(env, cfg, log_dir=None, device="cpu")
    runner.load(str(checkpoint), load_optimizer=False, load_wm_optimizer=False)
    policy = WMPPolicy(runner, env)
    obs, _ = env.reset()
    policy.reset(obs)
    real, decoded = [], []
    for _ in range(steps):
        actions = policy.act()
        obs, _, _, dones, infos, ids, _ = env.step(actions)
        policy.observe(actions, obs, dones, infos, ids)
        if infos["depth"] is not None and policy.latent is not None:
            real.append(infos["depth"][0].numpy())
            decoded.append(policy.decode_depth()[0].numpy())
    # decoded at a model step reconstructs the image it has seen (posterior of the previous update)
    return np.stack(real[:-1]), np.stack(decoded[1:])


def depth(out: Path, npz: Path | None, mock_checkpoint: Path | None, frames: int):
    plt = _plt()
    if npz is not None:
        data = np.load(npz)
        real, decoded = data["real"][:-1], data["decoded"][1:]
    else:
        real, decoded = _mock_depth(mock_checkpoint)
    idx = np.linspace(0, len(real) - 1, frames).round().astype(int)
    fig, axes = plt.subplots(3, frames, figsize=(1.9 * frames, 6.2), dpi=150)
    for c, i in enumerate(idx):
        err = np.abs(real[i] - decoded[i])
        for r, (img, cmap, vmin, vmax) in enumerate(((real[i], "gray_r", -0.5, 0.5), (decoded[i], "gray_r", -0.5, 0.5),
                                                     (err, "Oranges", 0, 0.3))):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        axes[0, c].set_title(f"t = {i * 0.1:.1f} s", fontsize=9, weight="normal")
        axes[2, c].set_xlabel(f"MAE {err.mean():.3f}", fontsize=9)
    for r, name in enumerate(("Camera depth", "World model\nreconstruction", "|error|")):
        axes[r, 0].set_ylabel(name, fontsize=10, color=INK)
    mae = np.abs(real - decoded).mean()
    fig.suptitle(f"World model depth reconstruction (darker = closer); mean abs. error {mae:.3f} "
                 f"(normalized range 1.0)", x=0.01, ha="left", fontsize=12, weight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("architecture")
    a.add_argument("--out", type=Path, default=ROOT / "docs/wmp/architecture.png")
    c = sub.add_parser("curves")
    c.add_argument("--log", type=Path, required=True)
    c.add_argument("--out", type=Path, default=ROOT / "docs/wmp/training.png")
    c.add_argument("--title")
    c.add_argument("--smooth", type=int, default=5)
    d = sub.add_parser("depth")
    d.add_argument("--out", type=Path, default=ROOT / "docs/wmp/wm_depth.png")
    d.add_argument("--npz", type=Path)
    d.add_argument("--mock-checkpoint", type=Path)
    d.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "architecture":
        architecture(args.out)
    elif args.cmd == "curves":
        curves(args.log, args.out, args.title, args.smooth)
    else:
        if (args.npz is None) == (args.mock_checkpoint is None):
            parser.error("give exactly one of --npz / --mock-checkpoint")
        depth(args.out, args.npz, args.mock_checkpoint, args.frames)


if __name__ == "__main__":
    main()
