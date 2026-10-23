"""Save the actual five MuJoCo camera poses and ray-depth images at a static probe pose."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim2sim"))
from build_scene import ASSETS, build_scene
from multiview_depth import MultiViewDepthInput, add_cameras
from run_student import validate_export
from run_teacher import ObservationBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = validate_export(args.policy_dir)
    cfg, contract = meta["configuration"], meta["contract"]
    model = mujoco.MjModel.from_xml_string(add_cameras(
        build_scene(contract, "isaac_step", difficulty=.5, terrain_seed=42), cfg["cameras"]))
    data = mujoco.MjData(model)
    home_dict = json.loads((ASSETS / "official/info.json").read_text())["config_info"]["init_state"]["default_joint_angles"]
    home = np.array([home_dict[n] for n in contract["joint_order"]])
    bridge = ObservationBridge(model, data, contract, home, False, .6)
    data.qpos[bridge.qids] = home
    data.qpos[:3] = [2.0, 0., .55]
    mujoco.mj_forward(model, data)
    camera = MultiViewDepthInput(model, data, cfg)
    image = camera.sample()
    rotation = data.xmat[bridge.base].reshape(3, 3)
    positions = (camera.latest_origins - data.xpos[bridge.base]) @ rotation
    forwards = np.array([data.cam_xmat[i].reshape(3, 3) @ [0., 0., -1.] for i in camera.camera_ids]) @ rotation
    colors = plt.get_cmap("tab10").colors
    fig = plt.figure(figsize=(15, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 5, height_ratios=(1.3, 1))
    ax = fig.add_subplot(grid[0, :])
    ax.plot([-.42, .42, .42, -.42, -.42], [-.08, -.08, .1, .1, -.08], color="grey", label="trunk outline (schematic)")
    for i, name in enumerate(cfg["view_order"]):
        x, z = positions[i, [0, 2]]
        dx, dz = forwards[i, [0, 2]] * .18
        ax.scatter(x, z, color=colors[i], s=35)
        ax.arrow(x, z, dx, dz, color=colors[i], width=.003, length_includes_head=True)
        ax.text(x, z+.025+(i%2)*.025, name, ha="center", color=colors[i])
    ax.set(xlim=(-.57, .7), ylim=(-.27, .23), xlabel="body X [m] (forward +)", ylabel="body Z [m]",
           title="Actual MuJoCo camera transforms: side projection; arrows = optical forward")
    ax.set_aspect("equal"); ax.grid(alpha=.25)
    for i, name in enumerate(cfg["view_order"]):
        sub = fig.add_subplot(grid[1, i])
        im = sub.imshow(camera.latest_raw[i], vmin=0., vmax=2., cmap="viridis")
        sub.set_title(name, color=colors[i]); sub.set_xticks([]); sub.set_yticks([])
    fig.colorbar(im, ax=fig.axes[1:], shrink=.75, label="Euclidean range [m] (2 m = clipped/no hit)")
    fig.suptitle("RBQ10 five-view MuJoCo input | static diagnostic pose, not a policy rollout", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    plt.close(fig)
    np.savez_compressed(args.output.with_suffix(".npz"), raw_m=camera.latest_raw, processed=image.numpy(),
                        camera_position_body=positions, optical_forward_body=forwards, qpos=data.qpos)
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        checkpoint_sha256=meta["checkpoint_sha256"], configuration=cfg, course="isaac_step", difficulty=.5, seed=42,
        scope="Static diagnostic pose [2,0,.55] with home joints, terrain-only rays; not a locomotion result",
        camera_position_body_m=positions.tolist(), optical_forward_body=forwards.tolist(),
        png_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2)+"\n")


if __name__ == "__main__":
    main()
