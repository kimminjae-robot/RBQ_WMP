"""Record the real five-view Student in Isaac Sim, without starting another trainer."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "upstream/Isaaclab_Parkour", ROOT / "upstream/Isaaclab_Parkour/parkour_tasks"):
    sys.path.insert(0, str(directory))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True, help="MP4 destination; metadata uses the same stem")
parser.add_argument("--terrain", choices=("parkour_hurdle", "parkour_gap", "parkour_step", "parkour", "parkour_demo", "parkour_flat"), default="parkour_hurdle")
parser.add_argument("--level", type=int, choices=range(10), default=4)
parser.add_argument("--seconds", type=float, default=12.)
parser.add_argument("--speed", type=float, default=.6)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.output.parent.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app


def main():
    import cv2
    import numpy as np
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
    from rbq10_parkour.contract import FOOT_NAMES
    from rbq10_parkour.evaluation import FixedLevelEvaluationEnv
    from rbq10_parkour.multiview_config import RBQ10MultiViewStudentEnvCfg
    from rbq10_parkour.multiview_contract import validate_multiview_checkpoint
    from rbq10_parkour.multiview_network import MultiViewDepthCNN
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
    from export_sim2sim import load_original_actor

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = args.checkpoint.resolve(strict=True)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_multiview_checkpoint(saved)
    actor, estimator = load_original_actor(saved)
    del estimator
    actor.load_state_dict(saved["depth_actor_state_dict"], strict=True)
    depth_module = importlib.import_module("_rbq_original_networks.feature_extractors.depth_backbone")
    depth = depth_module.RecurrentDepthBackbone(MultiViewDepthCNN(depth_module.DepthOnlyFCBackbone58x87(32, 512)), {"num_prop": 53})
    depth.load_state_dict(saved["depth_encoder_state_dict"], strict=True)
    updates = saved["rbq10_student_state"]["next_iteration"]
    del saved

    cfg = RBQ10MultiViewStudentEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.training_metrics_enabled = False
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 10, 1
    gen.curriculum = True
    gen.random_difficulty = False
    gen.difficulty_range = (0., 1.)
    for name, terrain in gen.sub_terrains.items():
        terrain.proportion = 1. if name == args.terrain else 0.
    terrain = gen.sub_terrains[args.terrain]
    geometry = {}
    if args.terrain == "parkour_gap":
        pixels = round(eval(terrain.gap_size, {"difficulty":args.level/9.}) / terrain.horizontal_scale)
        geometry["heightfield_nominal_gap_width_m"] = 2 * (pixels // 2) * terrain.horizontal_scale
    if args.terrain == "parkour_step":
        height = eval(terrain.step_height, {"difficulty":args.level/9.})
        geometry["heightfield_nominal_step_height_m"] = round(height / terrain.vertical_scale) * terrain.vertical_scale
    cfg.scene.terrain.max_init_terrain_level = args.level
    cfg.parkours.base_parkour.fixed_terrain_level = args.level
    cfg.parkours.base_parkour.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (args.speed, args.speed)
    cfg.commands.base_velocity.ranges.heading = (0., 0.)
    cfg.commands.base_velocity.resampling_time_range = (60., 60.)
    cfg.events.push_by_setting_velocity = None
    cfg.events.randomize_rigid_body_mass = None
    cfg.events.randomize_rigid_body_com = None
    cfg.events.random_camera_position = None
    cfg.events.physics_material.params["friction_range"] = (.9, .9)
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
    cfg.viewer.resolution = (960, 540)
    cfg.viewer.eye = (1.0, 3.8, 1.5)
    # The viewer follows the trunk origin: aim below it to include every foot.
    cfg.viewer.lookat = (.4, 0., -.25)
    raw = FixedLevelEvaluationEnv(cfg=cfg, render_mode="rgb_array")
    env = ParkourRslRlVecEnvWrapper(raw)
    writer = None
    try:
        actor.to(raw.device).eval()
        depth.to(raw.device).eval()
        depth.hidden_states = torch.zeros(1, 0, 512, device=raw.device)
        raw.common_step_counter = 0
        raw.reset(seed=args.seed)
        obs, extras = env.get_observations()
        assert tuple(extras["observations"]["depth_camera"].shape) == (1, 5, 58, 87)
        robot = raw.scene["robot"]
        sensor = raw.scene["contact_forces"]
        foot_ids, _ = sensor.find_bodies(FOOT_NAMES, preserve_order=True)
        robot_foot_ids, _ = robot.find_bodies(FOOT_NAMES, preserve_order=True)
        parkour = raw.parkour_manager.get_term("base_parkour")
        assert int(raw.scene.terrain.terrain_levels[0]) == args.level
        assert str(parkour.env_per_terrain_name.reshape(-1)[0]) == args.terrain
        raw.viewport_camera_controller.set_view_env_index(0)
        episodes, states = [], []
        frame_count = 0
        def finish(ids):
            for index in ids.cpu().tolist():
                roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[index:index+1])
                fallen = bool(torch.maximum(wrap_to_pi(roll).abs(), wrap_to_pi(pitch).abs()).max() > 1.5 or robot.data.root_pos_w[index, 2] < -.25)
                goals = int(parkour.cur_goal_idx[index])
                episodes.append(dict(time_s=raw.common_step_counter * raw.step_dt,
                    outcome="fall" if fallen else "success" if goals >= parkour.num_goals else "timeout", goals=goals))
        raw.episode_end_observer = finish
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 25., cfg.viewer.resolution)
        if not writer.isOpened():
            raise RuntimeError("Could not open MP4 writer")
        torch.backends.cuda.matmul.allow_tf32 = True
        with torch.inference_mode():
            for tick in range(round(args.seconds / raw.step_dt)):
                if tick == 0 or raw.common_step_counter % 5 == 0:
                    prop = obs[:, :53].clone()
                    prop[:, 6:8] = 0
                    latent = depth(extras["observations"]["depth_camera"], prop)[:, :32]
                action = actor(obs, hist_encoding=True, scandots_latent=latent)
                if not torch.isfinite(action).all():
                    raise FloatingPointError("Nonfinite Student actions")
                obs, _, _, extras = env.step(action)
                forces = sensor.data.net_forces_w[0, foot_ids].norm(dim=-1)
                contact = forces > 2.
                roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                row = dict(time_s=(tick+1)*raw.step_dt, base_position_m=robot.data.root_pos_w[0].tolist(),
                    foot_contacts=contact.tolist(), airborne=not bool(contact.any()), goal_index=int(parkour.cur_goal_idx[0]),
                    foot_height_world_m=robot.data.body_pos_w[0, robot_foot_ids, 2].tolist(),
                    tilt_rad=float(torch.maximum(wrap_to_pi(roll).abs(),wrap_to_pi(pitch).abs()).max()),
                    resets=len(episodes))
                states.append(row)
                if (tick + 1) % 2:
                    continue
                frame = cv2.cvtColor(raw.render(), cv2.COLOR_RGB2BGR)
                cv2.rectangle(frame, (0,0), (960,58), (20,20,20), -1)
                cv2.putText(frame, f"Isaac Sim | RBQ10 5-view Student {checkpoint.stem} ({updates} updates)", (12,23), cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
                status = "AIRBORNE" if row["airborne"] else "GROUND CONTACT"
                cv2.putText(frame, f"{args.terrain} L{args.level}/9 | t={row['time_s']:.2f}s | goal {row['goal_index']}/{parkour.num_goals} | {status} | resets {len(episodes)}", (12,47), cv2.FONT_HERSHEY_SIMPLEX,.49,(225,225,225),1,cv2.LINE_AA)
                writer.write(frame)
                frame_count += 1
                if (tick+1) % 100 == 0:
                    print("CAPTURE_PROGRESS", row, flush=True)
        report = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            student_updates=updates, terrain=args.terrain, level=args.level, difficulty=args.level/9., seed=args.seed,
            terrain_geometry=geometry,
            seconds=args.seconds, speed_command_m_s=args.speed, depth_input_shape=[1,5,58,87],
            inference="Actual trained five-view Student, original CNN/GRU/actor; no Teacher actions",
            conditions="Single fixed-level nominal Isaac terrain; pushes/mass/CoM/camera randomization disabled; original action delay and timing preserved",
            scope="Qualitative uninterrupted preview of this seed, not a success-rate evaluation",
            episodes=episodes, states=states, telemetry_hz=1/raw.step_dt, video=str(args.output), frame_count=frame_count)
        args.output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
        print("CAPTURE_RESULT", {k:v for k,v in report.items() if k!="states"}, flush=True)
    finally:
        if writer is not None:
            writer.release()
        raw.episode_end_observer = None
        env.close()


try:
    main()
except BaseException:
    error = traceback.format_exc()
    print(error, flush=True)
    args.output.with_suffix(".error.txt").write_text(error)
    raise
finally:
    app.close()
