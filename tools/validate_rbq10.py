"""Run real RBQ10 model checks with static PD or the manufacturer's locomotion policy.

This is Phase 2 asset/physics validation, not an RBQ10 parkour policy or training run.
"""
import argparse
import faulthandler
import json
from pathlib import Path
import signal
import traceback
from isaaclab.app import AppLauncher

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=False)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("probe", "stand", "walk", "step"), default="probe")
parser.add_argument("--role", choices=("Teacher", "Student"), default="Student")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video", type=Path)
parser.add_argument("--parkour_timing", action="store_true",
                    help="Validate static standing at the 50 Hz parkour training rate (stand mode only).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.parkour_timing and args.mode != "stand":
    parser.error("--parkour_timing is for stand mode; the official walking policy requires 100 Hz")
if args.video:
    args.enable_cameras = True
args.output.parent.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    import xml.etree.ElementTree as ET
    import parkour_tasks
    import rbq10_parkour
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
    from rbq10_parkour.config import validation_config
    from rbq10_parkour.contract import (
        ASSETS, FOOT_NAMES, JOINT_NAMES, INFO, ACTION_SCALE,
        robot_contract, check_checkpoint_contract,
    )
    from rbq10_parkour.factory import load_factory_policy

    def signature(cfg):
        return json.dumps(cfg.to_dict(), default=str, sort_keys=True)

    go2_task = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"
    go2_before = signature(parse_env_cfg(go2_task, num_envs=4))
    tasks = sorted(t for t in gym.registry if t.startswith("Isaac-Extreme-Parkour-"))
    assert len(tasks) == 12, tasks
    for task in tasks:
        candidate = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        print("CONFIG_OK", task, agent.algorithm.class_name, flush=True)
    assert go2_before == signature(parse_env_cfg(go2_task, num_envs=4)), "Go2 config changed during RBQ10 creation"
    # Reject same-dimensional Go2 weights before applying them to the robot.
    try:
        check_checkpoint_contract(None, robot_contract())
    except ValueError:
        checkpoint_guard = True
    else:
        raise AssertionError("Go2 checkpoint was not rejected")
    task = f"Isaac-Extreme-Parkour-{args.role}-Rainbow-RBQ10-v0"
    if args.mode == "probe":
        cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.terrain_generator.num_cols = 5
    else:
        cfg = validation_config("step" if args.mode == "step" else "flat", args.num_envs, args.seed)
        cfg.sim.device = args.device
        if args.parkour_timing:
            cfg.sim.dt, cfg.decimation = 0.005, 4
            cfg.sim.render_interval = cfg.decimation
            for name in ("depth_camera", "height_scanner", "contact_forces"):
                getattr(cfg.scene, name).update_period = cfg.sim.dt * cfg.decimation
    cfg.viewer.resolution = (960, 540)
    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    sensor = raw.scene["contact_forces"]
    obs, _ = env.reset(seed=args.seed)
    factory_ids, factory_order = robot.find_joints(JOINT_NAMES, preserve_order=True)
    assert factory_order == JOINT_NAMES
    foot_ids, foot_order = sensor.find_bodies(FOOT_NAMES, preserve_order=True)
    robot_foot_ids, _ = robot.find_bodies(FOOT_NAMES, preserve_order=True)
    assert foot_order == FOOT_NAMES
    nominal_urdf = ET.parse(ASSETS / "resources/urdf/rbq10.urdf").getroot()
    nominal_mass = sum(float(x.attrib["value"]) for x in nominal_urdf.findall("link/inertial/mass"))
    expected_limits = {j.attrib["name"]: [float(j.find("limit").attrib[k]) for k in ("lower", "upper")]
                       for j in nominal_urdf.findall("joint") if j.attrib["type"] == "revolute"}
    physical_limits = robot.data.joint_pos_limits[0].cpu().tolist()
    for name, limits in zip(robot.joint_names, physical_limits):
        assert np.allclose(limits, expected_limits[name], atol=1e-5), (name, limits)
    camera_report = None
    if hasattr(cfg.scene, "depth_camera"):
        camera = raw.scene["depth_camera"]
        depth = camera.data.output["distance_to_camera"]
        camera_report = {
            "parent": camera.cfg.prim_path, "offset_position_m": list(camera.cfg.offset.pos),
            "offset_quaternion_wxyz": list(camera.cfg.offset.rot), "convention": camera.cfg.offset.convention,
            "raw_depth_shape": list(depth.shape), "raw_depth_min_m": float(depth.min()),
            "raw_depth_max_m": float(depth.max()), "raw_depth_std_m": float(depth.std()),
            "processed_depth_std": float(obs["depth_camera"].std()),
            "max_distance_m": camera.cfg.max_distance,
        }
        assert camera_report["raw_depth_std_m"] > 1e-4, "Depth camera is constant"
    report = {
        "mode": args.mode, "seed": args.seed, "gui": bool(raw.sim.has_gui()), "num_envs": args.num_envs,
        "registered_tasks": tasks, "go2_configuration_unchanged": True,
        "wrong_robot_checkpoint_rejected": checkpoint_guard,
        "robot_contract": robot_contract(), "camera": camera_report,
        "nominal_urdf_mass_kg": nominal_mass,
        "actual_mass_kg": robot.root_physx_view.get_masses().sum(-1).cpu().tolist(),
        "joint_names": robot.joint_names, "factory_joint_names": factory_order,
        "factory_to_native_indices": factory_ids, "body_names": robot.body_names,
        "joint_limits_rad": physical_limits, "urdf_joint_limits_match": True,
        "foot_names": foot_order, "foot_indices": foot_ids,
        "observation": {k: {"shape": list(v.shape), "finite": bool(torch.isfinite(v).all())} for k, v in obs.items()},
        "physics_dt": raw.physics_dt, "control_dt": raw.step_dt,
        "controller": "none; initialization only" if args.mode == "probe" else
                      "static official home-position PD" if args.mode == "stand" else "official rbq_gym locomotion ONNX weights",
        "rbq10_parkour_policy_trained": False,
    }
    assert report["observation"]["policy"]["shape"] == [args.num_envs, 753]
    assert all(v["finite"] for v in report["observation"].values())
    if args.mode == "probe":
        report["passed"] = True
        args.output.write_text(json.dumps(report, indent=2))
        print("RBQ10_PROBE_OK", json.dumps(report), flush=True)
        env.close()
        return

    assert np.allclose(report["actual_mass_kg"], nominal_mass, atol=1e-3)
    factory, parity_error = load_factory_policy(device=raw.device, verify=True)
    report["factory_onnx_max_abs_error_32_inputs"] = parity_error
    count = args.num_envs
    previous = torch.zeros(count, 12, device=raw.device)
    canonical_home = robot.data.default_joint_pos[:, factory_ids]
    raw_action = torch.zeros(count, 12, device=raw.device)
    start = robot.data.root_pos_w.clone()
    metrics = {k: torch.zeros(count, device=raw.device) for k in (
        "fallen_samples", "reset_count", "touchdowns", "elevated_foot_support_samples", "joint_limit_violation_samples", "effort_limit_violation_samples", "velocity_limit_violation_samples"
    )}
    min_z = torch.full((count,), float("inf"), device=raw.device)
    max_tilt = torch.zeros(count, device=raw.device)
    max_effort = torch.zeros(count, device=raw.device)
    max_velocity = torch.zeros(count, device=raw.device)
    previous_contact = torch.zeros(count, 4, dtype=torch.bool, device=raw.device)
    previous_episode = raw.episode_length_buf.clone()
    telemetry_path = args.output.with_suffix(".jsonl")
    writer = None
    if args.video:
        import cv2
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, cfg.viewer.resolution)
        assert writer.isOpened()
    steps = round(args.seconds / raw.step_dt)
    try:
        with telemetry_path.open("w") as stream, torch.inference_mode():
            for tick in range(steps):
                t = tick * raw.step_dt
                speed = 0.0 if args.mode == "stand" else 0.4 * max(0.0, min(1.0, t - 2.0))
                commands = torch.zeros(count, 3, device=raw.device)
                commands[:, 0] = speed
                if args.mode != "stand":
                    factory_obs = torch.cat((robot.data.root_ang_vel_b * 0.25, robot.data.projected_gravity_b,
                                             commands * commands.new_tensor([2.0, 2.0, 0.25]),
                                             robot.data.joint_pos[:, factory_ids] - canonical_home,
                                             robot.data.joint_vel[:, factory_ids] * 0.05, previous), dim=-1)
                    canonical_raw = factory(factory_obs.clamp(-100.0, 100.0)).clamp(-100.0, 100.0)
                    previous.copy_(canonical_raw)
                    target = canonical_home + INFO["control"]["action_scale"] * canonical_raw
                    limits = robot.data.soft_joint_pos_limits[:, factory_ids]
                    target = target.clamp(limits[..., 0], limits[..., 1])
                    raw_action[:, factory_ids] = (target - canonical_home) / ACTION_SCALE
                obs, reward, terminated, truncated, extras = env.step(raw_action)
                assert torch.isfinite(robot.data.root_state_w).all(), "Nonfinite physical state"
                assert all(torch.isfinite(v).all() for v in obs.values()), "Nonfinite observations"
                contact_force = sensor.data.net_forces_w[:, foot_ids].norm(dim=-1)
                contact = contact_force > 5.0
                foot_z = robot.data.body_pos_w[:, robot_foot_ids, 2]
                roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                tilt = torch.maximum(wrap_to_pi(roll).abs(), wrap_to_pi(pitch).abs())
                z = robot.data.root_pos_w[:, 2]
                if t > 1.0:
                    min_z = torch.minimum(min_z, z)
                    max_tilt = torch.maximum(max_tilt, tilt)
                    metrics["fallen_samples"] += ((z < 0.25) | (tilt > 0.8)).float()
                    metrics["touchdowns"] += (contact & ~previous_contact).sum(-1)
                    metrics["elevated_foot_support_samples"] += (contact & (foot_z > 0.06)).sum(-1)
                metrics["reset_count"] += (raw.episode_length_buf < previous_episode).float()
                limits = robot.data.joint_pos_limits
                metrics["joint_limit_violation_samples"] += ((robot.data.joint_pos < limits[..., 0] - 0.02) |
                                                              (robot.data.joint_pos > limits[..., 1] + 0.02)).any(-1).float()
                metrics["effort_limit_violation_samples"] += (robot.data.applied_torque.abs() > robot.data.joint_effort_limits + 0.1).any(-1).float()
                metrics["velocity_limit_violation_samples"] += (robot.data.joint_vel.abs() > robot.data.joint_vel_limits + 0.1).any(-1).float()
                max_effort = torch.maximum(max_effort, robot.data.applied_torque.abs().max(-1).values)
                max_velocity = torch.maximum(max_velocity, robot.data.joint_vel.abs().max(-1).values)
                previous_contact.copy_(contact)
                previous_episode.copy_(raw.episode_length_buf)
                if tick % 10 == 0:
                    stream.write(json.dumps({"step": tick, "time_s": t, "command_vx": speed,
                                             "root_pos_w": robot.data.root_pos_w.tolist(),
                                             "foot_forces_n": contact_force.tolist(), "foot_z_world_m": foot_z.tolist(),
                                             "goal_index": raw.parkour_manager.get_term("base_parkour").cur_goal_idx.tolist(),
                                             "max_tilt_rad": tilt.tolist()}) + "\n")
                    stream.flush()
                if writer is not None and tick % 4 == 0:
                    frame = raw.render(recompute=True)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    displacement = robot.data.root_pos_w - start
    report.update({"steps": steps, "simulation_seconds": steps * raw.step_dt,
                   "displacement_xyz_m": displacement.tolist(), "min_base_z_m_after_1s": min_z.tolist(),
                   "max_tilt_rad_after_1s": max_tilt.tolist(), "peak_abs_torque_nm": max_effort.tolist(),
                   "peak_abs_joint_velocity_rad_s": max_velocity.tolist(),
                   "final_foot_forces_n": sensor.data.net_forces_w[:, foot_ids].norm(dim=-1).tolist(),
                   "metrics": {k: v.tolist() for k, v in metrics.items()},
                   "telemetry": str(telemetry_path), "video": str(args.video) if args.video else None})
    stable = (metrics["fallen_samples"] == 0) & (metrics["reset_count"] == 0)
    stable &= (metrics["joint_limit_violation_samples"] == 0) & (metrics["effort_limit_violation_samples"] == 0) & (metrics["velocity_limit_violation_samples"] == 0)
    if args.mode == "stand":
        qualified = stable & (displacement[:, :2].norm(dim=-1) < 0.2) & (max_tilt < 0.25)
    else:
        expected_distance = 0.4 * max(0.0, args.seconds - 2.5)
        report["command_integral_distance_m"] = expected_distance
        qualified = stable & (displacement[:, 0] > 0.6 * expected_distance) & (metrics["touchdowns"] > 4)
        if args.mode == "step":
            qualified &= metrics["elevated_foot_support_samples"] > 0
    report["qualified_environments"] = qualified.tolist()
    report["passed"] = bool(qualified.all())
    args.output.write_text(json.dumps(report, indent=2))
    print("RBQ10_VALIDATION_RESULT", json.dumps(report), flush=True)
    env.close()
    if not report["passed"]:
        raise RuntimeError("RBQ10 validation did not meet the stated physical criteria; inspect the saved measurements")


try:
    main()
except BaseException:
    traceback.print_exc()
    raise
finally:
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance()
    app.close()
