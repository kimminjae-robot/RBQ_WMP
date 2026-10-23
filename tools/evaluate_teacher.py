"""Score actual Teacher episodes on a fixed curriculum level and optionally film them."""
import argparse
import hashlib
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--video', type=Path)
parser.add_argument('--level', type=int, default=9, choices=range(10))
parser.add_argument('--num_envs', type=int, default=20)
parser.add_argument('--episodes', type=int, default=3, help='Scored episodes per environment.')
parser.add_argument('--seed', type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 10 or args.num_envs % 10 or args.episodes < 1:
    parser.error('Use a positive episode count and num_envs divisible by 10, to cover all five terrains.')
args.enable_cameras = bool(args.video)
args.output.parent.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app


def main():
    import csv
    import cv2
    import numpy as np
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
    from rbq10_parkour.agents import RBQ10TeacherRunnerCfg
    from rbq10_parkour.contract import FOOT_NAMES, robot_contract
    from rbq10_parkour.evaluation import FixedLevelEvaluationEnv, fixed_level_config
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
    from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

    cfg = fixed_level_config(args.level, args.num_envs, args.seed)
    cfg.sim.device = args.device
    raw = FixedLevelEvaluationEnv(cfg=cfg, render_mode='rgb_array' if args.video else None)
    agent = RBQ10TeacherRunnerCfg()
    env = ParkourRslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
    runner = OnPolicyRunnerWithExtractor(env, agent.to_dict(), log_dir=None, device=raw.device)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=raw.device)
    estimator = runner.get_estimator_inference_policy(device=raw.device)
    robot = raw.scene['robot']
    contact = raw.scene['contact_forces']
    foot_ids, _ = contact.find_bodies(FOOT_NAMES, preserve_order=True)
    parkour = raw.parkour_manager.get_term('base_parkour')
    names = parkour.env_per_terrain_name.reshape(-1)
    counts = np.zeros(args.num_envs, dtype=int)
    peaks = torch.zeros(args.num_envs, 3, device=raw.device)
    records = []

    def sample(ids):
        force = contact.data.net_forces_w[ids][:, foot_ids].norm(dim=-1).max(-1).values
        effort = robot.data.applied_torque[ids].abs().max(-1).values
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[ids])
        tilt = torch.maximum(wrap_to_pi(roll).abs(), wrap_to_pi(pitch).abs())
        peaks[ids] = torch.maximum(peaks[ids], torch.stack((force, effort, tilt), dim=1))

    def finish(ids):
        sample(ids)
        for env_id in ids.cpu().tolist():
            if counts[env_id] >= args.episodes:
                peaks[env_id] = 0
                continue
            assert int(raw.scene.terrain.terrain_levels[env_id]) == args.level
            goal = int(parkour.cur_goal_idx[env_id])
            fallen = bool(peaks[env_id, 2] > 1.5 or robot.data.root_pos_w[env_id, 2] < -0.25)
            outcome = 'fall' if fallen else 'success' if goal >= parkour.num_goals else 'timeout'
            records.append({'env_id': env_id, 'episode': int(counts[env_id]), 'terrain': str(names[env_id]),
                            'level': args.level, 'difficulty': args.level / 9.0,
                            'outcome': outcome, 'goal_index': goal,
                            'episode_steps': int(raw.episode_length_buf[env_id]),
                            'peak_foot_force_n': float(peaks[env_id, 0]),
                            'peak_estimated_torque_nm': float(peaks[env_id, 1])})
            counts[env_id] += 1
            peaks[env_id] = 0

    raw.episode_end_observer = finish
    terrain_order = ['parkour_flat', 'parkour_hurdle', 'parkour_step', 'parkour_gap', 'parkour']
    cameras = {name: int(np.where(names == name)[0][0]) for name in terrain_order}
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*'mp4v'), 25.0, cfg.viewer.resolution)
        assert writer.isOpened()
    obs, extras = env.get_observations()
    tick = 0
    # 10 simulation seconds per terrain; the original auto-reset remains visible.
    video_steps = 2500 if args.video else 0
    maximum_steps = args.episodes * raw.max_episode_length + 50
    try:
        with torch.inference_mode():
            while tick < maximum_steps and (not (counts >= args.episodes).all() or tick < video_steps):
                assert bool((raw.scene.terrain.terrain_levels == args.level).all()), 'Evaluation level drifted'
                sample(torch.arange(raw.num_envs, device=raw.device))
                obs[:, 185:194] = estimator.inference(obs[:, :53])
                actions = policy(obs, hist_encoding=True)
                if not torch.isfinite(actions).all():
                    raise FloatingPointError('Nonfinite Teacher actions')
                obs, reward, dones, extras = env.step(actions)
                if writer is not None and tick < video_steps and tick % 2 == 0:
                    name = terrain_order[min(tick // 500, 4)]
                    raw.viewport_camera_controller.set_view_env_index(cameras[name])
                    frame = raw.render()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    label = f'{args.checkpoint.stem} | Level {args.level}/9 | difficulty {args.level/9:.2f} | {name}'
                    cv2.rectangle(frame, (0, 0), (960, 60), (20, 20, 20), -1)
                    cv2.putText(frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f'Actual simulation, t={tick*raw.step_dt:.2f}s; automatic retries included', (12, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
                    writer.write(frame)
                tick += 1
                if tick % 500 == 0:
                    print('EVALUATION_PROGRESS', tick, 'scored', len(records), flush=True)
    finally:
        if writer is not None:
            writer.release()
    summary = {}
    for name in terrain_order:
        trials = [r for r in records if r['terrain'] == name]
        n = len(trials)
        summary[name] = {'episodes': n,
                         **{outcome: sum(r['outcome'] == outcome for r in trials) for outcome in ('success','fall','timeout')},
                         'success_rate': sum(r['outcome'] == 'success' for r in trials) / n if n else None}
    report = {'checkpoint': args.checkpoint.name, 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'level': args.level, 'difficulty': args.level/9.0, 'seed': args.seed, 'num_envs': args.num_envs,
              'episodes_per_env': args.episodes, 'all_episodes_completed': bool((counts >= args.episodes).all()),
              'steps': tick, 'control_dt': raw.step_dt, 'policy_inference': 'original estimator + history encoder, hist_encoding=True',
              'robot_contract': robot_contract(), 'conditions': {'vx_m_s':0.6,'friction':0.9,'push':False,
              'mass_com_randomization':False,'joint_reset_scale':[0.95,1.05],'roughness':'original enabled'},
              'summary': summary, 'episodes': records,
              'video': str(args.video) if args.video else None,
              'scope': 'Teacher fixed-level simulation evaluation; not vision Student or hardware validation'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w') as stream:
        rows = csv.DictWriter(stream, fieldnames=list(records[0]))
        rows.writeheader(); rows.writerows(records)
    print('EVALUATION_RESULT', json.dumps(summary), flush=True)
    env.close()
    if not report['all_episodes_completed']:
        raise RuntimeError('Evaluation ended without the requested number of episodes')


try:
    main()
finally:
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance()
    app.close()
