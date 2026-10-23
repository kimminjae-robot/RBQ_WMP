"""Launch the isolated five-view adapter with the original Student learning loop."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=192)
parser.add_argument("--max_iterations", type=int, default=2000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run_name", default="depth5")
parser.add_argument("--smoke", action="store_true", help="Small terrain only for real smoke training, not production")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.max_iterations <= 0:
    parser.error("num_envs and max_iterations must be positive")
if not args.checkpoint.is_file():
    parser.error(f"Missing Student checkpoint: {args.checkpoint}")

app = AppLauncher(args).app
try:
    import torch
    import gymnasium as gym
    import rbq10_parkour
    from isaaclab.utils.io import dump_yaml
    from rbq10_parkour.contract import ROOT
    from rbq10_parkour.multiview_config import RBQ10MultiViewStudentEnvCfg, RBQ10MultiViewStudentRunnerCfg
    from rbq10_parkour.multiview_contract import SENSOR_NAMES
    from rbq10_parkour.multiview_runner import RBQ10MultiViewStudentRunner
    from rbq10_parkour.multiview_audit import audit_cameras
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

    torch.set_num_threads(2)
    # Isaac converts camera quaternions with small matrix multiplications. TF32
    # truncates these mounting transforms; construct/audit them in full float32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    env_cfg = RBQ10MultiViewStudentEnvCfg()
    agent_cfg = RBQ10MultiViewStudentRunnerCfg()
    device = args.device or "cuda:0"
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = device
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = args.run_name
    agent_cfg.logger = "tensorboard"
    if args.smoke:
        gen = env_cfg.scene.terrain.terrain_generator
        gen.num_rows, gen.num_cols = 2, 6
        env_cfg.scene.terrain.max_init_terrain_level = 1
    log_dir = ROOT / "logs/rsl_rl" / agent_cfg.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    log_dir.mkdir(parents=True)
    (log_dir / "launch.json").write_text(json.dumps({**vars(args), "checkpoint":str(args.checkpoint.resolve()), "device":device}, indent=2) + "\n")
    env = gym.make("Isaac-Extreme-Parkour-Student-MultiDepth-Rainbow-RBQ10-v0", cfg=env_cfg)
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = RBQ10MultiViewStudentRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=device)
        runner.load(str(args.checkpoint))
        # Validate actual camera output before any optimizer step.
        obs, extras = env.get_observations()
        images = extras["observations"]["depth_camera"]
        if list(images.shape) != [args.num_envs, 5, 58, 87] or not torch.isfinite(images).all():
            raise ValueError(f"Invalid five-view observations: {images.shape}")
        camera_stats = {}
        for index, name in enumerate(SENSOR_NAMES):
            raw = env.unwrapped.scene[name].data.output["distance_to_camera"]
            camera_stats[name] = dict(min_m=float(raw.min()), max_m=float(raw.max()), mean_m=float(raw.mean()), finite_fraction=float(torch.isfinite(raw).float().mean()), processed_std=float(images[:, index].std()))
        (log_dir / "camera_validation.json").write_text(json.dumps(dict(policy_shape=list(obs.shape), depth_shape=list(images.shape), cameras=camera_stats), indent=2) + "\n")
        (log_dir / "hardware_camera_audit.json").write_text(json.dumps(audit_cameras(env.unwrapped), indent=2) + "\n")
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        print(f"[RBQ10 MultiDepth] log_dir={log_dir} depth_shape={list(images.shape)}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True  # Original network training setting.
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()
except BaseException:
    # Kit shutdown can suppress the Python traceback; preserve it before closing.
    error = traceback.format_exc()
    print(error, flush=True)
    if "log_dir" in globals():
        (log_dir / "training_error.txt").write_text(error)
    raise
finally:
    app.close()
