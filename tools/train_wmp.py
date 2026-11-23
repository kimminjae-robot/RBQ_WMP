"""Train WMP (World Model-based Perception, arXiv 2409.16784) on RBQ10 in Isaac Lab.

    ./run.sh train-rbq10-wmp --num_envs 4096 --max_iterations 20000
    ./run.sh train-rbq10-wmp --smoke            # 2 x 6 terrain, 64 envs, quick check
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=20000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run_name", default="wmp")
parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
parser.add_argument("--wm_device", default=None, help="device of the world model (default: same as policy)")
parser.add_argument("--smoke", action="store_true", help="small terrain and env count, not for real training")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app
try:
    import torch

    from rbq10_wmp.agents import ROOT, train_cfg
    from rbq10_wmp.env.config import RBQ10WMPEnvCfg
    from rbq10_wmp.env.env import RBQ10WMPEnv
    from rbq10_wmp.env_api import WMPEnv
    from rbq10_wmp.runner import WMPRunner

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = args.device or "cuda:0"
    env_cfg = RBQ10WMPEnvCfg()
    env_cfg.seed = args.seed
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args.num_envs
    cfg = train_cfg(max_iterations=args.max_iterations, run_name=args.run_name,
                    wm_dataset_envs=min(1024, args.num_envs))
    if args.wm_device:
        cfg["world_model"]["device"] = args.wm_device
    if args.smoke:
        gen = env_cfg.scene.terrain.terrain_generator
        gen.num_rows, gen.num_cols = 2, 20
        env_cfg.scene.num_envs = min(args.num_envs, 64)
        cfg["runner"]["wm_dataset_envs"] = env_cfg.scene.num_envs
        cfg["world_model"].update(train_start_steps=500)
    torch.manual_seed(args.seed)
    log_dir = ROOT / "logs/wmp" / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    log_dir.mkdir(parents=True)
    (log_dir / "launch.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                     "train_cfg": cfg}, indent=2, default=str) + "\n")
    env = WMPEnv(RBQ10WMPEnv(cfg=env_cfg))
    (log_dir / "contract.json").write_text(json.dumps(env.contract(), indent=2) + "\n")
    runner = WMPRunner(env, cfg, log_dir=str(log_dir), device=device)
    if args.resume:
        runner.load(str(args.resume))
    runner.learn(cfg["runner"]["max_iterations"], init_at_random_ep_len=True)
    print(f"[WMP] logs and checkpoints: {log_dir}")
finally:
    app.close()
