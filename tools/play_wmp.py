"""Play a trained RBQ10 WMP policy in Isaac Lab (world model + actor, as on the robot).

    ./run.sh play-rbq10-wmp --checkpoint logs/wmp/<run>/model_20000.pt
"""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--save_depth", type=Path, default=None,
                    help="save real vs world-model-decoded depth of env 0 (npz) for tools/plot_wmp.py")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import numpy as np
    import torch

    from rbq10_wmp.agents import train_cfg
    from rbq10_wmp.env.config import RBQ10WMPEnvCfg_PLAY
    from rbq10_wmp.env.env import RBQ10WMPEnv
    from rbq10_wmp.env_api import WMPEnv
    from rbq10_wmp.inference import WMPPolicy
    from rbq10_wmp.runner import WMPRunner

    device = args.device or "cuda:0"
    env_cfg = RBQ10WMPEnvCfg_PLAY()
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args.num_envs
    env = WMPEnv(RBQ10WMPEnv(cfg=env_cfg))
    cfg = train_cfg(amp_num_preload_transitions=1000, wm_dataset_envs=1)
    runner = WMPRunner(env, cfg, log_dir=None, device=device)
    runner.load(str(args.checkpoint), load_optimizer=False, load_wm_optimizer=False)
    policy = WMPPolicy(runner, env)
    obs, _ = env.reset()
    policy.reset(obs)
    real, decoded = [], []
    for step in range(args.steps):
        actions = policy.act()
        obs, _, _, dones, infos, reset_ids, _ = env.step(actions)
        policy.observe(actions, obs, dones, infos, reset_ids)
        if args.save_depth and infos["depth"] is not None and policy.latent is not None:
            real.append(infos["depth"][0].cpu().numpy())
            decoded.append(policy.decode_depth()[0].cpu().numpy())
        if not app.is_running():
            break
    if args.save_depth:
        np.savez_compressed(args.save_depth, real=np.stack(real), decoded=np.stack(decoded))
finally:
    app.close()
