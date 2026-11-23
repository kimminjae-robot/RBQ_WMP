"""Run the full WMP loop (RSSM world model + AMP-PPO) on the CPU mock env.

    python3 tools/train_wmp_mock.py --iterations 60 --log-dir logs/wmp_mock

Checks the plumbing of the port without Isaac Sim; it is not a substitute for
training in Isaac Lab (``./run.sh train-rbq10-wmp``).
"""
import argparse
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rbq10_wmp.agents import train_cfg  # noqa: E402
from rbq10_wmp.mock_env import MockWMPEnv  # noqa: E402
from rbq10_wmp.runner import WMPRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/wmp_mock")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp-coef", type=float, default=0.5 * 0.02)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    env = MockWMPEnv(num_envs=args.num_envs, episode_length=200, seed=args.seed)
    cfg = train_cfg(num_steps_per_env=24, save_interval=max(args.iterations // 2, 1), amp_reward_coef=args.amp_coef,
                    amp_num_preload_transitions=50_000, wm_dataset_envs=args.num_envs)
    cfg["algorithm"]["amp_replay_buffer_size"] = 50_000
    # CPU-sized world-model training schedule; the network sizes stay those of the paper.
    cfg["world_model"] = dict(batch_size=8, batch_length=16, train_start_steps=400, train_steps_per_iter=4,
                              compile=False)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for f in args.log_dir.glob("progress.jsonl"):
        f.unlink()
    runner = WMPRunner(env, cfg, log_dir=str(args.log_dir), device="cpu")
    start = time.time()
    runner.learn(args.iterations)
    print(f"done in {time.time() - start:.0f}s -> {args.log_dir}")


if __name__ == "__main__":
    main()
