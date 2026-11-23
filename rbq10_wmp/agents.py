"""WMP training configuration (bytedance/WMP ``A1AMPCfgPPO``) as plain dicts for WMPRunner."""
from __future__ import annotations

import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = ROOT / "datasets/rbq10_mocap"

TRAIN_CFG = {
    "seed": 1,
    "policy": dict(
        init_noise_std=1.0,
        encoder_hidden_dims=[256, 128],
        wm_encoder_hidden_dims=[64, 64],
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        latent_dim=32 + 3,
        wm_latent_dim=32,
        activation="elu",
    ),
    "algorithm": dict(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        vel_predict_coef=1.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        amp_replay_buffer_size=1_000_000,
    ),
    "runner": dict(
        experiment_name="rbq10_wmp",
        run_name="",
        num_steps_per_env=24,
        max_iterations=20000,
        save_interval=1000,
        amp_reward_coef=0.5 * 0.02,
        amp_motion_files=sorted(str(p) for p in MOTION_DIR.glob("*.txt")),
        amp_num_preload_transitions=2_000_000,
        amp_task_reward_lerp=0.3,
        amp_discr_hidden_dims=[1024, 512],
        # WMP: the 1024 camera envs (of 4096) feed the world-model dataset.
        wm_dataset_envs=1024,
        wm_image_fp16=True,
        wm_buffer_device="cpu",
    ),
    # Overrides of rbq10_wmp/dreamer/config.py DEFAULTS (empty: paper / original values).
    "world_model": {},
}


def train_cfg(**runner_overrides) -> dict:
    cfg = copy.deepcopy(TRAIN_CFG)
    cfg["runner"].update(runner_overrides)
    return cfg
