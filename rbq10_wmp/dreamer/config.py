"""World-model hyper-parameters of WMP (bytedance/WMP ``dreamer/configs.yaml``).

The original loads ``configs.yaml`` through argparse; this port keeps the same
values as a dictionary and returns an attribute namespace accepted by
:class:`rbq10_wmp.dreamer.models.WorldModel`.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

DEFAULTS = dict(
    device="cuda:0",
    compile=True,
    precision=32,
    debug=False,
    video_pred_log=False,
    # Environment
    task="rbq10",
    num_actions=12,  # per policy step; multiplied by the model interval k
    # Model
    dyn_hidden=512,
    dyn_deter=512,
    dyn_stoch=32,
    dyn_discrete=32,
    dyn_rec_depth=1,
    dyn_mean_act="none",
    dyn_std_act="sigmoid2",
    dyn_min_std=0.1,
    grad_heads=["decoder", "reward"],
    units=512,
    act="SiLU",
    norm=True,
    encoder=dict(mlp_keys=".*", cnn_keys="image", act="SiLU", norm=True, cnn_depth=32, kernel_size=4,
                 minres=4, mlp_layers=5, mlp_units=1024, symlog_inputs=True),
    decoder=dict(mlp_keys=".*", cnn_keys="image", act="SiLU", norm=True, cnn_depth=32, kernel_size=4,
                 minres=4, mlp_layers=5, mlp_units=1024, cnn_sigmoid=False, image_dist="mse",
                 vector_dist="symlog_mse", outscale=1.0),
    actor=dict(layers=2, dist="normal", entropy=3e-4, unimix_ratio=0.01, std="learned", min_std=0.1,
               max_std=1.0, temp=0.1, lr=3e-5, eps=1e-5, grad_clip=100.0, outscale=1.0),
    critic=dict(layers=2, dist="symlog_disc", slow_target=True, slow_target_update=1,
                slow_target_fraction=0.02, lr=3e-5, eps=1e-5, grad_clip=100.0, outscale=0.0),
    reward_head=dict(layers=2, dist="symlog_disc", loss_scale=0.0, outscale=0.0),
    cont_head=dict(layers=2, loss_scale=1.0, outscale=1.0),
    dyn_scale=0.5,
    rep_scale=0.1,
    kl_free=1.0,
    weight_decay=0.0,
    unimix_ratio=0.01,
    initial="learned",
    # Training
    train_steps_per_iter=10,
    train_start_steps=10000,
    batch_size=16,
    batch_length=64,  # 64 model steps x k=5 x 0.02 s = 6.4 s (paper Sec. V-B)
    train_ratio=512,
    pretrain=100,
    model_lr=1e-4,
    opt_eps=1e-8,
    grad_clip=1000,
    opt="adam",
)


def make_world_model_config(model_interval: int, device: str, **overrides) -> SimpleNamespace:
    """Return the WMP world-model config; actions of ``k`` policy steps are concatenated."""
    values = copy.deepcopy(DEFAULTS)
    for key, value in overrides.items():
        if key not in values:
            raise KeyError(f"Unknown world-model option: {key}")
        values[key] = value
    values["device"] = device
    values["num_actions"] = values["num_actions"] * model_interval
    return SimpleNamespace(**values)
