"""A1 -> RBQ10 dynamic-similarity scaling shared by terrains, commands and rewards.

WMP was tuned for Unitree A1 (thigh = calf = 0.20 m). RBQ10 legs are 0.33 m, so
``S = 1.65``. Under Froude similarity (same ``v^2 / (g L)``) lengths scale by
``S``, times by ``sqrt(S)``, linear velocities by ``sqrt(S)`` and angular
velocities by ``1 / sqrt(S)``. The reference motions (tools/retarget_a1_mocap.py)
use the same factors, so AMP style, command ranges and obstacle sizes stay
mutually consistent. ``S = 1`` gives the original A1 numbers.
"""
import math

LENGTH = 0.33 / 0.20                 # 1.65: RBQ10 / A1 thigh (and calf) length
# Tilt passages scale with the lateral foot stance: 2 x (hip_y + thigh offset).
LATERAL = (0.09 + 0.10285) / (0.047 + 0.08505)   # 1.46
TIME = math.sqrt(LENGTH)             # 1.2845
LIN_VEL = math.sqrt(LENGTH)          # m/s
ANG_VEL = 1.0 / math.sqrt(LENGTH)    # rad/s

# Values of A1AMPCfg (bytedance/WMP legged_gym/envs/a1/a1_amp_config.py) and their RBQ10 counterparts.
A1 = dict(
    lin_vel_x=(0.0, 0.8), ang_vel_yaw=(-1.0, 1.0), tracking_sigma=0.15, lin_vel_clip=0.1,
    feet_air_time_target=0.5, violation_threshold=1.5, base_height=0.3, max_push_vel_xy=1.0,
    height_grid=(1.6, 1.0, 0.1), depth_far_clip=2.0, push_interval_s=15.0, resampling_time_s=10.0,
)


def rbq10_values(s: float = LENGTH) -> dict:
    v, w, t = math.sqrt(s), 1.0 / math.sqrt(s), math.sqrt(s)
    a = A1
    return dict(
        lin_vel_x=(a["lin_vel_x"][0] * v, a["lin_vel_x"][1] * v),
        ang_vel_yaw=(a["ang_vel_yaw"][0] * w, a["ang_vel_yaw"][1] * w),
        tracking_sigma_lin=a["tracking_sigma"] * v * v,   # error^2 of a velocity
        tracking_sigma_ang=a["tracking_sigma"] * w * w,
        lin_vel_clip=a["lin_vel_clip"] * v,
        feet_air_time_target=a["feet_air_time_target"] * t,
        violation_threshold=a["violation_threshold"] * v,
        base_height=0.5,                                   # RBQ10 nominal trunk height (A1: 0.3)
        max_push_vel_xy=a["max_push_vel_xy"] * v,
        # 17 x 11 = 187 points, same count as the original 1.6 m x 1.0 m grid.
        height_grid=(a["height_grid"][0] * s, a["height_grid"][1] * s, a["height_grid"][2] * s),
        depth_far_clip=a["depth_far_clip"] * s,
        push_interval_s=a["push_interval_s"],
        resampling_time_s=a["resampling_time_s"],
    )
