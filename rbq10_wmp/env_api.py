"""The legged_gym-style interface the WMP runner was written against.

``WMPRunner`` (port of WMP ``wmp_runner.py``) talks to the env exactly like the
original talks to ``LeggedRobot``: flat observation tensors, ``step`` returning
``(obs, privileged_obs, rew, done, infos, reset_env_ids, terminal_amp_states)``
and ``infos["depth"]`` every ``k`` steps. :class:`WMPEnv` provides that on top of
the Isaac Lab env; :class:`rbq10_wmp.mock_env.MockWMPEnv` provides it without a
simulator for CPU tests.
"""
from __future__ import annotations

import hashlib
import json

import torch

PRIVILEGED_DIM = 24 + 26 + 3   # gains(24) + contacts(8+12) + com/mass/rest./friction(6) + lin vel(3)
PROP_DIM = 33                  # ang vel, gravity, command, q, qd
HEIGHT_DIM = 187
ACTION_DIM = 12
AMP_DIM = 30


class WMPEnvBase:
    """Dimensions shared by the Isaac Lab adapter and the CPU mock."""

    privileged_dim = PRIVILEGED_DIM
    prop_dim = PROP_DIM
    height_dim = HEIGHT_DIM
    num_actions = ACTION_DIM
    num_amp_obs = AMP_DIM
    num_obs = PRIVILEGED_DIM + PROP_DIM + ACTION_DIM + HEIGHT_DIM  # 285
    num_privileged_obs = num_obs
    use_camera = True

    def contract(self) -> dict:
        raise NotImplementedError


class WMPEnv(WMPEnvBase):
    def __init__(self, env):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        e = self.env
        self.cfg = e.cfg
        self.device = e.device
        self.num_envs = e.num_envs
        self.dt = e.step_dt
        self.model_interval = int(e.cfg.model_interval)
        self.depth_shape = tuple(e.cfg.depth_shape)
        self.max_episode_length = e.max_episode_length
        self.clip_actions = float(e.cfg.clip_actions)
        obs_dim = e.observation_manager.group_obs_dim["policy"][0]
        if obs_dim != self.num_obs:
            raise ValueError(f"WMP observation has {obs_dim} dims, expected {self.num_obs}")
        if e.action_manager.total_action_dim != ACTION_DIM:
            raise ValueError("RBQ10 WMP expects 12 actions")
        self._obs = None
        self.min_action_std = self._min_action_std()

    # legged_gym attribute names used by the runner
    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf[:] = value

    @property
    def global_counter(self):
        return self.env.wmp_global_counter

    def _min_action_std(self):
        """WMP: ``min_normalized_std * |dof_pos_limits[:, 1] - dof_pos_limits[:, 0]|``."""
        robot = self.env.scene["robot"]
        limits = robot.data.joint_pos_limits[0]
        width = torch.clamp(torch.abs(limits[:, 1] - limits[:, 0]), max=self.cfg.min_std_max_joint_range)
        per_kind = dict(zip("RPK", self.cfg.min_normalized_std))
        normalized = torch.tensor([per_kind[name[-1]] for name in robot.joint_names], device=self.device)
        return (normalized * width).tolist()

    # ------------------------------------------------------------------ API
    def reset(self):
        obs, _ = self.env.reset()
        self._obs = obs
        return obs["policy"], obs["policy"]

    def get_observations(self):
        return self._obs["policy"]

    def get_privileged_observations(self):
        return self._obs["policy"]

    def get_amp_observations(self):
        return self.env.amp_observation()

    def update_reward_curriculum(self, iteration):
        self.env.update_reward_curriculum(iteration)

    def step(self, actions):
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        self._obs = obs
        dones = terminated | truncated
        reset_env_ids = dones.nonzero(as_tuple=False).flatten()
        infos = {"time_outs": truncated}
        if self.env.wmp_global_counter % self.model_interval == 0:
            infos["depth"] = obs["depth"].reshape(self.num_envs, *self.depth_shape)
        else:
            infos["depth"] = None
        if len(reset_env_ids) > 0:
            log = extras.get("log", {})
            episode = {key.replace("Episode_Reward/", "rew_"): value for key, value in log.items()
                       if key.startswith("Episode_Reward/")}
            if "Curriculum/terrain_levels" in log:
                episode["terrain_level"] = log["Curriculum/terrain_levels"]
            for key, value in log.items():
                if key.startswith("Episode_Termination/"):
                    episode["term_" + key.split("/", 1)[1]] = value
            infos["episode"] = episode
            terminal_amp = self.env.wmp_terminal_amp
        else:
            terminal_amp = torch.zeros(0, AMP_DIM, device=self.device)
        policy = obs["policy"]
        return policy, policy, rew, dones, infos, reset_env_ids, terminal_amp

    def contract(self) -> dict:
        from rbq10_parkour.contract import ASSETS, NATIVE_JOINT_NAMES, FOOT_NAMES, CAMERA_POSITION, CAMERA_ROTATION
        from . import scaling

        urdf = (ASSETS / "resources/urdf/rbq10.urdf").read_bytes()
        return {
            "schema": "rbq10_wmp_v1",
            "robot": "Rainbow_RBQ10_without_arm",
            "urdf_sha256": hashlib.sha256(urdf).hexdigest(),
            "joint_order": NATIVE_JOINT_NAMES, "foot_order": FOOT_NAMES,
            "action_scale_rad": 0.25, "physics_dt": float(self.cfg.sim.dt), "policy_dt": float(self.dt),
            "model_interval": self.model_interval, "obs_dim": self.num_obs, "prop_dim": self.prop_dim,
            "privileged_dim": self.privileged_dim, "height_dim": self.height_dim,
            "depth_shape": list(self.depth_shape), "camera_position_m": list(CAMERA_POSITION),
            "camera_quaternion_wxyz_ros": list(CAMERA_ROTATION),
            "length_scale": scaling.LENGTH,
            "amp_joint_order": list(self.cfg.amp_joint_order),
            "scaled_values_sha": hashlib.sha256(json.dumps(scaling.rbq10_values(), sort_keys=True).encode()).hexdigest()[:16],
        }
