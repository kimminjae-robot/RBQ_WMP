"""Simulator-free stand-in for :class:`rbq10_wmp.env_api.WMPEnv` (CPU tests only).

Not a physics model. It has the exact WMP tensor interface (285-dim obs,
33-dim prop, 64x64 depth every k steps, 30-dim AMP state, terminal AMP states,
time-outs) and a small task where *vision matters*: each env walks toward a
box obstacle whose distance is only visible in the depth image. The robot must
command a "high step" (a fixed action direction) when the box is close, else
the episode terminates. A policy that uses the world-model feature h_t can
learn this; a blind one cannot. This lets the full RSSM + PPO + AMP loop be
checked end to end on CPU.
"""
from __future__ import annotations

import math

import torch

from .env_api import WMPEnvBase


class MockWMPEnv(WMPEnvBase):
    def __init__(self, num_envs=64, device="cpu", model_interval=5, depth_shape=(64, 64),
                 episode_length=200, seed=0, dt=0.02):
        self.num_envs, self.device = num_envs, device
        self.model_interval, self.depth_shape = model_interval, tuple(depth_shape)
        self.dt = dt
        self.max_episode_length = episode_length
        self.min_action_std = [0.05, 0.02, 0.05] * 4
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.gen = g
        self.default_q = torch.tensor([0.0, 0.81, -1.40] * 4, device=device)
        self.walk_dir = torch.nn.functional.normalize(torch.randn(12, generator=g), dim=0).to(device)
        jump = torch.randn(12, generator=g)
        jump -= (jump @ self.walk_dir.cpu()) * self.walk_dir.cpu()
        self.jump_dir = torch.nn.functional.normalize(jump, dim=0).to(device)
        n = num_envs
        self.q = self.default_q.repeat(n, 1)
        self.qd = torch.zeros(n, 12, device=device)
        self.v = torch.zeros(n, 3, device=device)
        self.w = torch.zeros(n, 3, device=device)
        self.cmd = torch.zeros(n, 3, device=device)
        self.obstacle = torch.zeros(n, device=device)   # distance to box [m]
        self.box_h = torch.zeros(n, device=device)
        self.actions = torch.zeros(n, 12, device=device)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=device)
        self.global_counter = 0
        self.depth_buffer = torch.zeros(n, 2, *self.depth_shape, device=device)
        self.feet_edge_coef = 0.1
        # Camera rays: elevation of each image row (58 deg vertical FOV), 0.5 m camera height.
        rows = torch.linspace(math.radians(29), math.radians(-29), self.depth_shape[0], device=device)
        self.row_tan = torch.tan(rows)
        self.cam_h, self.far = 0.5, 3.3
        self.episode_sums = {k: torch.zeros(n, device=device) for k in ("tracking", "obstacle", "action_rate")}
        self._reset(torch.arange(n, device=device))
        self._update_depth(force=True)

    # ---------------------------------------------------------------- helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen).to(self.device)

    def _reset(self, ids):
        k = len(ids)
        self.q[ids] = self.default_q + 0.1 * (self._rand(k, 12) - 0.5)
        self.qd[ids] = 0
        self.v[ids] = 0
        self.w[ids] = 0
        self.cmd[ids, 0] = 0.3 + 0.7 * self._rand(k)
        self.obstacle[ids] = 1.5 + 2.0 * self._rand(k)
        self.box_h[ids] = 0.1 + 0.2 * self._rand(k)
        self.actions[ids] = 0
        self.episode_length_buf[ids] = 0
        for v in self.episode_sums.values():
            v[ids] = 0

    def _render(self):
        """Depth of a flat floor with a box of height ``box_h`` at ``obstacle`` m (normalized like WMP)."""
        t = self.row_tan[None, :]                               # (1, H)
        ground = torch.where(t < 0, self.cam_h / (-t).clamp_min(1e-6), torch.full_like(t, self.far))
        d = self.obstacle[:, None]
        hit_height = self.cam_h + t * d                         # ray height at the box face
        hits_box = (hit_height <= self.box_h[:, None]) & (d > 0) & (ground > d)
        ground = ground.expand(self.num_envs, -1)
        depth = torch.where(hits_box, d.expand_as(ground), ground).clamp(0, self.far)
        img = depth[:, :, None].expand(-1, -1, self.depth_shape[1])
        return img / self.far - 0.5

    def _update_depth(self, force=False):
        if not force and self.global_counter % self.model_interval != 0:
            return
        image = self._render()
        fill = (self.episode_length_buf <= 1)[:, None, None, None] | force
        rolled = torch.cat((self.depth_buffer[:, 1:], image[:, None]), dim=1)
        self.depth_buffer = torch.where(fill, image[:, None].expand_as(self.depth_buffer), rolled)

    def _obs(self):
        n = self.num_envs
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, 3)
        privileged = torch.cat((torch.zeros(n, 50, device=self.device), self.v), dim=1)
        prop = torch.cat((self.w * 0.25, gravity, self.cmd * torch.tensor([1.0, 1.0, 0.25], device=self.device),
                          self.q - self.default_q, self.qd * 0.05), dim=1)
        heights = torch.zeros(n, self.height_dim, device=self.device)
        near = (self.obstacle < 1.0).float()[:, None]
        heights[:, 100:] = near * self.box_h[:, None] * 5.0
        return torch.cat((privileged, prop, self.actions, heights), dim=1)

    # -------------------------------------------------------------------- API
    def reset(self):
        self._reset(torch.arange(self.num_envs, device=self.device))
        self._update_depth(force=True)
        obs = self._obs()
        return obs, obs

    def get_observations(self):
        return self._obs()

    def get_privileged_observations(self):
        return self._obs()

    def get_amp_observations(self):
        return torch.cat((self.q, self.v, self.w, self.qd), dim=1)

    def update_reward_curriculum(self, iteration):
        p = min(max((iteration - 4000) / 6000, 0.0), 1.0)
        self.feet_edge_coef = 0.1 + 0.9 * p

    def step(self, actions):
        self.global_counter += 1
        actions = torch.clip(actions, -6.0, 6.0).to(self.device)
        last_actions = self.actions.clone()
        self.actions = actions
        target = self.default_q + 0.25 * actions
        q_new = self.q + 0.5 * (target - self.q)
        self.qd = (q_new - self.q) / self.dt
        self.q = q_new
        speed = 1.2 * torch.sigmoid(2.0 * actions @ self.walk_dir)
        self.v[:, 0] = speed
        self.w[:, 2] = 0.1 * torch.tanh(actions[:, 0])
        self.obstacle -= speed * self.dt
        lifting = (actions @ self.jump_dir) > 1.0
        at_box = (self.obstacle < 0.25) & (self.obstacle > 0.0)
        crash = at_box & ~lifting
        passed = self.obstacle <= 0.0
        self.obstacle = torch.where(passed, 1.5 + 2.0 * self._rand(self.num_envs), self.obstacle)

        self.episode_length_buf += 1
        tracking = torch.exp(-torch.square(self.v[:, 0] - self.cmd[:, 0]) / 0.25) * 1.5 * self.dt
        lift_cost = -0.5 * self.dt * lifting.float() * (~at_box).float()
        action_rate = -0.03 * self.dt * torch.sum(torch.square(actions - last_actions), dim=1)
        crash_pen = -2.0 * crash.float()
        rew = tracking + lift_cost + action_rate + crash_pen
        self.episode_sums["tracking"] += tracking
        self.episode_sums["obstacle"] += crash_pen + lift_cost
        self.episode_sums["action_rate"] += action_rate

        time_out = self.episode_length_buf > self.max_episode_length
        dones = crash | time_out
        ids = dones.nonzero(as_tuple=False).flatten()
        terminal_amp = self.get_amp_observations()[ids]
        infos = {"time_outs": time_out}
        if len(ids) > 0:
            infos["episode"] = {"rew_" + k: torch.mean(v[ids]) / (self.max_episode_length * self.dt)
                                for k, v in self.episode_sums.items()}
            infos["episode"]["crash_rate"] = crash.float().sum() / len(ids)
            self._reset(ids)
        self._update_depth()
        infos["depth"] = self.depth_buffer[:, -2] if self.global_counter % self.model_interval == 0 else None
        obs = self._obs()
        return obs, obs, rew, dones, infos, ids, terminal_amp

    def contract(self):
        return {"schema": "rbq10_wmp_mock_v1", "obs_dim": self.num_obs, "model_interval": self.model_interval,
                "depth_shape": list(self.depth_shape)}
