"""Deployment-side WMP loop: world model posterior every k steps + actor (WMP ``play.py``)."""
from __future__ import annotations

import torch


class WMPPolicy:
    """Stateful policy: call :meth:`reset` once, then :meth:`act` every policy step and
    :meth:`observe` with the step results. Mirrors the rollout in :class:`WMPRunner.learn`."""

    def __init__(self, runner, env):
        self.runner, self.env = runner, env
        self.wm = runner._world_model
        self.k = runner.wm_update_interval
        self.actor_critic = runner.alg.actor_critic.eval()
        self.wm.eval()

    @torch.inference_mode()
    def reset(self, obs):
        env, wm_dev, n = self.env, self.wm.device, self.env.num_envs
        self.obs = obs
        self.latent = None
        self.wm_action = None
        self.is_first = torch.ones(n, device=wm_dev)
        self.wm_obs = {"prop": self.runner._prop(obs).to(wm_dev), "is_first": self.is_first}
        if env.use_camera:
            self.wm_obs["image"] = torch.zeros((n,) + tuple(env.depth_shape) + (1,), device=wm_dev)
        self.action_history = torch.zeros(n, self.k, env.num_actions, device=wm_dev)
        self.feature = torch.zeros(n, self.runner.wm_feature_dim, device=wm_dev)
        self.history = torch.zeros(n, self.runner.history_length, self.runner.history_entry_dim, device=obs.device)
        self.history = torch.cat((self.history[:, 1:], self.runner._history_entry(obs).unsqueeze(1)), dim=1)

    @torch.inference_mode()
    def act(self):
        if self.env.global_counter % self.k == 0:
            self.latent, self.feature = self.runner.world_model_step(self.latent, self.wm_action, self.wm_obs)
            self.is_first[:] = 0
        return self.actor_critic.act_inference(self.obs, self.history.flatten(1), self.feature.to(self.obs.device))

    @torch.inference_mode()
    def observe(self, actions, obs, dones, infos, reset_env_ids):
        wm_dev = self.wm.device
        self.obs = obs
        self.action_history = torch.cat((self.action_history[:, 1:], actions.unsqueeze(1).to(wm_dev)), dim=1)
        self.wm_obs = {"prop": self.runner._prop(obs).to(wm_dev), "is_first": self.is_first}
        if len(reset_env_ids) > 0:
            self.action_history[reset_env_ids.to(wm_dev)] = 0
            self.is_first[reset_env_ids.to(wm_dev)] = 1
        self.wm_action = self.action_history.flatten(1)
        if self.env.global_counter % self.k == 0 and self.env.use_camera:
            self.wm_obs["image"] = infos["depth"].unsqueeze(-1).to(wm_dev)
        ids = dones.nonzero(as_tuple=False).flatten()
        self.history[ids] = 0
        self.history = torch.cat((self.history[:, 1:], self.runner._history_entry(obs).unsqueeze(1)), dim=1)

    @torch.inference_mode()
    def decode_depth(self):
        """Depth image the world model reconstructs from its current latent (for visualization)."""
        feat = self.wm.dynamics.get_feat(self.latent)
        return self.wm.heads["decoder"](feat.unsqueeze(1))["image"].mode()[:, 0, ..., 0]
