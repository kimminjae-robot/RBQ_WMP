# Copyright (c) 2021 ETH Zurich, Nikita Rudin (BSD-3-Clause, rsl_rl)
# This file may have been modified by Bytedance Ltd. and/or its affiliates ("Bytedance's Modifications").
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.
#
# RBQ10 port of ``rsl_rl/runners/wmp_runner.py`` from https://github.com/bytedance/WMP.
# The rollout / world-model / PPO update order is kept. Differences (docs/WMP.md):
#   * every env has a real depth camera (Isaac Lab ray-cast camera), so the
#     heightmap->depth "DepthPredictor" used to fill non-camera envs is not needed;
#     the world-model dataset is recorded from ``wm_dataset_envs`` evenly strided envs;
#   * the env is accessed through :class:`WMPEnv` (rbq10_wmp/env_api.py);
#   * images are stored as float16 in the replay dataset to bound memory;
#   * checkpoints also store the AMP discriminator/normalizer and resume state.
"""World Model-based Perception (WMP) runner: RSSM world model + AMP-PPO policy."""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn

from .dreamer.config import make_world_model_config
from .dreamer.models import WorldModel
from .rl.actor_critic_wmp import ActorCriticWMP
from .rl.amp_discriminator import AMPDiscriminator
from .rl.amp_ppo import AMPPPO
from .rl.motion_loader import AMPLoader
from .rl.utils import Normalizer


class WMPRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu",
                 history_length: int = 5):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.wm_overrides = dict(train_cfg.get("world_model", {}))
        self.device = device
        self.env = env
        self.history_length = history_length
        num_critic_obs = env.num_privileged_obs
        num_actor_obs = env.num_obs

        self.wm_update_interval = int(env.model_interval)
        self._build_world_model()

        # Proprioceptive history without commands: (ang vel, gravity) + (q, qd, a).
        self.history_entry_dim = env.num_obs - env.privileged_dim - env.height_dim - 3
        self.history_dim = history_length * self.history_entry_dim
        actor_critic = ActorCriticWMP(
            num_actor_obs=num_actor_obs, num_critic_obs=num_critic_obs, num_actions=env.num_actions,
            height_dim=env.height_dim, privileged_dim=env.privileged_dim, history_dim=self.history_dim,
            wm_feature_dim=self.wm_feature_dim, **self.policy_cfg,
        ).to(self.device)

        self.amp_enabled = float(self.cfg["amp_reward_coef"]) > 0.0
        amp_data = AMPLoader(
            device, time_between_frames=env.dt, preload_transitions=True,
            num_preload_transitions=int(self.cfg["amp_num_preload_transitions"]),
            motion_files=self.cfg["amp_motion_files"],
        )
        if amp_data.observation_dim != env.num_amp_obs:
            raise ValueError(f"AMP reference dim {amp_data.observation_dim} != env AMP dim {env.num_amp_obs}")
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2, self.cfg["amp_reward_coef"], self.cfg["amp_discr_hidden_dims"],
            device, self.cfg["amp_task_reward_lerp"],
        ).to(self.device)
        min_std = torch.as_tensor(env.min_action_std, device=self.device, dtype=torch.float)
        self.alg = AMPPPO(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                          min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg["save_interval"])
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, [num_actor_obs], [num_critic_obs],
                              [env.num_actions], self.history_dim, self.wm_feature_dim)

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.stop_requested = False
        self.env.reset()

    # ------------------------------------------------------------------ build
    def _build_world_model(self):
        wm_device = self.wm_overrides.pop("device", self.device)
        self.wm_config = make_world_model_config(self.wm_update_interval, wm_device, **self.wm_overrides)
        obs_shape = {"prop": (self.env.prop_dim,), "image": tuple(self.env.depth_shape) + (1,)}
        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.env.use_camera)
        self._world_model = self._world_model.to(self._world_model.device)
        # Policy input is the deterministic recurrent state h_t (paper Eq. 6-7).
        self.wm_feature_dim = self.wm_config.dyn_deter

    # ------------------------------------------------------------- utilities
    def _prop(self, obs):
        pd = self.env.privileged_dim
        return obs[:, pd:pd + self.env.prop_dim]

    def _history_entry(self, obs):
        pd, hd = self.env.privileged_dim, self.env.height_dim
        return torch.cat((obs[:, pd:pd + 6], obs[:, pd + 9:obs.shape[1] - hd]), dim=1)

    def world_model_step(self, wm_latent, wm_action, wm_obs):
        """One posterior step of the RSSM; returns (latent, deterministic feature h_t)."""
        embed = self._world_model.encoder(wm_obs)
        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, embed, wm_obs["is_first"])
        return wm_latent, self._world_model.dynamics.get_deter_feat(wm_latent)

    # ----------------------------------------------------------------- learn
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        from torch.utils.tensorboard import SummaryWriter

        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        env, wm, wm_dev = self.env, self._world_model, self._world_model.device
        obs = env.get_observations().to(self.device)
        critic_obs = env.get_privileged_observations().to(self.device)
        amp_obs = env.get_amp_observations().to(self.device)
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer, lenbuffer = deque(maxlen=100), deque(maxlen=100)
        cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=self.device)
        tot_iter = self.current_learning_iteration + num_learning_iterations

        self.trajectory_history = torch.zeros(env.num_envs, self.history_length, self.history_entry_dim, device=self.device)
        self.trajectory_history = torch.cat((self.trajectory_history[:, 1:], self._history_entry(obs).unsqueeze(1)), dim=1)

        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(env.num_envs, device=wm_dev)
        wm_obs = {"prop": self._prop(obs).to(wm_dev), "is_first": wm_is_first}
        if env.use_camera:
            wm_obs["image"] = torch.zeros((env.num_envs,) + tuple(env.depth_shape) + (1,), device=wm_dev)
        k = self.wm_update_interval
        wm_action_history = torch.zeros(env.num_envs, k, env.num_actions, device=wm_dev)
        wm_reward = torch.zeros(env.num_envs, device=wm_dev)
        wm_feature = torch.zeros(env.num_envs, self.wm_feature_dim, device=wm_dev)
        self.init_wm_dataset()

        for it in range(self.current_learning_iteration, tot_iter):
            env.update_reward_curriculum(it)
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if env.global_counter % k == 0:
                        wm_latent, wm_feature = self.world_model_step(wm_latent, wm_action, wm_obs)
                        wm_is_first[:] = 0

                    history = self.trajectory_history.flatten(1).to(self.device)
                    actions = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature.to(self.device))
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = env.step(actions)
                    next_amp_obs = env.get_amp_observations()
                    critic_obs = privileged_obs
                    obs, critic_obs, next_amp_obs = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)

                    wm_action_history = torch.cat((wm_action_history[:, 1:], actions.unsqueeze(1).to(wm_dev)), dim=1)
                    wm_obs = {"prop": self._prop(obs).to(wm_dev), "is_first": wm_is_first}

                    # Move finished episodes from the per-env buffer into the dataset.
                    reset_np = reset_env_ids.cpu().numpy()
                    if len(reset_np) > 0:
                        self._flush_episodes(reset_np)
                        sum_wm_dataset_size = int(np.sum(self.wm_dataset_size))
                        wm_action_history[reset_env_ids.to(wm_dev)] = 0
                        wm_is_first[reset_env_ids.to(wm_dev)] = 1
                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(wm_dev)

                    if env.global_counter % k == 0:
                        if env.use_camera:
                            wm_obs["image"] = infos["depth"].unsqueeze(-1).to(wm_dev)
                        self._store_step(wm_obs, wm_action, wm_reward, wm_is_first)
                        wm_reward[:] = 0

                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    if len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids.to(self.device)] = terminal_amp_states.to(self.device)
                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    self.trajectory_history = torch.cat(
                        (self.trajectory_history[:, 1:], self._history_entry(obs).unsqueeze(1)), dim=1)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs, wm_feature.to(self.device))
            (mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss,
             mean_policy_pred, mean_expert_pred) = self.alg.update()
            learn_time = time.time() - start

            wm_start = time.time()
            wm_metrics = {}
            if sum_wm_dataset_size > self.wm_config.train_start_steps:
                wm_metrics = self.train_world_model()
            wm_time = time.time() - wm_start
            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if self.stop_requested:
                self.current_learning_iteration = it + 1
                break
        else:
            self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ------------------------------------------------- world-model dataset
    def init_wm_dataset(self):
        env = self.env
        n_data = min(int(self.cfg.get("wm_dataset_envs", env.num_envs)), env.num_envs)
        # Evenly strided envs so every terrain column contributes model data.
        self.wm_data_envs = np.unique(np.linspace(0, env.num_envs - 1, n_data).round().astype(np.int64))
        self.wm_env_to_slot = -np.ones(env.num_envs, dtype=np.int64)
        self.wm_env_to_slot[self.wm_data_envs] = np.arange(len(self.wm_data_envs))
        n, T = len(self.wm_data_envs), int(env.max_episode_length / self.wm_update_interval) + 3
        dev = self._world_model.device
        img_dtype = torch.float16 if self.cfg.get("wm_image_fp16", True) else torch.float32

        def alloc(device):
            data = {
                "prop": torch.zeros((n, T, env.prop_dim), device=device),
                "action": torch.zeros((n, T, env.num_actions * self.wm_update_interval), device=device),
                "reward": torch.zeros((n, T), device=device),
            }
            if env.use_camera:
                data["image"] = torch.zeros((n, T) + tuple(env.depth_shape) + (1,), device=device, dtype=img_dtype)
            return data

        self.wm_dataset = alloc(dev)
        self.wm_buffer = alloc(self.cfg.get("wm_buffer_device", "cpu"))
        self.wm_dataset_size = np.zeros(n, dtype=np.int64)
        self.wm_buffer_index = np.zeros(n, dtype=np.int64)
        self.wm_capacity = T

    def _flush_episodes(self, reset_env_ids):
        slots = self.wm_env_to_slot[reset_env_ids]
        slots = slots[slots >= 0]
        if len(slots) == 0:
            return
        for key, value in self.wm_dataset.items():
            value[slots] = self.wm_buffer[key][slots].to(value.device)
        self.wm_dataset_size[slots] = self.wm_buffer_index[slots]
        self.wm_buffer_index[slots] = 0

    def _store_step(self, wm_obs, wm_action, wm_reward, wm_is_first):
        """Append the k-step model transition of data envs that did not just reset."""
        not_first = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
        slots = self.wm_env_to_slot[not_first]
        keep = slots >= 0
        envs, slots = not_first[keep], slots[keep]
        # Guard against episodes longer than the buffer (e.g. evaluation configs).
        ok = self.wm_buffer_index[slots] < self.wm_capacity
        envs, slots = envs[ok], slots[ok]
        if len(slots) == 0:
            return
        idx = self.wm_buffer_index[slots]
        bdev = self.wm_buffer["prop"].device
        envs_t = torch.as_tensor(envs, device=wm_obs["prop"].device)
        for key in ("prop", "image"):
            if key in wm_obs and key in self.wm_buffer:
                self.wm_buffer[key][slots, idx] = wm_obs[key][envs_t].to(bdev, self.wm_buffer[key].dtype)
        self.wm_buffer["action"][slots, idx] = wm_action[envs_t].to(bdev)
        self.wm_buffer["reward"][slots, idx] = wm_reward[envs_t].to(bdev)
        self.wm_buffer_index[slots] += 1

    def sample_wm_batch(self, rng=np.random):
        """Sample (batch, length) segments exactly as the original runner."""
        p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
        batch_idx = rng.choice(len(self.wm_dataset_size), self.wm_config.batch_size, replace=True, p=p)
        batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
        if batch_length <= 1:
            return None
        end_idx = np.array([rng.randint(batch_length, self.wm_dataset_size[i] + 1) for i in batch_idx])
        offsets = end_idx[:, None] - batch_length + np.arange(batch_length)[None]
        rows = torch.as_tensor(batch_idx[:, None].repeat(batch_length, 1), device=self._world_model.device)
        cols = torch.as_tensor(offsets, device=self._world_model.device)
        batch = {key: value[rows, cols].float() for key, value in self.wm_dataset.items()}
        is_first = torch.zeros((self.wm_config.batch_size, batch_length), device=self._world_model.device)
        is_first[:, 0] = 1
        batch["is_first"] = is_first
        return batch

    def train_world_model(self):
        metrics = {}
        for _ in range(self.wm_config.train_steps_per_iter):
            batch = self.sample_wm_batch()
            if batch is None:
                continue
            _, _, mets = self._world_model._train(batch)
            metrics.update(mets)
        return metrics

    # ------------------------------------------------------------- logging
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        iteration_time = locs["collection_time"] + locs["learn_time"] + locs["wm_time"]
        self.tot_time += iteration_time
        it = locs["it"]
        scalars = {}
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                values = []
                for ep_info in locs["ep_infos"]:
                    v = ep_info[key]
                    v = v if isinstance(v, torch.Tensor) else torch.tensor([float(v)])
                    values.append(v.reshape(-1).float().cpu())
                scalars["Episode/" + key] = float(torch.cat(values).mean())
        fps = int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-9))
        scalars.update({
            "Loss/value_function": locs["mean_value_loss"], "Loss/surrogate": locs["mean_surrogate_loss"],
            "Loss/vel_predict": locs["mean_vel_predict_loss"], "Loss/AMP": locs["mean_amp_loss"],
            "Loss/AMP_grad": locs["mean_grad_pen_loss"], "Loss/learning_rate": self.alg.learning_rate,
            "Loss/AMP_mean_policy_pred": locs["mean_policy_pred"], "Loss/AMP_mean_expert_pred": locs["mean_expert_pred"],
            "Policy/mean_noise_std": float(self.alg.actor_critic.std.detach().mean()),
            "Perf/total_fps": fps, "Perf/collection time": locs["collection_time"],
            "Perf/learning_time": locs["learn_time"], "Perf/world_model_time": locs["wm_time"],
            "World_model/dataset_steps": float(np.sum(self.wm_dataset_size)),
        })
        for name, values in locs["wm_metrics"].items():
            scalars["World_model/" + name] = float(np.mean(values))
        if len(locs["rewbuffer"]) > 0:
            scalars["Train/mean_reward"] = statistics.mean(locs["rewbuffer"])
            scalars["Train/mean_episode_length"] = statistics.mean(locs["lenbuffer"])
        for key, value in scalars.items():
            self.writer.add_scalar(key, value, it)
        entry = {"iteration": it, "total_timesteps": self.tot_timesteps, "total_time_s": self.tot_time, **scalars}
        with open(Path(self.log_dir) / "progress.jsonl", "a") as stream:
            stream.write(json.dumps(entry) + "\n")
        title = f" WMP iteration {it}/{self.current_learning_iteration + locs['num_learning_iterations']} "
        lines = [f"{'#' * width}", title.center(width), ""]
        for key in ("Train/mean_reward", "Train/mean_episode_length", "Episode/terrain_level", "Loss/value_function",
                    "Loss/surrogate", "Loss/vel_predict", "Loss/AMP", "World_model/model_loss", "World_model/image_loss",
                    "World_model/prop_loss", "World_model/kl", "Policy/mean_noise_std"):
            if key in scalars:
                lines.append(f"{key + ':':>{pad}} {scalars[key]:.4f}")
        lines.append(f"{'Computation:':>{pad}} {fps} steps/s (collect {locs['collection_time']:.2f}s, "
                     f"PPO {locs['learn_time']:.2f}s, WM {locs['wm_time']:.2f}s)")
        print("\n".join(lines), flush=True)

    # ------------------------------------------------------------ checkpoint
    def save(self, path, infos=None):
        temporary = str(path) + ".tmp"
        torch.save({
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "world_model_dict": self._world_model.state_dict(),
            "wm_optimizer_state_dict": self._world_model._model_opt._opt.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": {"mean": self.alg.amp_normalizer.mean, "var": self.alg.amp_normalizer.var,
                               "count": self.alg.amp_normalizer.count},
            "learning_rate": self.alg.learning_rate,
            "iter": self.current_learning_iteration,
            "wmp_contract": self.env.contract(),
            "infos": infos,
        }, temporary)
        os.replace(temporary, path)

    def load(self, path, load_optimizer=True, load_wm_optimizer=True):
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved.get("wmp_contract") != self.env.contract():
            raise ValueError("Checkpoint was trained with a different RBQ10 WMP observation/robot contract")
        self.alg.actor_critic.load_state_dict(saved["model_state_dict"])
        self._world_model.load_state_dict(saved["world_model_dict"])
        if "discriminator_state_dict" in saved:
            self.alg.discriminator.load_state_dict(saved["discriminator_state_dict"])
            norm = saved["amp_normalizer"]
            self.alg.amp_normalizer.mean, self.alg.amp_normalizer.var = norm["mean"], norm["var"]
            self.alg.amp_normalizer.count = norm["count"]
        if load_wm_optimizer:
            self._world_model._model_opt._opt.load_state_dict(saved["wm_optimizer_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
            self.alg.learning_rate = saved.get("learning_rate", self.alg.learning_rate)
        self.current_learning_iteration = saved["iter"]
        return saved.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
