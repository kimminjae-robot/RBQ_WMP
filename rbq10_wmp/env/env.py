"""Isaac Lab manager-based env with the WMP bookkeeping the MDP terms need."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from rbq10_parkour.contract import FOOT_NAMES, NATIVE_JOINT_NAMES

from .terrain import WMPTerrainGenerator


class RBQ10WMPEnv(ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` plus WMP state:

    * ``wmp_type_ranges``  env index range of each terrain type (legged_gym ``*_start_idx``),
    * ``wmp_edge_mask``    vertical-edge cells of the height field (``feet_edge`` reward),
    * ``wmp_global_counter`` policy steps since start (depth / world-model clock),
    * ``wmp_terminal_amp`` AMP state of the envs reset in the last step, captured before reset.
    """

    def load_managers(self):
        # Terms query these while the managers are built, so set them first.
        gen = WMPTerrainGenerator.last_instance
        if gen is None:
            raise RuntimeError("RBQ10 WMP requires the WMP terrain generator")
        terrain = gen.terrain
        self.wmp_terrain = terrain
        self.wmp_type_ranges = terrain.type_env_ranges(self.num_envs)
        self.wmp_edge_mask = torch.as_tensor(terrain.x_edge_mask, device=self.device, dtype=torch.bool)
        self.wmp_border = float(terrain.cfg.border_size)
        self.wmp_horizontal_scale = float(terrain.cfg.horizontal_scale)
        self.wmp_env_length = float(terrain.env_length)
        self.wmp_global_counter = 0
        self.wmp_index_cache = {}
        self.wmp_terminal_amp = torch.zeros(0, 30, device=self.device)
        self._wmp_reward_weights = {}
        super().load_managers()
        robot = self.scene["robot"]
        if robot.joint_names != NATIVE_JOINT_NAMES:
            raise ValueError(f"RBQ10 joint order differs from the WMP contract: {robot.joint_names}")
        if not set(FOOT_NAMES).issubset(robot.body_names):
            raise ValueError("RBQ10 fixed feet must be preserved as separate contact bodies")
        self._wmp_reward_weights = {name: self.reward_manager.get_term_cfg(name).weight
                                    for name in self.reward_manager.active_terms}

    # ----------------------------------------------------------------- AMP
    def amp_observation(self, env_ids: Sequence[int] | slice | None = None) -> torch.Tensor:
        from .mdp import wmp_amp_observation
        from rbq10_parkour.contract import JOINT_NAMES
        from isaaclab.managers import SceneEntityCfg

        amp = wmp_amp_observation(self, SceneEntityCfg("robot"), getattr(self.cfg, "amp_joint_order", JOINT_NAMES))
        return amp if env_ids is None else amp[env_ids]

    def _reset_idx(self, env_ids: Sequence[int]):
        # WMP post_physics_step: terminal AMP states are taken before reset_idx.
        self.wmp_terminal_amp = self.amp_observation(env_ids).clone()
        super()._reset_idx(env_ids)

    def step(self, action):
        self.wmp_global_counter += 1  # WMP increments the global counter before simulating
        return super().step(action)

    # ---------------------------------------------------- reward curriculum
    def update_reward_curriculum(self, iteration: int):
        """WMP ``update_reward_curriculum``: linear ramp of selected reward weights."""
        for name, (start, end, w0, w1) in getattr(self.cfg, "reward_curriculum", {}).items():
            if name not in self._wmp_reward_weights:
                continue
            progress = min(max((iteration - start) / (end - start), 0.0), 1.0)
            coef = (1.0 - progress) * w0 + progress * w1
            term = self.reward_manager.get_term_cfg(name)
            term.weight = self._wmp_reward_weights[name] * coef
            self.reward_manager.set_term_cfg(name, term)
