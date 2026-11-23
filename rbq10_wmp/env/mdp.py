# Rewards, terminations, commands, observations and events ported from bytedance/WMP
# ``legged_gym/envs/base/legged_robot.py`` (BSD-3-Clause, ETH Zurich / NVIDIA, with
# Bytedance's modifications) to Isaac Lab manager terms for RBQ10.
"""Isaac Lab MDP terms reproducing the WMP environment (paper Sec. IV-C)."""
from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg
    from .env import RBQ10WMPEnv


def _wmp(env) -> "RBQ10WMPEnv":
    return env  # typed accessor; the env keeps WMP terrain bookkeeping


def type_mask(env, *names) -> torch.Tensor:
    """Boolean mask of envs whose terrain type is in ``names`` (WMP *_start_idx ranges)."""
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in names:
        start, end = env.wmp_type_ranges[name]
        mask[start:end] = True
    return mask


def terrain_levels(env) -> torch.Tensor:
    return env.scene.terrain.terrain_levels


# =============================================================== observations
class WMPObservation(ManagerTermBase):
    """Full WMP observation ``[privileged(53) | prop(33) | action(12) | heights(187)]``.

    Same layout as WMP ``compute_observations``: the actor only reads the command
    and the proprioceptive history from it; the critic reads everything
    (asymmetric actor-critic, paper Eq. 6-7).
    """

    def __init__(self, cfg: ObservationTermCfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.asset: Articulation = env.scene[p["asset_cfg"].name]
        self.contacts: ContactSensor = env.scene.sensors[p["contact_cfg"].name]
        self.scanner: RayCaster = env.scene.sensors[p["height_cfg"].name]
        self.feet_ids, _ = self.contacts.find_bodies(p["foot_names"], preserve_order=True)
        self.penalised_ids, _ = self.contacts.find_bodies(p["penalised_names"])
        self.trunk_id = self.asset.find_bodies(p["base_body"])[0][0]
        # Observation terms are built before the "startup" events, so these are the nominal values.
        self.default_mass = self.asset.root_physx_view.get_masses()[:, self.trunk_id].clone().to(self.device)
        self.default_com = self.asset.root_physx_view.get_coms()[:, self.trunk_id, :3].clone().to(self.device)

    def __call__(self, env, asset_cfg, contact_cfg, height_cfg, foot_names, penalised_names, base_body,
                 base_height, scales):
        a = self.asset
        forces = self.contacts.data.net_forces_w
        contact_flag = (torch.norm(forces[:, self.penalised_ids], dim=-1) > 0.1).float()
        contact_force = forces[:, self.feet_ids].flatten(1) * scales["contact_force"]
        p_ratio = (a.data.joint_stiffness / a.data.default_joint_stiffness - 1) * scales["pd_gains"]
        d_ratio = (a.data.joint_damping / a.data.default_joint_damping - 1) * scales["pd_gains"]
        masses = a.root_physx_view.get_masses()[:, self.trunk_id].to(self.device)
        com = a.root_physx_view.get_coms()[:, self.trunk_id, :3].to(self.device)
        added_mass = (masses - self.default_mass).unsqueeze(-1)
        com_offset = (com - self.default_com) * scales["com_pos"]
        material = a.root_physx_view.get_material_properties().to(self.device)
        friction = material[:, 0, 0:1]
        restitution = material[:, 0, 2:3]
        command = env.command_manager.get_command("base_velocity")[:, :3]
        prop = torch.cat((
            a.data.root_ang_vel_b * scales["ang_vel"],
            a.data.projected_gravity_b,
            command * torch.tensor([scales["lin_vel"], scales["lin_vel"], scales["ang_vel"]], device=self.device),
            (a.data.joint_pos - a.data.default_joint_pos) * scales["dof_pos"],
            a.data.joint_vel * scales["dof_vel"],
        ), dim=-1)
        heights = torch.clip(
            a.data.root_pos_w[:, 2:3] - base_height - self.scanner.data.ray_hits_w[..., 2], -1.0, 1.0
        ) * scales["height_measurements"]
        heights = torch.nan_to_num(heights, nan=-1.0 * scales["height_measurements"], posinf=-scales["height_measurements"],
                                   neginf=scales["height_measurements"])
        privileged = torch.cat((contact_flag, contact_force, d_ratio, p_ratio, com_offset, added_mass,
                                restitution, friction, a.data.root_lin_vel_b * scales["lin_vel"]), dim=-1)
        return torch.cat((privileged, prop, env.action_manager.action, heights), dim=-1)


def wmp_amp_observation(env, asset_cfg: SceneEntityCfg, joint_order: Sequence[str]):
    """AMP state ``[q(12), v_b(3), w_b(3), qd(12)]`` in the reference-motion joint order."""
    asset: Articulation = env.scene[asset_cfg.name]
    index = env.wmp_index_cache.get("amp")
    if index is None:
        index = torch.tensor([asset.joint_names.index(n) for n in joint_order], device=env.device)
        env.wmp_index_cache["amp"] = index
    return torch.cat((asset.data.joint_pos[:, index], asset.data.root_lin_vel_b, asset.data.root_ang_vel_b,
                      asset.data.joint_vel[:, index]), dim=-1)


class WMPDepthImage(ManagerTermBase):
    """Egocentric 64x64 depth, refreshed every ``k`` policy steps, delivered with one
    refresh (k x 20 ms = 100 ms) of latency (WMP ``update_depth_buffer`` / extras["depth"])."""

    def __init__(self, cfg: ObservationTermCfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.camera: RayCasterCamera = env.scene.sensors[p["sensor_cfg"].name]
        h, w = p["shape"]
        self.buffer = torch.zeros(self.num_envs, p["buffer_len"], h, w, device=self.device)
        self.needs_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_counter = -1

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.needs_fill[env_ids] = True

    def _process(self, near, far):
        depth = self.camera.data.output["distance_to_image_plane"][..., 0]
        depth = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near)
        depth = torch.clip(depth, near, far)
        return (depth - near) / (far - near) - 0.5

    def __call__(self, env, sensor_cfg, shape, buffer_len, interval, near_clip, far_clip):
        counter = env.wmp_global_counter
        if counter % interval == 0 and counter != self.last_counter:
            self.last_counter = counter
            image = self._process(near_clip, far_clip)
            fill = self.needs_fill | (env.episode_length_buf <= 1)
            rolled = torch.cat((self.buffer[:, 1:], image.unsqueeze(1)), dim=1)
            self.buffer[:] = torch.where(fill[:, None, None, None], image.unsqueeze(1).expand_as(self.buffer), rolled)
            self.needs_fill[:] = False
        return self.buffer[:, -2].reshape(self.num_envs, -1)


# ==================================================================== actions
class LatencyJointPositionAction(JointPositionAction):
    """Joint position targets with WMP action latency: the previous target is held
    for a random number of physics sub-steps in ``latency_range_s`` each policy step."""

    cfg: "LatencyJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        dt = env.physics_dt
        lo, hi = cfg.latency_range_s
        self._latency_steps = (int((lo + 1e-8) / dt), int((hi - 1e-8) / dt) + 1)
        self._previous = torch.zeros_like(self.processed_actions)
        self._substep = 0
        self._latency = 0

    def process_actions(self, actions):
        self._previous[:] = self.processed_actions
        super().process_actions(actions)
        self._substep = 0
        if self.cfg.randomize_latency:
            self._latency = int(torch.randint(self._latency_steps[0], self._latency_steps[1] + 1, (1,)).item())

    def apply_actions(self):
        target = self._previous if self._substep < self._latency else self.processed_actions
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._substep += 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._previous[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids] \
            if self.cfg.use_default_offset else 0.0


@configclass
class LatencyJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = LatencyJointPositionAction
    randomize_latency: bool = True
    latency_range_s: tuple[float, float] = (0.0, 0.005)


# =================================================================== commands
class WMPVelocityCommand(UniformVelocityCommand):
    """WMP ``_resample_commands``: heading control on obstacle terrains (target heading 0),
    direct yaw-rate commands on rough-flat terrain, small commands zeroed."""

    cfg: "WMPVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        r = torch.empty(len(env_ids), device=self.device)
        flat = type_mask(self._env, "rough_flat")[env_ids]
        ranges, flat_ranges = self.cfg.ranges, self.cfg.flat_ranges
        self.vel_command_b[env_ids, 0] = torch.where(flat, r.uniform_(*flat_ranges.lin_vel_x),
                                                     torch.empty_like(r).uniform_(*ranges.lin_vel_x))
        self.vel_command_b[env_ids, 1] = torch.where(flat, torch.empty_like(r).uniform_(*flat_ranges.lin_vel_y),
                                                     torch.empty_like(r).uniform_(*ranges.lin_vel_y))
        self.vel_command_b[env_ids, 2] = torch.where(flat, torch.empty_like(r).uniform_(*flat_ranges.ang_vel_z),
                                                     self.vel_command_b[env_ids, 2])
        self.heading_target[env_ids] = torch.empty_like(r).uniform_(*ranges.heading)
        self.is_heading_env[env_ids] = ~flat
        self.is_standing_env[env_ids] = False
        small = torch.norm(self.vel_command_b[env_ids, :2], dim=1) <= 0.2
        self.vel_command_b[env_ids[small], :2] = 0.0
        zero_heading = type_mask(self._env, "tilt", "climb")[env_ids]
        self.heading_target[env_ids[zero_heading]] = 0.0

    def _update_command(self):
        env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
        heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
        lo, hi = self.cfg.ranges.ang_vel_z
        self.vel_command_b[env_ids, 2] = torch.clip(self.cfg.heading_control_stiffness * heading_error, lo, hi)


@configclass
class WMPVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = WMPVelocityCommand
    flat_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.8), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=None)


# ==================================================================== rewards
def _asset(env, name="robot") -> Articulation:
    return env.scene[name]


def reward_tracking_lin_vel(env, sigma: float, lin_vel_clip: float):
    """Paper Eq. (9): velocity above the command (by < clip) is not penalised."""
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel = _asset(env).data.root_lin_vel_b[:, :2]
    upper = torch.where(cmd < 0, torch.full_like(cmd, 1e5), cmd + lin_vel_clip)
    lower = torch.where(cmd > 0, torch.full_like(cmd, -1e5), cmd - lin_vel_clip)
    error = torch.sum(torch.square(cmd - torch.clip(lin_vel, lower, upper)), dim=1)
    return torch.exp(-error / sigma)


def reward_tracking_ang_vel(env, sigma: float):
    cmd = env.command_manager.get_command("base_velocity")[:, 2]
    return torch.exp(-torch.square(cmd - _asset(env).data.root_ang_vel_b[:, 2]) / sigma)


def reward_torques(env):
    return torch.sum(torch.square(_asset(env).data.applied_torque), dim=1)


def reward_dof_acc(env):
    return torch.sum(torch.square(_asset(env).data.joint_acc), dim=1)


def reward_action_rate(env):
    return torch.sum(torch.square(env.action_manager.prev_action - env.action_manager.action), dim=1)


def reward_collision(env, sensor_cfg: SceneEntityCfg):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.sum((torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > 0.1).float(), dim=1)


def reward_dof_error(env):
    a = _asset(env)
    return torch.sum(torch.square(a.data.joint_pos - a.data.default_joint_pos), dim=1)


def reward_lin_vel_z(env):
    return torch.square(_asset(env).data.root_lin_vel_b[:, 2])


def reward_cheat(env):
    """Penalise heading away (|yaw| > 1 rad) to bypass obstacles, except on rough flat."""
    heading = _asset(env).data.heading_w
    return ((heading > 1.0) | (heading < -1.0)).float() * (~type_mask(env, "rough_flat")).float()


def reward_stuck(env):
    v = _asset(env).data.root_lin_vel_b[:, 0]
    cmd = env.command_manager.get_command("base_velocity")[:, 0]
    return ((torch.abs(v) < 0.1) & (torch.abs(cmd) > 0.1)).float()


class FootContactState:
    """Shared filtered foot contact (WMP ``contact_filt`` with ``last_contacts``)."""

    @staticmethod
    def get(env, sensor_cfg: SceneEntityCfg):
        state = env.wmp_index_cache.setdefault("contact_state", {})
        counter = env.common_step_counter
        if state.get("counter") != counter:
            sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
            contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 1.0
            last = state.get("last", torch.zeros_like(contact))
            state["filt"] = contact | last
            state["last"] = contact
            state["counter"] = counter
        return state["filt"]


class reward_feet_air_time(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self.air_time = torch.zeros(env.num_envs, 4, device=env.device)

    def reset(self, env_ids=None):
        self.air_time[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, sensor_cfg: SceneEntityCfg, target: float = 0.5):
        contact_filt = FootContactState.get(env, sensor_cfg)
        first_contact = (self.air_time > 0.0) & contact_filt
        self.air_time += env.step_dt
        reward = torch.sum((self.air_time - target) * first_contact, dim=1)
        reward *= torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
        self.air_time *= ~contact_filt
        return reward


def reward_feet_edge(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Feet in contact on a vertical-edge cell of the height field (gap/climb, level > 3)."""
    contact_filt = FootContactState.get(env, sensor_cfg)
    feet = _asset(env).data.body_pos_w[:, asset_cfg.body_ids, :2]
    mask = env.wmp_edge_mask
    idx = ((feet + env.wmp_border) / env.wmp_horizontal_scale).round().long()
    ix = idx[..., 0].clip(0, mask.shape[0] - 1)
    iy = idx[..., 1].clip(0, mask.shape[1] - 1)
    at_edge = contact_filt & mask[ix, iy]
    reward = (terrain_levels(env) > 3) * torch.sum(at_edge, dim=-1)
    return reward.float() * type_mask(env, "gap", "climb").float()


def reward_feet_stumble(env, sensor_cfg: SceneEntityCfg):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    hit = torch.any(torch.norm(f[..., :2], dim=2) > 4 * torch.abs(f[..., 2]), dim=1)
    return (hit & (terrain_levels(env) > 3) & type_mask(env, "gap", "climb")).float()


# =============================================================== terminations
def terminate_base_contact(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > threshold, dim=1)


def terminate_velocity_violation(env, threshold: float = 1.5):
    v = _asset(env).data.root_lin_vel_b[:, 0]
    cmd = env.command_manager.get_command("base_velocity")[:, 0]
    err = v - cmd
    violate = ((err > threshold) & (cmd < 0.0)) | ((err < -threshold) & (cmd > 0.0))
    return violate & (terrain_levels(env) > 3)


def terminate_fall(env):
    a = _asset(env)
    return (a.data.root_lin_vel_w[:, 2] < -3.0) | (a.data.projected_gravity_b[:, 2] > 0.0)


def time_out(env):
    return env.episode_length_buf > env.max_episode_length


# ================================================================= curriculum
def wmp_terrain_levels(env, env_ids: Sequence[int]):
    """legged_gym ``_update_terrain_curriculum`` (skipped on the initial reset)."""
    terrain = env.scene.terrain
    if env.common_step_counter == 0:
        return torch.mean(terrain.terrain_levels.float())
    asset = _asset(env)
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > env.wmp_env_length / 2
    cmd = env.command_manager.get_command("base_velocity")[env_ids, :2]
    move_down = (distance < torch.norm(cmd, dim=1) * env.max_episode_length_s * 0.5) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


# ===================================================================== events
def reset_root_state_wmp(env, env_ids: torch.Tensor, xy_range: float = 1.0, vel_range: float = 0.5,
                         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Spawn within +-1 m of the terrain origin; tilt/gap envs keep y = origin."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = asset.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    rnd = math_utils.sample_uniform(-xy_range, xy_range, (len(env_ids), 2), device=env.device)
    narrow = type_mask(env, "tilt", "gap")[env_ids]
    rnd[:, 1] = torch.where(narrow, torch.zeros_like(rnd[:, 1]), rnd[:, 1])
    state[:, :2] += rnd
    state[:, 7:13] = math_utils.sample_uniform(-vel_range, vel_range, (len(env_ids), 6), device=env.device)
    asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(state[:, 7:13], env_ids=env_ids)


def randomize_camera_pitch(env, env_ids, sensor_cfg: SceneEntityCfg, pitch_range_deg: tuple[float, float]):
    """Per-env camera pitch noise (WMP ``depth.y_angle``; positive pitches down)."""
    camera: RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    pitch = torch.deg2rad(math_utils.sample_uniform(*pitch_range_deg, (len(ids),), device=env.device))
    zeros = torch.zeros_like(pitch)
    # Offset quaternion is stored in the "world" camera convention (x forward, z up):
    # a positive rotation about +y pitches the optical axis down.
    noise = math_utils.quat_from_euler_xyz(zeros, pitch, zeros)
    camera._offset_quat[ids] = math_utils.quat_mul(camera._offset_quat[ids], noise)
