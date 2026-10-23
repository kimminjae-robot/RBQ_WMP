"""Read-only episode measurements; does not change rewards, resets, or curriculum."""
from collections import Counter, deque
import numpy as np
import torch
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from .contract import FOOT_NAMES


class TeacherMetrics:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene['robot']
        self.contact = env.scene['contact_forces']
        self.foot_ids, _ = self.contact.find_bodies(FOOT_NAMES, preserve_order=True)
        self.parkour = env.parkour_manager.get_term('base_parkour')
        self.in_step = False
        # The runner randomizes initial episode lengths. Exclude each initial
        # partial episode; measure only episodes observed from a real reset.
        self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.values = torch.zeros(env.num_envs, 7, device=env.device)
        self.air_ticks = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.names = sorted(set(self.parkour.total_terrain_names.flatten()))
        self.windows = {name: deque(maxlen=1000) for name in self.names}
        self.counts = Counter()
        self.scalars = {}
        self.refresh()

    def begin(self):
        self.in_step = True
        self.values[:, 0] += 1  # Actual observed control steps, not randomized episode_length_buf.

    def observe(self, ids):
        if ids.numel() == 0:
            return
        state = self.robot.data.root_state_w[ids]
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite RBQ10 physical state')
        forces = self.contact.data.net_forces_w[ids][:, self.foot_ids].norm(dim=-1)
        contact = (forces > 2.0).any(-1)
        self.values[ids, 1] = torch.maximum(self.values[ids, 1], forces.max(-1).values)
        torque_ratio = (self.robot.data.applied_torque[ids].abs() /
                        self.robot.data.joint_effort_limits[ids]).max(-1).values
        self.values[ids, 2] = torch.maximum(self.values[ids, 2], torque_ratio)
        self.values[ids, 3] += (contact & (self.air_ticks[ids] >= 2)).float()
        self.air_ticks[ids] = torch.where(contact, 0, self.air_ticks[ids] + 1)
        limits = self.robot.data.joint_pos_limits[ids]
        position = self.robot.data.joint_pos[ids]
        self.values[ids, 4] += ((position < limits[..., 0] - 0.02) |
                               (position > limits[..., 1] + 0.02)).any(-1).float()
        velocity_ratio = (self.robot.data.joint_vel[ids].abs() /
                          self.robot.data.joint_vel_limits[ids]).max(-1).values
        self.values[ids, 5] = torch.maximum(self.values[ids, 5], velocity_ratio)
        self.values[ids, 6] += self.env.reward_buf[ids]

    def finish(self, ids):
        if not self.in_step or ids.numel() == 0:
            return
        self.observe(ids)  # Capture terminal state before original auto-reset.
        robot = self.robot
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w[ids])
        fallen = ((wrap_to_pi(roll).abs() > 1.5) | (wrap_to_pi(pitch).abs() > 1.5) |
                  (robot.data.root_pos_w[ids, 2] < -0.25))
        goal = self.parkour.cur_goal_idx[ids] >= self.parkour.num_goals
        timeout = self.env.episode_length_buf[ids] >= self.env.max_episode_length
        rows = torch.cat((self.values[ids], self.ready[ids, None].float(),
                          fallen[:, None].float(), (goal & ~fallen)[:, None].float(),
                          (timeout & ~fallen & ~goal)[:, None].float(),
                          self.parkour.cur_goal_idx[ids, None].float(),
                          self.env.scene.terrain.terrain_levels[ids, None].float()), dim=-1).cpu().tolist()
        terrain_names = self.parkour.env_per_terrain_name[ids.cpu().numpy()].reshape(-1)
        for name, row in zip(terrain_names, rows):
            if row[7]:
                self.windows[name].append(row)
                self.counts[name] += 1
        self.values[ids] = 0
        self.air_ticks[ids] = 0
        self.ready[ids] = True

    def end(self):
        continuing = (~self.env.reset_buf).nonzero(as_tuple=False).squeeze(-1)
        self.observe(continuing)
        self.in_step = False
        if self.env.common_step_counter % 24 == 0:
            self.refresh()
        return self.scalars

    def refresh(self):
        scalars = {}
        for name in self.names:
            rows = np.asarray(self.windows[name], dtype=float)
            prefix = f'Parkour/{name}/'
            scalars[prefix + 'completed_episodes'] = self.counts[name]
            scalars[prefix + 'window_episodes'] = len(rows)
            for key, index, operation in (
                ('course_success_rate', 9, np.mean), ('fall_rate', 8, np.mean),
                ('timeout_rate', 10, np.mean), ('episode_steps', 0, np.mean),
                ('episode_reward', 6, np.mean), ('goal_index', 11, np.mean),
                ('terrain_level', 12, np.mean), ('peak_foot_force_n', 1, np.max),
                ('peak_estimated_effort_ratio', 2, np.max), ('landing_events', 3, np.mean),
                ('joint_limit_violation_steps', 4, np.mean), ('peak_velocity_ratio', 5, np.max),
            ):
                scalars[prefix + key] = float(operation(rows[:, index])) if len(rows) else 0.0
        self.scalars = scalars
