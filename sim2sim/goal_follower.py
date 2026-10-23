"""Terrain waypoint commands following the original ParkourEvent update semantics.

This supplies the current/next heading observations, not joint actions or a Teacher.
Goals come from the unchanged upstream terrain generator. The timer, 0.2 m
threshold, strict >0.1 s delay and one-update target lag match ParkourEvent.
"""
import numpy as np


class TerrainGoalFollower:
    def __init__(self, terrain, origin=None, policy_dt=.02):
        self.goals = np.asarray(terrain['goals'], dtype=float) + np.asarray(terrain['origin_xy'] if origin is None else origin)
        self.index = 0
        self.timer = 0
        self.policy_dt = policy_dt
        self.threshold = .2
        self.delay = .1
        self.current = self.goals[0].copy()
        self.next = self.goals[1].copy()
        self.heading_errors = np.zeros(2)
        self.steps = 0

    @property
    def complete(self):
        return self.index >= len(self.goals)

    def update(self, position_xy, yaw):
        if self.complete:
            return self.heading_errors.copy()
        if self.timer > self.delay / self.policy_dt:
            self.index += 1
            self.timer = 0
        if np.linalg.norm(np.asarray(position_xy) - self.current) < self.threshold:
            self.timer += 1
        # Original event computes headings against the previous gathered targets,
        # then gathers the new targets. Observation holds them for five steps.
        targets = np.stack((self.current, self.next)) - position_xy
        angles = np.arctan2(targets[:, 1], targets[:, 0]) - yaw
        if self.steps % 5 == 0:
            self.heading_errors = angles
        self.current = self.goals[min(self.index, len(self.goals)-1)].copy()
        self.next = self.goals[min(self.index+1, len(self.goals)-1)].copy()
        self.steps += 1
        return self.heading_errors.copy()
