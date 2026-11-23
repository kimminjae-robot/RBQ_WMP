"""AMP expert transitions from JSON motion clips (WMP/AMP_for_hardware format).

Re-implementation of ``rsl_rl/datasets/motion_loader.py`` of bytedance/WMP for the
code path WMP actually uses (``preload_transitions=True``): sample clip and time,
linearly interpolate the frame at ``t`` and ``t + dt`` and return AMP states
``[joint_pos(12), base_lin_vel(3), base_ang_vel(3), joint_vel(12)]``.
The original depends on ``pybullet_utils`` and ``np.int`` (removed in NumPy 1.24)
only for root-pose blending, which the preloaded AMP states do not use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..retarget.a1_to_rbq10 import I_AVEL, I_FPOS, I_JPOS, I_JVEL, I_LVEL, JPOS, JVEL

AMP_OBS_DIM = 30


def frames_to_amp(frames: np.ndarray) -> np.ndarray:
    """Frame rows -> AMP observation ``[q, v_lin, w, qd]``."""
    return np.concatenate([
        frames[:, I_JPOS:I_JPOS + JPOS],
        frames[:, I_LVEL:I_LVEL + 3],
        frames[:, I_AVEL:I_AVEL + 3],
        frames[:, I_JVEL:I_JVEL + JVEL],
    ], axis=1)


class AMPLoader:
    def __init__(self, device, time_between_frames, motion_files, preload_transitions=True,
                 num_preload_transitions=2_000_000, seed=None):
        if not motion_files:
            raise FileNotFoundError("AMP needs reference motion files (run tools/retarget_a1_mocap.py)")
        self.device = device
        self.time_between_frames = float(time_between_frames)
        rng = np.random.default_rng(seed)
        self.trajectories, self.weights, self.frame_durations, self.lengths = [], [], [], []
        self.names = []
        for path in sorted(str(p) for p in motion_files):
            data = json.loads(Path(path).read_text())
            frames = np.asarray(data["Frames"], dtype=np.float32)
            self.trajectories.append(frames_to_amp(frames))
            self.weights.append(float(data.get("MotionWeight", 1.0)))
            self.frame_durations.append(float(data["FrameDuration"]))
            self.lengths.append((len(frames) - 1) * float(data["FrameDuration"]))
            self.names.append(Path(path).stem)
            print(f"[AMP] loaded {self.lengths[-1]:.2f}s motion from {path}")
        self.weights = np.asarray(self.weights) / np.sum(self.weights)
        self.frame_durations = np.asarray(self.frame_durations)
        self.lengths = np.asarray(self.lengths)
        if not preload_transitions:
            raise NotImplementedError("WMP uses preloaded AMP transitions only")
        idx = rng.choice(len(self.trajectories), size=num_preload_transitions, p=self.weights)
        # Same time sampling as the original: uniform in [0, len - dt - frame_dt].
        subst = self.time_between_frames + self.frame_durations[idx]
        times = np.maximum(0.0, self.lengths[idx] * rng.uniform(size=len(idx)) - subst)
        self.preloaded_s = self._frames_at(idx, times)
        self.preloaded_s_next = self._frames_at(idx, times + self.time_between_frames)

    def _frames_at(self, idx, times):
        out = np.zeros((len(idx), AMP_OBS_DIM), dtype=np.float32)
        for k in np.unique(idx):
            mask = idx == k
            traj = self.trajectories[k]
            n = len(traj)
            pos = np.clip(times[mask] / self.frame_durations[k], 0, n - 1)
            lo = np.floor(pos).astype(np.int64)
            hi = np.minimum(lo + 1, n - 1)
            blend = (pos - lo)[:, None].astype(np.float32)
            out[mask] = (1.0 - blend) * traj[lo] + blend * traj[hi]
        return torch.as_tensor(out, device=self.device)

    @property
    def observation_dim(self) -> int:
        return AMP_OBS_DIM

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            idxs = torch.randint(0, self.preloaded_s.shape[0], (mini_batch_size,), device=self.preloaded_s.device)
            yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
