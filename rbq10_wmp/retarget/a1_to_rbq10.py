"""Retarget the WMP/AMP A1 dog mocap clips to the RBQ10 leg geometry.

The WMP authors train the AMP style reward (paper Eq. 10-11) with A1 reference
clips (``datasets/mocap_motions`` of bytedance/WMP, originally from
AMP_for_hardware).  RBQ10 has the same leg topology (hip roll about x, thigh
pitch and knee pitch about y) but different link lengths, so copying joint
angles would produce RBQ10 foot paths with the wrong stance width/height.

Retargeting per frame and per leg:

1. Foot positions in the base frame are taken from the clip and checked against
   A1 forward kinematics of the recorded joint angles.
2. The foot offset relative to the A1 hip is scaled with *dynamic similarity*:
   lengths by ``s = L_rbq / L_a1`` (leg length ratio); the lateral hip/thigh
   offset is replaced by the RBQ10 offset rather than scaled.
3. RBQ10 joint angles are found with closed-form 3-DoF leg IK.
4. Time is scaled by ``sqrt(s)`` (Froude similarity), so root linear velocity is
   scaled by ``sqrt(s)``, angular velocity by ``1/sqrt(s)`` and the frame duration
   by ``sqrt(s)``.  Joint/foot velocities are recomputed by finite differences of
   the retargeted trajectories.

The output uses the same JSON layout as the source clips so the original
``AMPLoader`` indexing keeps working; joints are written in RBQ10 URDF order
``[HR, HL, FR, FL] x [R, P, K]`` (see ``RBQ10_LEG_ORDER``).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

# Frame layout of the WMP/AMP motion files.
POS, ROT, JPOS, FPOS, LVEL, AVEL, JVEL, FVEL = 3, 4, 12, 12, 3, 3, 12, 12
I_POS = 0
I_ROT = I_POS + POS
I_JPOS = I_ROT + ROT
I_FPOS = I_JPOS + JPOS
I_LVEL = I_FPOS + FPOS
I_AVEL = I_LVEL + LVEL
I_JVEL = I_AVEL + AVEL
I_FVEL = I_JVEL + JVEL
FRAME_SIZE = I_FVEL + FVEL  # 61


@dataclass(frozen=True)
class LegGeometry:
    """Single-leg geometry; x forward, y left, z up (base frame)."""

    hip_x: float  # |x| of hip roll joint
    hip_y: float  # |y| of hip roll joint
    lateral: float  # |y| offset from roll joint to thigh pitch joint
    thigh: float
    calf: float
    foot_radius: float


A1 = LegGeometry(hip_x=0.183, hip_y=0.047, lateral=0.08505, thigh=0.2, calf=0.2, foot_radius=0.02)
# assets/rbq10/resources/urdf/rbq10.urdf: joint*_R origin (+-0.31218, +-0.09),
# thigh origin y +-0.10285, knee z -0.33, foot z -0.33.
RBQ10 = LegGeometry(hip_x=0.31218, hip_y=0.09, lateral=0.10285, thigh=0.33, calf=0.33, foot_radius=0.03)

# The A1 clips store foot positions shifted by the A1 trunk inertial origin
# (a1.urdf trunk <inertial> xyz); measured residual of A1 FK is exactly this
# constant, so it is removed before retargeting. RBQ10 output feet are written
# in the RBQ10 trunk (link) frame.
A1_FOOT_FRAME_OFFSET = np.array([0.012731, 0.002186, 0.000515])

# A1 clip leg order (Isaac Gym A1 order) and the sign of x / y of each hip.
A1_LEG_ORDER = ("FL", "FR", "RL", "RR")
RBQ10_LEG_ORDER = ("HR", "HL", "FR", "FL")  # RBQ10 URDF joint order
LEG_SIGNS = {  # (x sign, y sign)
    "FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1),
    "HL": (-1, 1), "HR": (-1, -1),
}
# RBQ10 leg name -> A1 source leg.
RBQ10_FROM_A1 = {"HR": "RR", "HL": "RL", "FR": "FR", "FL": "FL"}


def leg_fk(q: np.ndarray, geo: LegGeometry, x_sign: int, y_sign: int) -> np.ndarray:
    """Foot position in base frame. ``q``: (..., 3) roll, pitch, knee."""
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    # Position in the hip-roll frame before applying roll.
    l = y_sign * geo.lateral
    px = -geo.thigh * np.sin(q2) - geo.calf * np.sin(q2 + q3)
    pz = -geo.thigh * np.cos(q2) - geo.calf * np.cos(q2 + q3)
    py = l
    # Roll about x.
    c, s = np.cos(q1), np.sin(q1)
    y = c * py - s * pz
    z = s * py + c * pz
    return np.stack([px + x_sign * geo.hip_x, y + y_sign * geo.hip_y, z], axis=-1)


def leg_ik(p: np.ndarray, geo: LegGeometry, x_sign: int, y_sign: int) -> np.ndarray:
    """Closed-form inverse of :func:`leg_fk` with knee-backward (q3 < 0) branch."""
    d = p - np.array([x_sign * geo.hip_x, y_sign * geo.hip_y, 0.0])
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    l = y_sign * geo.lateral
    # Roll: [y, z] = R(q1) [l, -h] with h > 0 the sagittal-plane leg height.
    r2 = np.maximum(y * y + z * z - l * l, 1e-9)
    h = np.sqrt(r2)
    q1 = np.arctan2(z * l + y * h, y * l - z * h)
    # Sagittal 2-link: px = -L1 sin q2 - L2 sin(q2+q3), pz' = -h.
    L1, L2 = geo.thigh, geo.calf
    D2 = x * x + h * h
    cos_k = np.clip((D2 - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    q3 = -np.arccos(cos_k)
    # Solve q2: foot = -(L1 + L2 cos q3)[sin q2, cos q2] - L2 sin q3 [cos q2, -sin q2]
    a = L1 + L2 * np.cos(q3)
    b = L2 * np.sin(q3)
    # x = -a sin q2 - b cos q2 ; h = a cos q2 - b sin q2
    q2 = np.arctan2(-x * a - h * b, h * a - x * b)
    return np.stack([q1, q2, q3], axis=-1)


def _legs(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], 4, 3)


def retarget_frames(frames: np.ndarray, frame_duration: float, scale: float | None = None):
    """Return (retargeted frames, new frame duration, diagnostics)."""
    if frames.shape[1] != FRAME_SIZE:
        raise ValueError(f"Expected {FRAME_SIZE} values per frame, got {frames.shape[1]}")
    s = scale if scale is not None else (RBQ10.thigh + RBQ10.calf) / (A1.thigh + A1.calf)
    t = np.sqrt(s)
    a1_q = _legs(frames[:, I_JPOS:I_JPOS + JPOS])
    a1_foot = _legs(frames[:, I_FPOS:I_FPOS + FPOS]) - A1_FOOT_FRAME_OFFSET

    fk_error, fk_mean = [], []
    out_q = np.zeros((len(frames), 4, 3))
    out_foot = np.zeros((len(frames), 4, 3))
    for k, leg in enumerate(RBQ10_LEG_ORDER):
        src = RBQ10_FROM_A1[leg]
        j = A1_LEG_ORDER.index(src)
        sx, sy = LEG_SIGNS[src]
        rx, ry = LEG_SIGNS[leg]
        assert (sx, sy) == (rx, ry)
        fk = leg_fk(a1_q[:, j], A1, sx, sy)
        fk_error.append(np.abs(fk - a1_foot[:, j]).max())
        fk_mean.append(np.linalg.norm(fk - a1_foot[:, j], axis=-1).mean())
        # Offset from the A1 hip roll axis; remove/replace the lateral offset.
        rel = a1_foot[:, j] - np.array([sx * A1.hip_x, sy * A1.hip_y, 0.0])
        rel_y = rel[:, 1] - sy * A1.lateral
        target = np.stack([
            s * rel[:, 0] + rx * RBQ10.hip_x,
            s * rel_y + ry * (RBQ10.lateral + RBQ10.hip_y),
            s * rel[:, 2],
        ], axis=-1)
        q = leg_ik(target, RBQ10, rx, ry)
        out_q[:, k] = q
        out_foot[:, k] = leg_fk(q, RBQ10, rx, ry)

    new_dt = frame_duration * t
    out = np.zeros_like(frames)
    out[:, I_POS:I_POS + POS] = frames[:, I_POS:I_POS + POS] * s
    # Keep the feet on the ground: scale the height above the foot-sphere centre.
    out[:, I_POS + 2] = s * (frames[:, I_POS + 2] - A1.foot_radius) + RBQ10.foot_radius
    out[:, I_ROT:I_ROT + ROT] = frames[:, I_ROT:I_ROT + ROT]
    out[:, I_JPOS:I_JPOS + JPOS] = out_q.reshape(len(frames), -1)
    out[:, I_FPOS:I_FPOS + FPOS] = out_foot.reshape(len(frames), -1)
    out[:, I_LVEL:I_LVEL + LVEL] = frames[:, I_LVEL:I_LVEL + LVEL] * t
    out[:, I_AVEL:I_AVEL + AVEL] = frames[:, I_AVEL:I_AVEL + AVEL] / t
    out[:, I_JVEL:I_JVEL + JVEL] = np.gradient(out_q.reshape(len(frames), -1), new_dt, axis=0)
    out[:, I_FVEL:I_FVEL + FVEL] = np.gradient(out_foot.reshape(len(frames), -1), new_dt, axis=0)
    diag = dict(
        scale=float(s), time_scale=float(t), frame_duration=float(new_dt),
        a1_fk_max_error_m=float(max(fk_error)),
        a1_fk_mean_error_m=float(np.mean(fk_mean)),
    )
    return out, new_dt, diag


def check_ik_roundtrip(num: int = 20000, seed: int = 0) -> float:
    """Max foot error of IK(FK(q)) over random reachable RBQ10 configurations."""
    rng = np.random.default_rng(seed)
    q = np.stack([rng.uniform(-0.5, 0.5, num), rng.uniform(-0.5, 1.8, num), rng.uniform(-2.7, -0.45, num)], -1)
    worst = 0.0
    for leg in RBQ10_LEG_ORDER:
        sx, sy = LEG_SIGNS[leg]
        p = leg_fk(q, RBQ10, sx, sy)
        q2 = leg_ik(p, RBQ10, sx, sy)
        worst = max(worst, float(np.abs(leg_fk(q2, RBQ10, sx, sy) - p).max()))
    return worst


def retarget_file(src: Path, dst: Path, scale: float | None = None) -> dict:
    data = json.loads(Path(src).read_text())
    frames = np.asarray(data["Frames"], dtype=np.float64)
    out, dt, diag = retarget_frames(frames, float(data["FrameDuration"]), scale)
    result = dict(data)
    result["FrameDuration"] = dt
    result["Frames"] = out.round(7).tolist()
    result["Retarget"] = {
        "source": str(Path(src).name), "robot": "Rainbow_RBQ10",
        "joint_order": [f"{leg}{j}" for leg in RBQ10_LEG_ORDER for j in ("R", "P", "K")],
        "method": "foot-space scaling (Froude similarity) + closed-form RBQ10 leg IK",
        **diag,
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(result))
    return result["Retarget"] | {"num_frames": len(out)}
