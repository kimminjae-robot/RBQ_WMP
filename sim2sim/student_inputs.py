"""Explicit command, measured-velocity, course-label and ray-depth interfaces."""
import json
import os
from pathlib import Path
import struct

import mujoco
import numpy as np
import torch
import torch.nn.functional as F


def joystick_axis(value, deadzone=.08):
    value = float(np.clip(value / 32767., -1, 1))
    return 0. if abs(value) < deadzone else float(np.sign(value)*(abs(value)-deadzone)/(1-deadzone))


class CommandInput:
    """JSON can supply joystick commands and independent measured body velocity."""
    def __init__(self, vx=.6, yaw_rate=0., path=None, joystick=None):
        self.vx, self.yaw_rate, self.path = vx, yaw_rate, Path(path) if path else None
        self.axes = {0: 0., 1: 0.}
        self.fd = os.open(joystick, os.O_RDONLY | os.O_NONBLOCK) if joystick else None

    def read(self):
        measured = None
        if self.fd is not None:
            while True:
                try:
                    event = os.read(self.fd, 8)
                except BlockingIOError:
                    break
                if len(event) != 8:
                    raise RuntimeError('Joystick disconnected')
                _, value, event_type, number = struct.unpack('IhBB', event)
                if event_type & 2:
                    self.axes[number] = joystick_axis(value)
            vx, yaw_rate = -.8*self.axes[1], -1.0*self.axes[0]
        else:
            vx, yaw_rate = self.vx, self.yaw_rate
        if self.path:
            # Writer should use an atomic rename when replacing the file.
            data = json.loads(self.path.read_text())
            command = data.get('command', {})
            vx, yaw_rate = float(command.get('vx', vx)), float(command.get('yaw_rate', yaw_rate))
            if float(command.get('vy', 0.)) != 0:
                raise ValueError('Original policy has no lateral velocity command slot; vy must be 0')
            if 'measured_velocity_body_m_s' in data:
                measured = np.asarray(data['measured_velocity_body_m_s'], dtype=np.float32)
                if measured.shape != (3,) or not np.isfinite(measured).all():
                    raise ValueError('Measured velocity must be a finite body-frame [vx,vy,vz] vector in m/s')
        if not np.isfinite([vx, yaw_rate]).all() or abs(vx) > .8 or abs(yaw_rate) > 1.:
            raise ValueError('Command limits: |vx|<=0.8 m/s, |yaw_rate|<=1 rad/s')
        return vx, yaw_rate, measured

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class DepthInput:
    """Terrain-only ray distance, matching the existing Isaac camera preprocessing."""
    def __init__(self, model, data, config):
        self.m, self.d, self.cfg = model, data, config
        self.body = model.body(config['parent']).id
        w, h = config['width'], config['height']
        fx = w*config['focal_length']/config['horizontal_aperture']
        fy = h*config['focal_length']/config['vertical_aperture']
        u, v = np.meshgrid(np.arange(w)+.5, np.arange(h)+.5, indexing='xy')
        rays = np.stack(((u-w/2)/fx, (v-h/2)/fy, np.ones_like(u)), axis=-1).reshape(-1, 3)
        self.rays = rays/np.linalg.norm(rays, axis=1, keepdims=True)
        rotation = np.zeros(9)
        q = np.asarray(config['quaternion_wxyz_ros'], dtype=np.float64)
        if not np.isclose(np.linalg.norm(q), 1., atol=1e-5):
            raise ValueError('Camera quaternion must be unit length')
        mujoco.mju_quat2Mat(rotation, q)
        self.rotation = rotation.reshape(3, 3)
        self.buffer = None
        self.latest_raw = None

    def sample(self):
        R = self.d.xmat[self.body].reshape(3, 3)
        origin = self.d.xpos[self.body] + R @ np.asarray(self.cfg['position_m'])
        directions = self.rays @ (R @ self.rotation).T
        distance = np.full(len(directions), self.cfg['max_distance_m'], dtype=np.float32)
        group = np.array([1,0,0,0,0,0], dtype=np.uint8)
        hit = np.array([-1], dtype=np.int32)
        for i, ray in enumerate(directions):
            length = mujoco.mj_ray(self.m, self.d, origin, ray, group, 1, -1, hit)
            if length >= 0:
                distance[i] = min(length, self.cfg['max_distance_m'])
        self.latest_raw = distance.reshape(self.cfg['height'], self.cfg['width'])
        crop = torch.from_numpy(self.latest_raw[:-2, 4:-4].copy())[None,None]
        processed = F.interpolate(crop, size=(58,87), mode='bicubic', align_corners=False, antialias=True)[0,0]
        processed = processed / self.cfg['max_distance_m'] - .5
        if self.buffer is None:
            self.buffer = [processed.clone(), processed.clone()]
        else:
            self.buffer = [self.buffer[-1], processed]
        return self.buffer[-2][None]


def student_observation(observation, reference_vx, yaw_rate, measured_velocity, config):
    x = observation.copy()
    x[8:11] = [0., 0., reference_vx]
    x[11:13] = [0.,1.] if config['terrain_label']=='flat' else [1.,0.]
    # Both height scan and physics-parameter truth are absent from Student deployment.
    x[53:223] = 0
    x[185:188] = np.asarray(measured_velocity)*2.
    if config['direction_source']=='joystick':
        error = np.clip(yaw_rate*config['yaw_rate_to_error_seconds'], -1.5, 1.5)
        x[6:8] = [error, error]
    else:
        x[6:8] = 0
    return x
