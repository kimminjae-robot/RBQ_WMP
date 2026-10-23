"""CPU MuJoCo rollout of an exported RBQ10 Teacher. No Isaac Sim dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from build_scene import ASSETS, ROOT, build_scene


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class ObservationBridge:
    def __init__(self, model, data, contract, home, flat, speed, scan_mode='source'):
        self.m, self.d = model, data
        self.joints = np.array([model.joint(n).id for n in contract['joint_order']])
        self.qids = model.jnt_qposadr[self.joints]
        self.vids = model.jnt_dofadr[self.joints]
        self.feet = [model.body(n).id for n in contract['foot_order']]
        self.base = model.body('trunk').id
        self.home, self.flat, self.speed, self.scan_mode = home, flat, speed, scan_mode
        self.history = np.zeros((10, 53), dtype=np.float32)
        self.previous_contact = np.zeros(4, dtype=bool)
        self.scan = np.zeros(132, dtype=np.float32)
        self.yaws = np.zeros(2)
        self.step = 0
        self.goal_idx = 0
        self.goals = np.column_stack((np.arange(1, 25, 1.5), np.zeros(16)))
        x, y = np.meshgrid(np.arange(12)*.15-.825, np.arange(11)*.15-.75, indexing='xy')
        self.rays = np.column_stack((x.ravel(), y.ravel(), np.zeros(132)))
        self.current_contacts = np.zeros(4, dtype=bool)
        self.scan_min, self.scan_max = 0., 0.

    def sample(self, previous_action):
        d, m = self.d, self.m
        R = d.xmat[self.base].reshape(3, 3)
        pos = d.xpos[self.base]
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(np.clip(-R[2, 0], -1, 1))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 1)
        if np.linalg.norm(pos[:2] - self.goals[self.goal_idx]) < .3:
            self.goal_idx = min(self.goal_idx+1, len(self.goals)-2)
        if self.step % 5 == 0:
            delta = self.goals[self.goal_idx:self.goal_idx+2] - pos[:2]
            # Match the source: target heading is wrapped, difference itself is not wrapped.
            self.yaws = np.arctan2(delta[:, 1], delta[:, 0]) - yaw
            Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                           [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            # Isaac's RayCaster combines the full body transform and mount offset,
            # then applies yaw-only rotation to the ray grid.
            origin = pos + R @ np.array([.42, 0., 20.])
            origins = self.rays @ Ry.T + origin
            group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
            for i, start in enumerate(origins):
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(m, d, start, np.array([0., 0., -1.]), group, 1, -1, hit)
                height = start[2] - distance if distance >= 0 else -np.inf
                ref = origin[2] if self.scan_mode == 'source' else pos[2]
                self.scan[i] = np.clip(ref-height-.3, -1, 1)
            self.scan_min, self.scan_max = float(self.scan.min()), float(self.scan.max())
        forces = np.zeros((4, 3))
        for ci in range(d.ncon):
            con = d.contact[ci]
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, force)
            world_force = con.frame.reshape(3, 3).T @ force[:3]
            for side, sign in ((0, -1), (1, 1)):
                body = m.geom_bodyid[con.geom[side]]
                if body in self.feet:
                    forces[self.feet.index(body)] += sign * world_force
        contact = np.linalg.norm(forces, axis=1) > 2.
        self.current_contacts = contact
        filtered = contact | self.previous_contact
        self.previous_contact = contact.copy()
        prop = np.concatenate((velocity[:3]*.25, [roll, pitch], [0., *self.yaws],
                               [0., 0., self.speed], [not self.flat, self.flat],
                               d.qpos[self.qids]-self.home, d.qvel[self.vids]*.05,
                               previous_action, filtered.astype(float)-.5)).astype(np.float32)
        assert prop.shape == (53,)
        # Export embeds estimator and history encoder: explicit/latent truth slots are ignored.
        obs = np.concatenate((prop, self.scan, np.zeros(38), self.history.ravel())).astype(np.float32)
        prop[6:8] = 0
        if self.step <= 1:
            self.history[:] = prop
        else:
            self.history = np.concatenate((self.history[1:], prop[None]))
        self.step += 1
        return np.clip(obs, -100, 100), (roll, pitch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir', type=Path, default=ROOT/'outputs/sim2sim/teacher')
    p.add_argument('--course', choices=['flat', 'hurdle', 'step'], default='flat')
    p.add_argument('--height', type=float, default=.10)
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--speed', type=float, default=.6)
    p.add_argument('--action-delay', type=int, choices=[0, 1], default=0,
                   help='Teacher uses 0; source Student uses 1 policy step')
    p.add_argument('--scan-mode', choices=['source', 'base-height'], default='source',
                   help='base-height is a diagnostic distribution change; source matches existing training code')
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/sim2sim/rollout.json')
    args = p.parse_args()
    if args.seconds <= 0 or args.height <= 0 or not 0 <= args.speed <= .8:
        p.error('Require seconds>0, height>0 and 0<=speed<=0.8')
    torch.set_num_threads(1)
    meta = json.loads((args.policy_dir/'manifest.json').read_text())
    if meta['format'] != 'rbq10_mujoco_teacher_v1' or meta['role'] != 'teacher':
        raise ValueError('Expected Teacher export format')
    policy_file = args.policy_dir/'teacher.pt'
    assert hashlib.sha256(policy_file.read_bytes()).hexdigest() == meta['model_sha256']
    c = meta['contract']
    urdf_hash = hashlib.sha256((ASSETS/'resources/urdf/rbq10.urdf').read_bytes()).hexdigest()
    assert urdf_hash == c['urdf_sha256'], 'Robot URDF mismatch'
    assert c['policy_dt'] == .02 and c['physics_dt'] == .005 and c['action_dim'] == 12
    home_dict = json.loads((ASSETS/'official/info.json').read_text())['config_info']['init_state']['default_joint_angles']
    home = np.array([home_dict[n] for n in c['joint_order']])
    model = mujoco.MjModel.from_xml_string(build_scene(c, args.course, args.height))
    data = mujoco.MjData(model)
    bridge = ObservationBridge(model, data, c, home, args.course=='flat', args.speed, args.scan_mode)
    data.qpos[bridge.qids] = home
    mujoco.mj_forward(model, data)
    policy = torch.jit.load(str(policy_file), map_location='cpu').eval()
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
    actions = np.zeros(12, dtype=np.float32)
    records, max_torque, max_speed = [], 0., 0.
    termination = 'duration'
    start = data.xpos[bridge.base].copy()
    try:
        for step in range(int(args.seconds/.02)):
            wall = time.monotonic()
            obs, angles = bridge.sample(actions)
            with torch.inference_mode():
                new_actions = policy(torch.from_numpy(obs[None])).numpy()[0]
            if not np.isfinite(new_actions).all():
                raise FloatingPointError('Non-finite policy action')
            applied = actions if args.action_delay else new_actions
            data.ctrl[:] = home + .25*applied
            actions = new_actions.copy()
            for _ in range(4):
                mujoco.mj_step(model, data)
                max_torque = max(max_torque, float(np.max(np.abs(data.actuator_force))))
                max_speed = max(max_speed, float(np.max(np.abs(data.qvel[bridge.vids]))))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Non-finite simulator state')
            records.append({'time': float(data.time), 'base_xyz': data.xpos[bridge.base].tolist(),
                            'roll_pitch': list(map(float, angles)), 'scan_range': [bridge.scan_min, bridge.scan_max],
                            'max_abs_action': float(np.max(np.abs(actions))),
                            'foot_contacts': bridge.current_contacts.tolist()})
            # This bridge uses an explicit early fall condition; not original course-success scoring.
            if max(map(abs, angles)) > 1.5 or data.xpos[bridge.base, 2] < .15:
                termination = 'fall'
                break
            if viewer:
                if not viewer.is_running():
                    termination = 'viewer_closed'
                    break
                viewer.sync()
                time.sleep(max(0, .02-(time.monotonic()-wall)))
    finally:
        if viewer:
            viewer.close()
    result = {'checkpoint': meta['checkpoint'], 'checkpoint_sha256': meta['checkpoint_sha256'],
              'engine': mujoco.__version__, 'course': args.course, 'height': args.height,
              'scan_mode': args.scan_mode, 'action_delay_steps': args.action_delay,
              'physics_dt': .005, 'policy_dt': .02, 'simulated_seconds': float(data.time),
              'termination': termination, 'displacement_xyz': (data.xpos[bridge.base]-start).tolist(),
              'max_abs_torque_nm': max_torque, 'max_abs_joint_speed_rad_s': max_speed,
              'mass_kg': float(model.body_mass.sum()), 'joint_order': c['joint_order'],
              'limitation': 'Independent simple courses; no original terrain/curriculum success-rate equivalence. MuJoCo PD/contact solver differs; 50 rad/s velocity cap is monitored, not imposed.',
              'records': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()
