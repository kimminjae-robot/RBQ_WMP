"""CPU checks of the RBQ10 WMP port (no Isaac Sim): python3 -m unittest tests.test_wmp_cpu"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rbq10_wmp import scaling  # noqa: E402
from rbq10_wmp.agents import train_cfg  # noqa: E402
from rbq10_wmp.env.terrain import TERRAIN_NAMES, WMPTerrain, WMPTerrainParams  # noqa: E402
from rbq10_wmp.inference import WMPPolicy  # noqa: E402
from rbq10_wmp.mock_env import MockWMPEnv  # noqa: E402
from rbq10_wmp.retarget.a1_to_rbq10 import RBQ10, leg_fk, leg_ik  # noqa: E402
from rbq10_wmp.runner import WMPRunner  # noqa: E402


def small_cfg(num_envs):
    cfg = train_cfg(num_steps_per_env=10, save_interval=100, amp_num_preload_transitions=5000,
                    wm_dataset_envs=num_envs)
    cfg["algorithm"]["amp_replay_buffer_size"] = 5000
    cfg["world_model"] = dict(batch_size=2, batch_length=8, train_start_steps=20, train_steps_per_iter=1)
    return cfg


class RetargetTest(unittest.TestCase):
    def test_ik_inverts_fk(self):
        rng = np.random.default_rng(0)
        q = np.stack([rng.uniform(-0.3, 0.3, 64), rng.uniform(0.3, 1.2, 64), rng.uniform(-2.2, -0.8, 64)], 1)
        for side in (1, -1):
            for front in (1, -1):
                p = leg_fk(q, RBQ10, front, side)
                self.assertLess(np.abs(leg_ik(p, RBQ10, front, side) - q).max(), 1e-9)


class TerrainTest(unittest.TestCase):
    def test_layout(self):
        p = WMPTerrainParams(seed=1, num_rows=3, length_scale=scaling.LENGTH, lateral_scale=scaling.LATERAL)
        t = WMPTerrain(p)
        self.assertEqual(t.env_origins.shape, (3, 20, 3))
        self.assertEqual(t.x_edge_mask.shape, t.height_field_raw.shape)
        ranges = t.type_env_ranges(4096)
        self.assertEqual(set(ranges), set(TERRAIN_NAMES))
        self.assertEqual(max(e for _, e in ranges.values()), 4096)
        # 20 columns, 5 % each: every non-empty type spans whole columns of 4096 / 20 envs.
        self.assertEqual(ranges["gap"][1] - ranges["gap"][0], 1024)


class ScalingTest(unittest.TestCase):
    def test_height_grid_has_187_points(self):
        x, y, r = scaling.rbq10_values()["height_grid"]
        nx = len(torch.arange(-(x + 1e-4) / 2, (x + 1e-4) / 2 + 1e-9, r))
        ny = len(torch.arange(-(y + 1e-4) / 2, (y + 1e-4) / 2 + 1e-9, r))
        self.assertEqual(nx * ny, 187)

    def test_identity_scale_gives_a1(self):
        v = scaling.rbq10_values(1.0)
        self.assertEqual(v["lin_vel_x"], (0.0, 0.8))
        self.assertAlmostEqual(v["tracking_sigma_lin"], 0.15)


class LoopTest(unittest.TestCase):
    def test_mock_interface(self):
        env = MockWMPEnv(num_envs=4)
        obs, priv = env.reset()
        self.assertEqual(obs.shape, (4, 285))
        self.assertEqual(env.get_amp_observations().shape, (4, 30))
        seen_depth = False
        for _ in range(10):
            out = env.step(torch.zeros(4, 12))
            self.assertEqual(len(out), 7)
            if out[4]["depth"] is not None:
                seen_depth = True
                self.assertEqual(out[4]["depth"].shape, (4, 64, 64))
                self.assertLessEqual(float(out[4]["depth"].abs().max()), 0.5 + 1e-6)
        self.assertTrue(seen_depth)

    def test_runner_trains_and_reloads(self):
        torch.manual_seed(0)
        env = MockWMPEnv(num_envs=8, episode_length=30)
        with tempfile.TemporaryDirectory() as log_dir:
            runner = WMPRunner(env, small_cfg(8), log_dir=log_dir, device="cpu")
            runner.learn(4)
            lines = (Path(log_dir) / "progress.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertIn("World_model/image_loss", lines[-1])  # world model was trained
            ckpt = Path(log_dir) / "model_4.pt"
            self.assertTrue(ckpt.is_file())
            other = WMPRunner(MockWMPEnv(num_envs=8, episode_length=30), small_cfg(8), log_dir=None, device="cpu")
            other.load(str(ckpt))
            for a, b in zip(runner._world_model.parameters(), other._world_model.parameters()):
                self.assertTrue(torch.equal(a, b))
            policy = WMPPolicy(other, other.env)
            obs, _ = other.env.reset()
            policy.reset(obs)
            for _ in range(6):
                actions = policy.act()
                self.assertEqual(actions.shape, (8, 12))
                o, _, _, d, info, ids, _ = other.env.step(actions)
                policy.observe(actions, o, d, info, ids)
            self.assertEqual(policy.decode_depth().shape, (8, 64, 64))


if __name__ == "__main__":
    unittest.main()
