"""Fixed-level evaluation using the original scene, controller and reset logic."""
from .config import RBQ10TeacherEnvCfg_EVAL
from .env import RBQ10ParkourEnv


def fixed_level_config(level=9, num_envs=20, seed=42):
    cfg = RBQ10TeacherEnvCfg_EVAL()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 20.0
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 10, 10
    gen.random_difficulty = False
    gen.difficulty_range = (0.0, 1.0)
    gen.curriculum = True  # Original deterministic generator, ten rows.
    for name, sub in gen.sub_terrains.items():
        sub.proportion = 0.2 if name in ('parkour_gap', 'parkour_hurdle', 'parkour_flat', 'parkour_step', 'parkour') else 0.0
    cfg.scene.terrain.max_init_terrain_level = level
    cfg.parkours.base_parkour.fixed_terrain_level = level
    cfg.parkours.base_parkour.debug_vis = False
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.resampling_time_range = (60.0, 60.0)
    cfg.events.push_by_setting_velocity = None
    cfg.events.randomize_rigid_body_mass = None
    cfg.events.randomize_rigid_body_com = None
    cfg.events.physics_material.params['friction_range'] = (0.9, 0.9)
    cfg.viewer.resolution = (960, 540)
    cfg.viewer.eye = (1.8, 2.8, 1.8)
    cfg.viewer.lookat = (0.3, 0.0, 0.35)
    return cfg


class FixedLevelEvaluationEnv(RBQ10ParkourEnv):
    def _reset_idx(self, env_ids):
        observer = getattr(self, 'episode_end_observer', None)
        if observer is not None:
            observer(env_ids)
        return super()._reset_idx(env_ids)
