"""Robot-only adapters around the original Go2 scene and MDP configurations."""
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from parkour_tasks.default_cfg import CAMERA_CFG, VIEWER
from parkour_tasks.extreme_parkour_task.config.go2.parkour_teacher_cfg import (
    UnitreeGo2TeacherParkourEnvCfg,
    UnitreeGo2TeacherParkourEnvCfg_PLAY,
    UnitreeGo2TeacherParkourEnvCfg_EVAL,
)
from parkour_tasks.extreme_parkour_task.config.go2.parkour_student_cfg import (
    UnitreeGo2StudentParkourEnvCfg,
    UnitreeGo2StudentParkourEnvCfg_PLAY,
    UnitreeGo2StudentParkourEnvCfg_EVAL,
)

from .contract import ACTION_SCALE, CAMERA_POSITION, CAMERA_ROTATION, FOOT_NAMES
from .robot import RBQ10_CFG


def configure_rbq10(cfg):
    """Change only the robot, selectors, mount, and explicit checkpoint identity."""
    cfg.scene.robot = RBQ10_CFG.copy()
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
    cfg.scene.height_scanner.offset.pos = (0.42, 0.0, 20.0)
    if hasattr(cfg.scene, "depth_camera"):
        cfg.scene.depth_camera = CAMERA_CFG.copy()
        cfg.scene.depth_camera.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        cfg.scene.depth_camera.offset.pos = CAMERA_POSITION
        cfg.scene.depth_camera.offset.rot = CAMERA_ROTATION
        cfg.scene.depth_camera.offset.convention = "ros"
        cfg.scene.depth_camera.update_period = cfg.sim.dt * cfg.decimation
        # The source camera mesh is cosmetic and has Go2-specific placement.
        cfg.scene.depth_camera_usd = None
        cfg.observations.depth_camera.depth_cam.params["debug_vis"] = False
    term = cfg.observations.policy.extreme_parkour_observations
    term.params["base_body_name"] = "trunk"
    term.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
    cfg.actions.joint_pos.scale = ACTION_SCALE
    for name in ("randomize_rigid_body_mass", "randomize_rigid_body_com", "base_external_force_torque"):
        event = getattr(cfg.events, name, None)
        if event is not None:
            event.params["asset_cfg"] = SceneEntityCfg("robot", body_names="trunk")
    collision = getattr(cfg.rewards, "reward_collision", None)
    if collision is not None:
        collision.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["trunk", ".*_calf", ".*_thigh"]
        )
    edge = getattr(cfg.rewards, "reward_feet_edge", None)
    if edge is not None:
        edge.params["asset_cfg"] = SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True)
        edge.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
        edge.params["base_body_name"] = "trunk"
    hip = getattr(cfg.rewards, "reward_hip_pos", None)
    if hip is not None:
        hip.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="joint.*_[HF][RL]R")
    cfg.viewer = VIEWER.copy()
    cfg.viewer.eye = (1.2, 2.2, 1.4)
    cfg.viewer.lookat = (0.3, 0.0, 0.45)


@configclass
class RBQ10TeacherEnvCfg(UnitreeGo2TeacherParkourEnvCfg):
    training_metrics_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        configure_rbq10(self)


@configclass
class RBQ10TeacherEnvCfg_PLAY(UnitreeGo2TeacherParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_rbq10(self)


@configclass
class RBQ10TeacherEnvCfg_EVAL(UnitreeGo2TeacherParkourEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        configure_rbq10(self)


@configclass
class RBQ10StudentEnvCfg(UnitreeGo2StudentParkourEnvCfg):
    training_metrics_enabled: bool = True
    def __post_init__(self):
        super().__post_init__()
        configure_rbq10(self)


@configclass
class RBQ10StudentEnvCfg_PLAY(UnitreeGo2StudentParkourEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_rbq10(self)


@configclass
class RBQ10StudentEnvCfg_EVAL(UnitreeGo2StudentParkourEnvCfg_EVAL):
    def __post_init__(self):
        super().__post_init__()
        configure_rbq10(self)


def validation_config(kind="flat", num_envs=4, seed=1):
    """Small original terrains for model validation; not the training curriculum."""
    cfg = RBQ10StudentEnvCfg_PLAY()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 60.0
    cfg.sim.dt, cfg.decimation = 0.002, 5  # Manufacturer locomotion controller: 100 Hz.
    cfg.sim.render_interval = cfg.decimation
    for sensor in (cfg.scene.depth_camera, cfg.scene.height_scanner, cfg.scene.contact_forces):
        sensor.update_period = cfg.sim.dt * cfg.decimation
    cfg.actions.joint_pos.use_delay = False
    cfg.events.random_camera_position = None
    cfg.events.randomize_rigid_body_mass = None
    cfg.events.randomize_rigid_body_com = None
    cfg.events.push_by_setting_velocity = None
    cfg.events.physics_material.params["friction_range"] = (0.9, 0.9)
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.debug_vis = False
    cfg.parkours.base_parkour.debug_vis = False
    gen = cfg.scene.terrain.terrain_generator
    gen.num_rows, gen.num_cols = 2, num_envs
    gen.random_difficulty = False
    gen.difficulty_range = (0.0, 0.0)
    cfg.scene.terrain.max_init_terrain_level = 0
    for sub in gen.sub_terrains.values():
        sub.proportion = 0.0
    key = "parkour_flat" if kind == "flat" else "parkour_step"
    sub = gen.sub_terrains[key]
    sub.proportion = 1.0
    sub.apply_roughness = False
    sub.half_valid_width = (1.5, 1.5)
    sub.y_range = (-0.1, 0.1)
    sub.x_range = (0.7, 1.0)
    if kind == "step":
        sub.step_height = "0.05"
    # Reuse the original terrain generator and parkour reset/waypoint handling.
    return cfg
