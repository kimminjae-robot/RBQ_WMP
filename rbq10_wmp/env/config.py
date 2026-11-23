# Environment values follow bytedance/WMP ``legged_gym/envs/a1/a1_amp_config.py``
# (BSD-3-Clause, ETH Zurich / NVIDIA, with Bytedance's modifications); RBQ10 values
# come from rbq10_wmp/scaling.py and the official RBQ10 asset (rbq10_parkour).
"""Isaac Lab configuration of the WMP task for RBQ10 (paper Sec. IV, Table I)."""
from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg, PinholeCameraPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from rbq10_parkour.contract import CAMERA_POSITION, CAMERA_ROTATION, FOOT_NAMES, JOINT_NAMES
from rbq10_parkour.robot import RBQ10_CFG

from .. import scaling
from . import mdp
from .terrain import WMPTerrainGenerator

S = scaling.LENGTH
V = scaling.rbq10_values(S)
FEET = SceneEntityCfg("contact_forces", body_names=FOOT_NAMES, preserve_order=True)
FEET_BODIES = SceneEntityCfg("robot", body_names=FOOT_NAMES, preserve_order=True)
PENALISED = [".*_thigh", ".*_calf"]  # WMP penalize_contacts_on = ["thigh", "calf"]
MODEL_INTERVAL = 5                   # k: world model / depth update every 5 policy steps (0.1 s)
DEPTH_SHAPE = (64, 64)
DEPTH_HFOV_DEG = 58.0
MASS_RATIO = 40.684 / 12.454        # URDF total mass RBQ10 / A1: WMP added base mass 0-3 kg -> 0-9.8 kg


@configclass
class WMPTerrainGeneratorCfg:
    """WMP terrain map (rbq10_wmp/env/terrain.py) served to Isaac Lab's TerrainImporter."""

    class_type: type = WMPTerrainGenerator
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 25.0
    terrain_length: float = 8.0 * S
    terrain_width: float = 8.0 * S
    num_rows: int = 10
    num_cols: int = 20
    # [wave, slope, stairs up, stairs down, discrete, gap, climb, tilt, crawl, rough flat]
    terrain_proportions: tuple = (0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05)
    slope_threshold: float = 0.75
    curriculum: bool = True
    length_scale: float = S
    lateral_scale: float = scaling.LATERAL
    seed: int = 1
    edge_half_width: int = 1


# ---------------------------------------------------------------------- scene
@configclass
class WMPSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WMPTerrainGeneratorCfg(),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = RBQ10_CFG.replace(
        init_state=RBQ10_CFG.init_state.replace(pos=(0.0, 0.0, 0.35 * S)),
    )
    # Critic height map: 17 x 11 = 187 points, the WMP 1.6 m x 1.0 m grid scaled by S.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=GridPatternCfg(resolution=V["height_grid"][2],
                                   size=[V["height_grid"][0] + 1e-4, V["height_grid"][1] + 1e-4]),
        mesh_prim_paths=["/World/ground"],
        max_distance=40.0,
        debug_vis=False,
    )
    # 64 x 64 egocentric depth (paper Sec. IV-A) at the manufacturer's front OAK-D mount.
    depth_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        offset=RayCasterCameraCfg.OffsetCfg(pos=CAMERA_POSITION, rot=CAMERA_ROTATION, convention="ros"),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2.0 * math.tan(math.radians(DEPTH_HFOV_DEG / 2)),
            vertical_aperture=2.0 * math.tan(math.radians(DEPTH_HFOV_DEG / 2)),
            width=DEPTH_SHAPE[1], height=DEPTH_SHAPE[0],
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=V["depth_far_clip"] + 1.0,
        update_period=0.0,  # lazily ray-cast only when the depth term reads it (every k steps)
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


# ---------------------------------------------------------------- observations
OBS_SCALES = dict(lin_vel=1.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05, height_measurements=5.0,
                  contact_force=0.005, com_pos=20.0, pd_gains=5.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """``[privileged(53) | prop(33) | action(12) | heights(187)]`` = 285 (num_obs = num_privileged_obs)."""

        wmp = ObsTerm(
            func=mdp.WMPObservation,
            clip=(-100.0, 100.0),
            params=dict(
                asset_cfg=SceneEntityCfg("robot"),
                contact_cfg=SceneEntityCfg("contact_forces"),
                height_cfg=SceneEntityCfg("height_scanner"),
                foot_names=FOOT_NAMES,
                penalised_names=PENALISED,
                base_body="trunk",
                base_height=V["base_height"],
                scales=OBS_SCALES,
            ),
        )

        def __post_init__(self):
            self.enable_corruption = False  # WMP: add_noise = False
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        image = ObsTerm(
            func=mdp.WMPDepthImage,
            params=dict(sensor_cfg=SceneEntityCfg("depth_camera"), shape=DEPTH_SHAPE, buffer_len=2,
                        interval=MODEL_INTERVAL, near_clip=0.0, far_clip=V["depth_far_clip"]),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    depth: DepthCfg = DepthCfg()


# --------------------------------------------------------------------- actions
@configclass
class ActionsCfg:
    joint_pos = mdp.LatencyJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True,
        randomize_latency=True, latency_range_s=(0.0, 0.005),
    )


# -------------------------------------------------------------------- commands
@configclass
class CommandsCfg:
    base_velocity = mdp.WMPVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(V["resampling_time_s"], V["resampling_time_s"]),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.WMPVelocityCommandCfg.Ranges(
            lin_vel_x=V["lin_vel_x"], lin_vel_y=(0.0, 0.0), ang_vel_z=V["ang_vel_yaw"], heading=(0.0, 0.0)),
        flat_ranges=mdp.WMPVelocityCommandCfg.Ranges(
            lin_vel_x=V["lin_vel_x"], lin_vel_y=(0.0, 0.0), ang_vel_z=V["ang_vel_yaw"], heading=None),
    )


# --------------------------------------------------------------------- rewards
@configclass
class RewardsCfg:
    """WMP reward scales (per second; the reward manager multiplies by dt like legged_gym)."""

    tracking_lin_vel = RewTerm(func=mdp.reward_tracking_lin_vel, weight=1.5,
                               params=dict(sigma=V["tracking_sigma_lin"], lin_vel_clip=V["lin_vel_clip"]))
    tracking_ang_vel = RewTerm(func=mdp.reward_tracking_ang_vel, weight=0.5,
                               params=dict(sigma=V["tracking_sigma_ang"]))
    torques = RewTerm(func=mdp.reward_torques, weight=-0.0001)
    dof_acc = RewTerm(func=mdp.reward_dof_acc, weight=-2.5e-7)
    feet_air_time = RewTerm(func=mdp.reward_feet_air_time, weight=0.5,
                            params=dict(sensor_cfg=FEET, target=V["feet_air_time_target"]))
    collision = RewTerm(func=mdp.reward_collision, weight=-1.0,
                        params=dict(sensor_cfg=SceneEntityCfg("contact_forces", body_names=PENALISED)))
    feet_stumble = RewTerm(func=mdp.reward_feet_stumble, weight=-0.1, params=dict(sensor_cfg=FEET))
    action_rate = RewTerm(func=mdp.reward_action_rate, weight=-0.03)
    feet_edge = RewTerm(func=mdp.reward_feet_edge, weight=-1.0, params=dict(sensor_cfg=FEET, asset_cfg=FEET_BODIES))
    dof_error = RewTerm(func=mdp.reward_dof_error, weight=-0.04)
    lin_vel_z = RewTerm(func=mdp.reward_lin_vel_z, weight=-1.0)
    cheat = RewTerm(func=mdp.reward_cheat, weight=-1.0)
    stuck = RewTerm(func=mdp.reward_stuck, weight=-1.0)


# WMP reward_curriculum: feet_edge weight ramps 0.1 -> 1.0 between iterations 4000 and 10000.
REWARD_CURRICULUM = {"feet_edge": (4000, 10000, 0.1, 1.0)}


# ---------------------------------------------------------------- terminations
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.terminate_base_contact,
                            params=dict(sensor_cfg=SceneEntityCfg("contact_forces", body_names="trunk"), threshold=1.0))
    velocity_violation = DoneTerm(func=mdp.terminate_velocity_violation,
                                  params=dict(threshold=V["violation_threshold"]))
    fall = DoneTerm(func=mdp.terminate_fall)


# ---------------------------------------------------------------------- events
@configclass
class EventsCfg:
    # startup (WMP _process_rigid_shape_props / _process_rigid_body_props, camera y_angle)
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material, mode="startup",
        params=dict(asset_cfg=SceneEntityCfg("robot", body_names=".*"), static_friction_range=(0.5, 2.0),
                    dynamic_friction_range=(0.5, 2.0), restitution_range=(0.0, 0.0), num_buckets=64,
                    make_consistent=True),
    )
    added_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass, mode="startup",
        params=dict(asset_cfg=SceneEntityCfg("robot", body_names="trunk"),
                    mass_distribution_params=(0.0, 3.0 * MASS_RATIO), operation="add"),
    )
    link_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass, mode="startup",
        params=dict(asset_cfg=SceneEntityCfg("robot", body_names=".*_(hip|thigh|calf|foot)"),
                    mass_distribution_params=(0.8, 1.2), operation="scale"),
    )
    base_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com, mode="startup",
        params=dict(asset_cfg=SceneEntityCfg("robot", body_names="trunk"),
                    com_range={"x": (-0.05 * S, 0.05 * S), "y": (-0.05 * S, 0.05 * S), "z": (-0.05 * S, 0.05 * S)}),
    )
    camera_pitch = EventTerm(
        func=mdp.randomize_camera_pitch, mode="startup",
        params=dict(sensor_cfg=SceneEntityCfg("depth_camera"), pitch_range_deg=(-5.0, 5.0)),
    )
    # reset (WMP reset_idx: gains, dofs x U(0.5, 1.5), root state)
    actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains, mode="reset",
        params=dict(asset_cfg=SceneEntityCfg("robot", joint_names=".*"), stiffness_distribution_params=(0.8, 1.2),
                    damping_distribution_params=(0.8, 1.2), operation="scale"),
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale, mode="reset",
        params=dict(position_range=(0.5, 1.5), velocity_range=(0.0, 0.0)),
    )
    reset_base = EventTerm(func=mdp.reset_root_state_wmp, mode="reset",
                           params=dict(xy_range=1.0, vel_range=0.5))
    # interval (WMP push_robots every 15 s)
    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity, mode="interval",
        interval_range_s=(V["push_interval_s"], V["push_interval_s"]),
        params=dict(velocity_range={"x": (-V["max_push_vel_xy"], V["max_push_vel_xy"]),
                                    "y": (-V["max_push_vel_xy"], V["max_push_vel_xy"])}),
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.wmp_terrain_levels)


# ------------------------------------------------------------------------- env
@configclass
class RBQ10WMPEnvCfg(ManagerBasedRLEnvCfg):
    scene: WMPSceneCfg = WMPSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # WMP-specific (read by RBQ10WMPEnv / WMPEnv)
    model_interval: int = MODEL_INTERVAL
    depth_shape: tuple = DEPTH_SHAPE
    clip_actions: float = 6.0
    amp_joint_order: list = JOINT_NAMES  # reference-motion order (datasets/rbq10_mocap)
    reward_curriculum: dict = REWARD_CURRICULUM
    min_normalized_std: tuple = (0.05, 0.02, 0.05)  # per R, P, K joint
    # RBQ10 pitch joints are +-2*pi in the URDF; cap the range used for min std at A1's thigh range.
    min_std_max_joint_range: float = 5.24

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.viewer.eye = (2.0, 3.0, 1.6)
        self.viewer.lookat = (0.4, 0.0, 0.4)


@configclass
class RBQ10WMPEnvCfg_PLAY(RBQ10WMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.terrain.max_init_terrain_level = None
        self.events.push_robot = None
        self.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
        self.actions.joint_pos.randomize_latency = False
