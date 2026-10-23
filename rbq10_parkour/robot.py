"""Official RBQ10 URDF and PD values, with separate fixed feet for contact sensing."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .contract import ASSETS, HOME, verify_assets

verify_assets()
RBQ10_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSETS / "resources/urdf/rbq10.urdf"),
        usd_dir=str(ASSETS / "usd"), usd_file_name="rbq10.usd",
        fix_base=False, merge_fixed_joints=False, make_instanceable=False,
        replace_cylinders_with_capsules=False, activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            linear_damping=0.0, angular_damping=0.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55), joint_pos=HOME, joint_vel={".*": 0.0},
    ),
    actuators={
        "legs_R": ImplicitActuatorCfg(
            joint_names_expr=[".*HRR", ".*HLR", ".*FRR", ".*FLR"],
            stiffness=123.39138, damping=2.5, effort_limit_sim=100.0,
            velocity_limit_sim=50.0, friction=0.0, armature=0.014058,
        ),
        "legs_P": ImplicitActuatorCfg(
            joint_names_expr=[".*HRP", ".*HLP", ".*FRP", ".*FLP"],
            stiffness=123.39138, damping=2.5, effort_limit_sim=100.0,
            velocity_limit_sim=50.0, friction=0.0, armature=0.014058,
        ),
        "legs_K": ImplicitActuatorCfg(
            joint_names_expr=[".*HRK", ".*HLK", ".*FRK", ".*FLK"],
            stiffness=127.77215, damping=2.5, effort_limit_sim=150.0,
            velocity_limit_sim=50.0, friction=0.0, armature=0.0214816,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
