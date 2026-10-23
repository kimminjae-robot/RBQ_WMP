"""Isolated front+four ground-view Student; source terrain/rewards/curriculum reused."""
import math
from isaaclab.utils import configclass
from .config import RBQ10StudentEnvCfg
from .agents import RBQ10StudentRunnerCfg
from .multiview_contract import CALIBRATION, SENSOR_NAMES, multiview_contract
from .multiview_observations import MultiViewImageFeatures


@configclass
class RBQ10MultiViewStudentEnvCfg(RBQ10StudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for view in CALIBRATION["views"][1:]:
            camera = self.scene.depth_camera.copy()
            camera.offset.pos = tuple(view["translation_m"])
            camera.offset.rot = tuple(view["quaternion_wxyz_ros"])
            camera.offset.convention = "ros"
            pattern = camera.pattern_cfg
            pattern.horizontal_aperture = 2 * pattern.focal_length * math.tan(math.radians(87 / 2))
            pattern.vertical_aperture = 2 * pattern.focal_length * math.tan(math.radians(58 / 2))
            setattr(self.scene, view["sensor_name"], camera)
        term = self.observations.depth_camera.depth_cam
        term.func = MultiViewImageFeatures
        term.params = {"sensor_names": list(SENSOR_NAMES)}


@configclass
class RBQ10MultiViewStudentRunnerCfg(RBQ10StudentRunnerCfg):
    experiment_name = "rbq10_extreme_parkour_multiview"
    robot_contract: dict = multiview_contract()
