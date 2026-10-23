"""A separate checkpoint identity prevents accidental front-only deployment/resume."""
import hashlib
import json
from .contract import ROOT, robot_contract

CALIBRATION_PATH = ROOT / "assets/rbq10/multiview_depth_cameras.json"
CALIBRATION = json.loads(CALIBRATION_PATH.read_text())
VIEW_NAMES = tuple(view["name"] for view in CALIBRATION["views"])
SENSOR_NAMES = tuple(view["sensor_name"] for view in CALIBRATION["views"])


def multiview_contract():
    contract = robot_contract("student")
    contract.update(
        schema="rbq10_extreme_parkour_student_depth5_v1",
        depth_input_shape=[5, 58, 87],
        depth_view_order=list(VIEW_NAMES),
        depth_calibration=CALIBRATION,
        depth_calibration_sha256=hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest(),
        depth_encoder="shared_original_CNN_5x32_linear_fusion_32_original_GRU512",
        depth_front_intrinsics="original_f11.041_aperture20.955x12.240_106x60",
        depth_ground_fov_hv_deg=[87.0, 58.0],
        depth_max_distance_m=2.0,
        depth_ground_min_optical_z_m=0.2,
        depth_ground_invalid_fill="max_distance_before_original_preprocessing",
        depth_measurement="euclidean_distance_to_camera",
        depth_update_policy_steps=5,
        depth_buffer_length=2,
    )
    return contract


def validate_multiview_checkpoint(checkpoint, allow_front=False):
    if not all(name in checkpoint for name in ("depth_encoder_state_dict", "depth_actor_state_dict")):
        raise ValueError("A real trained Student checkpoint is required")
    saved = checkpoint.get("robot_contract")
    if saved == multiview_contract():
        return "multiview_resume"
    if allow_front and saved == robot_contract("student"):
        return "front_warm_start"
    raise ValueError("Wrong Student camera contract; front-only and five-view checkpoints are separate")
