"""Robot identity and ordering shared by configuration, checks and checkpoints."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/rbq10"
COMMIT = "c7b80ba49f9a79f983cb977e2726cb2246701fcf"
JOINT_NAMES = [f"joint{i}_{suffix}" for i, suffix in enumerate(
    ("HRR", "HRP", "HRK", "HLR", "HLP", "HLK", "FRR", "FRP", "FRK", "FLR", "FLP", "FLK")
)]
# Isaac/PhysX traverses the four parallel leg chains breadth first.
NATIVE_JOINT_NAMES = [JOINT_NAMES[i] for i in (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)]
FOOT_NAMES = [f"{leg}_foot" for leg in ("HR", "HL", "FR", "FL")]
INFO = json.loads((ASSETS / "official/info.json").read_text())["config_info"]
HOME = INFO["init_state"]["default_joint_angles"]
ACTION_SCALE = 0.25  # Preserve the original parkour action interface.
CAMERA_CALIBRATION = json.loads((ASSETS / "front_depth_camera.json").read_text())
CAMERA_POSITION = tuple(CAMERA_CALIBRATION["translation_m"])
CAMERA_ROTATION = tuple(CAMERA_CALIBRATION["quaternion_wxyz_ros"])


def verify_assets():
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    for entry in manifest["files"]:
        if hashlib.sha256((ASSETS / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Official RBQ10 asset hash mismatch: {entry['path']}")


def robot_contract(role="teacher"):
    verify_assets()
    if role not in ("teacher", "student"):
        raise ValueError(f"Unknown policy role: {role}")
    contract = {
        "schema": "rbq10_extreme_parkour_v1",
        "robot": "Rainbow_RBQ10_without_arm",
        "source_commit": COMMIT,
        "urdf_sha256": hashlib.sha256((ASSETS / "resources/urdf/rbq10.urdf").read_bytes()).hexdigest(),
        "base_body": "trunk", "joint_order": NATIVE_JOINT_NAMES, "foot_order": FOOT_NAMES,
        "action_scale_rad": ACTION_SCALE, "physics_dt": 0.005, "policy_dt": 0.02,
        "policy_observation_dim": 753, "action_dim": 12,
        "actuator": "official_implicit_PD",
        "kp_R_P_K": [123.39138, 123.39138, 127.77215],
        "kd": 2.5, "effort_nm_R_P_K": [100.0, 100.0, 150.0], "velocity_rad_s": 50.0,
        "camera_parent": "trunk", "camera_position_m": [0.42, 0.0, 0.10],
        "camera_mount_status": "simulation_initial_value_not_hardware_calibrated",
        "depth_hw": [58, 87],
    }
    # Existing Teacher has no camera input: preserve its exact checkpoint/resume contract.
    if role == "student":
        contract.update({
            "schema": "rbq10_extreme_parkour_student_front_depth_v2",
            "camera_position_m": list(CAMERA_POSITION),
            "camera_quaternion_wxyz_ros": list(CAMERA_ROTATION),
            "camera_mount_status": "manufacturer_front_depth_TF",
            "camera_sensor": CAMERA_CALIBRATION["sensor"],
            "camera_tf_source": CAMERA_CALIBRATION["source_url"],
        })
    return contract


def check_checkpoint_contract(saved, expected, allow_teacher_initialization=False):
    if (allow_teacher_initialization and saved == robot_contract("teacher")
            and expected == robot_contract("student")):
        return  # Explicit original Teacher -> Student distillation initialization only.
    if saved != expected:
        raise ValueError("Robot checkpoint contract mismatch: RBQ10 requires an RBQ10 parkour checkpoint; "
                         "Go2 and manufacturer locomotion checkpoints cannot be used as RBQ10 parkour checkpoints.")
