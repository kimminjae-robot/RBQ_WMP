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
CAMERA_POSITION = (0.42, 0.0, 0.10)  # Provisional simulated mount, beyond the 0.40 m trunk front.


def verify_assets():
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    for entry in manifest["files"]:
        if hashlib.sha256((ASSETS / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Official RBQ10 asset hash mismatch: {entry['path']}")


def robot_contract():
    verify_assets()
    return {
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
        "camera_parent": "trunk", "camera_position_m": list(CAMERA_POSITION),
        "camera_mount_status": "simulation_initial_value_not_hardware_calibrated",
        "depth_hw": [58, 87],
    }


def check_checkpoint_contract(saved, expected):
    if saved != expected:
        raise ValueError("Robot checkpoint contract mismatch: RBQ10 requires an RBQ10 parkour checkpoint; "
                         "Go2 and manufacturer locomotion checkpoints cannot be used as RBQ10 parkour checkpoints.")
