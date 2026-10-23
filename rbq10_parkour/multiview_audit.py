"""Check actual Isaac sensor poses/intrinsics against published RBQ mounting TF."""
import math
import xml.etree.ElementTree as ET
import torch
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
from .contract import ASSETS
from .multiview_contract import CALIBRATION


def audit_cameras(env):
    robot = env.scene["robot"]
    body = robot.body_names.index("trunk")
    body_pos = robot.data.body_link_pos_w[:, body]
    body_quat = robot.data.body_link_quat_w[:, body]
    urdf = ET.parse(ASSETS / "resources/urdf/rbq10.urdf").getroot()
    hips = {name: tuple(float(x) for x in urdf.find(f"joint[@name='{name}']/origin").get("xyz").split())
            for name in ("joint0_HRR", "joint3_HLR", "joint6_FRR", "joint9_FLR")}
    # The manufacturer's body frame and URDF trunk agree at all four hip origins.
    assert hips == {"joint0_HRR":(-.31218,-.09,0.), "joint3_HLR":(-.31218,.09,0.),
                    "joint6_FRR":(.31218,-.09,0.), "joint9_FLR":(.31218,.09,0.)}
    assert not any("camera" in link.get("name", "").lower() for link in urdf.findall("link"))
    report = dict(scope="Published hardware geometry and live Isaac sensor audit, not a connection to the physical robot",
                  body_frame="URDF trunk link frame; no inertial/CoM offset applied",
                  body_frame_hip_offsets_m=hips, urdf_has_camera_links=False,
                  topics_source="https://rainbowrobotics.github.io/RBQ/software/sdk/ros/topics.html#vision-camera-sensors",
                  views=[])
    for view in CALIBRATION["views"]:
        camera = env.scene[view["sensor_name"]]
        data = camera.data
        position, quaternion = subtract_frame_transforms(body_pos, body_quat, data.pos_w, data.quat_w_ros)
        rotation = matrix_from_quat(quaternion)
        expected_pos = torch.tensor(view["translation_m"], device=env.device)
        expected_rot = torch.tensor(view["rotation_matrix"], device=env.device)
        pos_error = float((position - expected_pos).abs().max())
        rot_error = float((rotation - expected_rot).abs().max())
        K = data.intrinsic_matrices[0]
        pattern = camera.cfg.pattern_cfg
        hfov = math.degrees(2 * math.atan(pattern.width / (2 * float(K[0, 0]))))
        vfov = math.degrees(2 * math.atan(pattern.height / (2 * float(K[1, 1]))))
        ground = view["name"] != "front"
        if ground:
            # Full training maps span ~200m; float32 world-coordinate subtraction
            # has larger error than the small CPU smoke map. Allow 0.1mm.
            assert pos_error < 1e-4 and rot_error < 2e-5, (view["name"], pos_error, rot_error)
            assert abs(hfov - 87) < 1e-4 and abs(vfov - 58) < 1e-4
            assert float(rotation[0, 2, 2]) < -.8
        else:
            # Source training randomizes only front pitch by +/-5 degrees at startup.
            assert pos_error < 1e-4 and rot_error < .1, (view["name"], pos_error, rot_error)
        sensor_id = 4 if not ground else int(view["name"].split("_")[-1])
        report["views"].append(dict(name=view["name"], hardware_sensor_id=sensor_id,
            hardware_depth_topic=f"/rbq/vision/sensor_{sensor_id}/depth/compressed",
            hardware_intrinsics_topic=f"/rbq/vision/sensor_{sensor_id}/depth/camera_info",
            hardware_pose_topic=f"/rbq/vision/sensor_{sensor_id}/depth/tf",
            configured_position_m=list(camera.cfg.offset.pos),
            actual_position_m=position[0].tolist(), actual_rotation_matrix=rotation[0].tolist(),
            optical_forward_in_body=rotation[0,:,2].tolist(), position_max_error_m=pos_error,
            rotation_max_error=rot_error, actual_fov_hv_deg=[hfov, vfov],
            intrinsic_matrix=K.tolist(), raw_hw=[pattern.height, pattern.width],
            rate_policy_input_hz=10, max_range_m=camera.cfg.max_distance,
            nominal_hardware_fov_matches=ground,
        ))
    report["remaining_hardware_differences"] = [
        "Front optics retain the existing training H87/V58 approximately; hardware OAK-D Wide specifies H127/V79.5.",
        "Ground D430 H87/V58 are nominal specifications; per-device CameraInfo/distortion and stream TF must be measured on the robot.",
        "Ground camera 3 published y translation is -1.1e-050, retained literally; confirm the live stream TF.",
        "Raycasting covers terrain only, without robot self-occlusion, stereo holes, camera interference or sensor noise.",
        "Hardware optical-Z depth must be converted through CameraInfo to Euclidean range before the original preprocessing, with common 2m clipping and ground 0.2m optical-Z invalid handling.",
        "Native sensor resolution/FPS, exposure, per-camera delay and timestamp synchronization are not yet calibrated; policy consumes delayed 10Hz 58x87 views.",
    ]
    return report
