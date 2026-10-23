"""Five physically attached MuJoCo cameras with the trained ray-depth contract."""
import copy
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import torch
import torch.nn.functional as F


FORMAT = "rbq10_mujoco_student_depth5_v1"
VIEW_ORDER = ["front", "ground_0", "ground_1", "ground_2", "ground_3"]


def configuration(contract):
    if contract.get("schema") != "rbq10_extreme_parkour_student_depth5_v1":
        raise ValueError("Expected the five-view Student checkpoint contract")
    views = []
    for index, calibration in enumerate(contract["depth_calibration"]["views"]):
        focal = 11.041
        view = dict(name=calibration["name"], parent="trunk", position_m=calibration["translation_m"],
                    quaternion_wxyz_ros=calibration["quaternion_wxyz_ros"], width=106, height=60,
                    focal_length=focal, horizontal_aperture=20.955, vertical_aperture=12.240,
                    max_distance_m=contract["depth_max_distance_m"], min_optical_z_m=0.)
        if index:
            h, v = contract["depth_ground_fov_hv_deg"]
            view.update(horizontal_aperture=2*focal*math.tan(math.radians(h/2)),
                        vertical_aperture=2*focal*math.tan(math.radians(v/2)),
                        min_optical_z_m=contract["depth_ground_min_optical_z_m"])
        views.append(view)
    if [v["name"] for v in views] != VIEW_ORDER or contract["depth_input_shape"] != [5,58,87]:
        raise ValueError("Five-view ordering/shape mismatch")
    return dict(cameras=views, view_order=VIEW_ORDER, direction_source="joystick", yaw_rate_to_error_seconds=.5,
                terrain_label="parkour", velocity_source="mujoco", depth_update_policy_steps=5,
                depth_buffer_length=2, action_delay_policy_steps=1)


def validate_configuration(cfg, contract):
    expected = configuration(contract)
    for key in ("cameras", "view_order", "depth_update_policy_steps", "depth_buffer_length", "action_delay_policy_steps"):
        if cfg.get(key) != expected[key]:
            raise ValueError(f"Five-view configuration differs from the trained contract: {key}")


def add_cameras(xml, views):
    root = ET.fromstring(xml)
    for view in views:
        body = root.find(f".//body[@name='{view['parent']}']")
        if body is None:
            raise ValueError(f"Missing camera parent: {view['parent']}")
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, np.array(view["quaternion_wxyz_ros"], dtype=float))
        gl_matrix = matrix.reshape(3,3) @ np.diag([1.,-1.,-1.])
        gl_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(gl_quaternion, gl_matrix.ravel())
        fx = view["width"] * view["focal_length"] / view["horizontal_aperture"]
        fy = view["height"] * view["focal_length"] / view["vertical_aperture"]
        ET.SubElement(body, "camera", name="depth_"+view["name"], mode="fixed",
            pos=" ".join(map(str,view["position_m"])), quat=" ".join(map(str,gl_quaternion)),
            resolution=f"{view['width']} {view['height']}", sensorsize="1 1", focalpixel=f"{fx} {fy}", principalpixel="0 0")
    return ET.tostring(root, encoding="unicode")


class MultiViewDepthInput:
    def __init__(self, model, data, cfg):
        self.m, self.d, self.cfg = model, data, copy.deepcopy(cfg)
        self.views = cfg["cameras"]
        if [v["name"] for v in self.views] != VIEW_ORDER:
            raise ValueError("Camera order mismatch")
        self.camera_ids, self.rays = [], []
        for view in self.views:
            camera = model.camera("depth_"+view["name"]).id
            if model.cam_bodyid[camera] != model.body(view["parent"]).id:
                raise ValueError("MuJoCo camera is not attached to the trunk link")
            w, h = view["width"], view["height"]
            fx = w*view["focal_length"]/view["horizontal_aperture"]
            fy = h*view["focal_length"]/view["vertical_aperture"]
            actual = model.cam_intrinsic[camera,:2] / model.cam_sensorsize[camera] * model.cam_resolution[camera]
            if not np.allclose(actual, [fx,fy], rtol=2e-7):
                raise ValueError("MuJoCo camera intrinsics do not match policy configuration")
            u,v = np.meshgrid(np.arange(w)+.5,np.arange(h)+.5,indexing="xy")
            rays = np.stack(((u-w/2)/fx,(v-h/2)/fy,np.ones_like(u)),axis=-1).reshape(-1,3)
            self.rays.append(rays/np.linalg.norm(rays,axis=1,keepdims=True))
            self.camera_ids.append(camera)
        self.buffer = None
        self.latest_raw = None
        self.latest_processed = None
        self.latest_origins = None
        self.latest_directions = None

    def preprocess(self, raw):
        frames = []
        for index, view in enumerate(self.views):
            image = np.asarray(raw[index],dtype=np.float32).copy()
            optical_z = image * self.rays[index][:,2].reshape(image.shape)
            valid = np.isfinite(image) & (optical_z >= view["min_optical_z_m"])
            image = np.where(valid, image, view["max_distance_m"]).astype(np.float32)
            crop = torch.from_numpy(image[:-2,4:-4].copy())[None,None]
            processed = F.interpolate(crop,size=(58,87),mode="bicubic",align_corners=False,antialias=True)[0,0]
            frames.append(processed/view["max_distance_m"]-.5)
        return torch.stack(frames,dim=0)[None]

    def sample(self):
        raw, origins, directions = [], [], []
        group = np.array([1,0,0,0,0,0],dtype=np.uint8)  # Same terrain-only visibility as Isaac.
        for index,view in enumerate(self.views):
            camera = self.camera_ids[index]
            origin = self.d.cam_xpos[camera].copy()
            ros_rotation = self.d.cam_xmat[camera].reshape(3,3) @ np.diag([1.,-1.,-1.])
            rays = np.ascontiguousarray(self.rays[index] @ ros_rotation.T)
            distance = np.full(len(rays),-1.,dtype=np.float64)
            hit = np.full(len(rays),-1,dtype=np.int32)
            # MuJoCo 3.3.0 mj_multiRay missed analytic box hits in the parity
            # fixture. Use the scalar API, verified against Isaac's Warp rays.
            geom = np.array([-1],dtype=np.int32)
            for ray_index, ray in enumerate(rays):
                distance[ray_index] = mujoco.mj_ray(self.m,self.d,origin,ray,group,1,-1,geom)
            distance = np.where(distance>=0,np.minimum(distance,view["max_distance_m"]),view["max_distance_m"])
            raw.append(distance.astype(np.float32).reshape(view["height"],view["width"]))
            origins.append(origin);directions.append(rays)
        self.latest_raw = np.stack(raw)
        self.latest_origins = np.stack(origins)
        self.latest_directions = np.stack(directions)
        processed = self.preprocess(self.latest_raw)
        self.latest_processed = processed
        self.buffer = [processed.clone(),processed.clone()] if self.buffer is None else [self.buffer[-1],processed]
        return self.buffer[-2]
