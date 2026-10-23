"""Reuse the source crop/normalization/delay independently for each physical camera."""
import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from parkour_isaaclab.envs.mdp.observations import image_features


class GroundDepthImageFeatures(image_features):
    """D430 cannot measure axial depth below its nominal 0.2m operating limit."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        pattern = self.camera_sensor.cfg.pattern_cfg
        # RayCaster distance_to_camera is Euclidean; D430's depth is optical-Z.
        self.optical_z_cosine = self.camera_sensor.ray_directions[0, :, 0].reshape(pattern.height, pattern.width)

    def _process_depth_image(self, image):
        valid = torch.isfinite(image) & (image * self.optical_z_cosine >= 0.2)
        image = torch.where(valid, image, torch.full_like(image, self.clipping_range))
        return super()._process_depth_image(image)


class MultiViewImageFeatures(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.terms = []
        for name in cfg.params["sensor_names"]:
            params = dict(sensor_cfg=SceneEntityCfg(name), resize=(58, 87), buffer_len=2, debug_vis=False)
            feature_class = image_features if name == "depth_camera" else GroundDepthImageFeatures
            self.terms.append((feature_class(ObservationTermCfg(func=feature_class, params=params), env), params))

    def reset(self, env_ids=None):
        for term, _ in self.terms:
            term.reset(env_ids)

    def __call__(self, env, sensor_names):
        return torch.stack([term(env, **params) for term, params in self.terms], dim=1)
