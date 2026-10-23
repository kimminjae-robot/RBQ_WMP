"""Original environment with explicit RBQ10 asset/order assertions."""
from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
from .contract import NATIVE_JOINT_NAMES, FOOT_NAMES


class RBQ10ParkourEnv(ParkourManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        robot = self.scene["robot"]
        if robot.joint_names != NATIVE_JOINT_NAMES:
            raise ValueError(f"RBQ10 joint order differs from the checkpoint contract: {robot.joint_names}")
        if not set(FOOT_NAMES).issubset(robot.body_names):
            raise ValueError("RBQ10 fixed feet must be preserved as separate contact bodies")
        if self.action_manager.total_action_dim != 12:
            raise ValueError("RBQ10 parkour expects 12 actions")
        self.teacher_metrics = None
        if getattr(self.cfg, "training_metrics_enabled", False):
            from .metrics import TeacherMetrics
            self.teacher_metrics = TeacherMetrics(self)

    def _reset_idx(self, env_ids):
        metrics = getattr(self, "teacher_metrics", None)
        if metrics is not None:
            metrics.finish(env_ids)
        return super()._reset_idx(env_ids)

    def step(self, action):
        metrics = self.teacher_metrics
        if metrics is not None:
            metrics.begin()
        result = super().step(action)
        if metrics is not None:
            # A new dictionary keeps earlier rollout log samples immutable.
            self.extras["log"] = dict(self.extras.get("log", {}), **metrics.end())
        return result
