"""Original Student distillation with a five-view CNN adapter and explicit warm start."""
import hashlib
import os
from pathlib import Path
import shutil
import torch
from .student_runner import RBQ10StudentRunner
from .multiview_contract import CALIBRATION_PATH, validate_multiview_checkpoint
from .multiview_network import MultiViewDepthCNN, expand_front_depth_state


class RBQ10MultiViewStudentRunner(RBQ10StudentRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.empirical_normalization or self.alg.rnd:
            raise ValueError("This branch requires the original unnormalized, non-RND Student configuration")
        encoder = self.alg.depth_encoder
        encoder.base_backbone = MultiViewDepthCNN(encoder.base_backbone).to(self.device)
        self.alg.depth_actor_optimizer = torch.optim.Adam(
            list(self.alg.depth_actor.parameters()) + list(encoder.parameters()),
            lr=self.depth_encoder_cfg["learning_rate"],
        )
        self.multiview_initialization = None
        if self.log_dir:
            shutil.copyfile(CALIBRATION_PATH, Path(self.log_dir) / "source" / CALIBRATION_PATH.name)

    def load(self, path, load_optimizer=True):
        saved = torch.load(path, map_location=self.device, weights_only=False)
        mode = validate_multiview_checkpoint(saved, allow_front=True)
        self.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
        self.alg.estimator.load_state_dict(saved["estimator_state_dict"], strict=True)
        self.alg.depth_actor.load_state_dict(saved["depth_actor_state_dict"], strict=True)
        state = saved.get("rbq10_student_state", {})
        self.teacher_checkpoint = state.get("teacher_checkpoint")
        if mode == "front_warm_start":
            expanded = expand_front_depth_state(saved["depth_encoder_state_dict"], self.alg.depth_encoder.state_dict())
            self.alg.depth_encoder.load_state_dict(expanded, strict=True)
            self.current_learning_iteration = 0
            self.tot_timesteps = 0
            self.tot_time = 0
            self.multiview_initialization = dict(
                mode=mode, checkpoint=str(Path(path).resolve()),
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                source_updates=state.get("next_iteration", saved["iter"] + 1),
                optimizer="fresh_original_Adam_lr_0.001_new_architecture",
                fusion="front_identity_ground_zero",
            )
        else:
            self.alg.depth_encoder.load_state_dict(saved["depth_encoder_state_dict"], strict=True)
            self.multiview_initialization = saved["rbq10_multiview_initialization"]
            self.current_learning_iteration = state["next_iteration"]
            self.tot_timesteps = state["total_transitions"]
            self.tot_time = state["total_time"]
            if load_optimizer:
                self.alg.depth_actor_optimizer.load_state_dict(state["depth_optimizer"])
        print(f"[RBQ10 MultiDepth] {mode}; branch update {self.current_learning_iteration}", flush=True)
        return saved.get("infos")

    def save(self, path, infos=None):
        temporary = path + ".multiview.tmp"
        super().save(temporary, infos)
        saved = torch.load(temporary, map_location="cpu", weights_only=False)
        saved["rbq10_multiview_initialization"] = self.multiview_initialization
        torch.save(saved, temporary)
        os.replace(temporary, path)

    def log_vision(self, locs, *args, **kwargs):
        super().log_vision(locs, *args, **kwargs)
        if self.writer is not None:
            fusion = self.alg.depth_encoder.base_backbone.fusion.weight.detach()
            for index, name in enumerate(("front", "ground_0", "ground_1", "ground_2", "ground_3")):
                self.writer.add_scalar(f"MultiDepth/fusion_{name}_norm", fusion[:, index*32:(index+1)*32].norm().item(), locs["it"])
