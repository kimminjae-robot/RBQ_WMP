"""Five views through the original shared depth CNN; original GRU/actor stay intact."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class MultiViewDepthCNN(nn.Module):
    def __init__(self, shared_backbone: nn.Module):
        super().__init__()
        self.shared_backbone = shared_backbone
        self.fusion = nn.Linear(5 * 32, 32)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, :32].copy_(torch.eye(32))
            self.fusion.bias.zero_()

    def forward(self, images: torch.Tensor):
        if images.dim() != 4 or images.shape[1] != 5 or images.shape[2] != 58 or images.shape[3] != 87:
            raise ValueError("Expected N x 5 x 58 x 87 in front,ground_0,ground_1,ground_2,ground_3 order")
        features = []
        for index in range(5):
            if torch.jit.is_scripting():
                latent = self.shared_backbone(images[:, index])
            elif self.training and torch.is_grad_enabled():
                # Recompute CNN activations during backward; same loss and gradients.
                latent = checkpoint(self.shared_backbone, images[:, index], use_reentrant=False)
            else:
                latent = self.shared_backbone(images[:, index])
            features.append(latent)
        return self.fusion(torch.cat(features, dim=-1))


def expand_front_depth_state(source, target):
    """Exact pretrained weights plus identity-front/zero-ground fusion initialization."""
    converted = dict(target)
    mapped = {}
    for name, value in source.items():
        new_name = name.replace("base_backbone.", "base_backbone.shared_backbone.", 1) if name.startswith("base_backbone.") else name
        if new_name not in target or target[new_name].shape != value.shape:
            raise ValueError(f"Incompatible front depth parameter: {name}")
        mapped[new_name] = value
    expected = set(target) - {"base_backbone.fusion.weight", "base_backbone.fusion.bias"}
    if set(mapped) != expected:
        raise ValueError("Incomplete front-to-multiview checkpoint")
    converted.update(mapped)
    return converted
