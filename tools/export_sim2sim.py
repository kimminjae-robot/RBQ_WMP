"""Export the original Teacher actor + estimator without starting Isaac Sim."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_original_actor(checkpoint):
    # Avoid modules/__init__.py, which also imports the Isaac-dependent runner.
    name = '_rbq_original_networks'
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / 'upstream/Isaaclab_Parkour/scripts/rsl_rl/modules')]
    sys.modules[name] = package
    module = importlib.import_module(name + '.actor_critic_with_encoder')
    actor = module.Actor(
        num_actions=12, scan_encoder_dims=[128, 64, 32],
        actor_hidden_dims=[512, 256, 128], priv_encoder_dims=[64, 20],
        activation=torch.nn.ELU(), num_prop=53, num_scan=132, num_hist=10,
        num_priv_latent=29, num_priv_explicit=9,
        state_history_encoder={'class_name': 'StateHistoryEncoder', 'channel_size': 10},
    )
    actor.load_state_dict({k[6:]: v for k, v in checkpoint['model_state_dict'].items()
                           if k.startswith('actor.')}, strict=True)
    cls = importlib.import_module(name + '.feature_extractors.estimator').DefaultEstimator
    estimator = cls(53, 9, [128, 64])
    estimator.load_state_dict(checkpoint['estimator_state_dict'], strict=True)
    return actor.eval(), estimator.eval()


class TeacherInference(torch.nn.Module):
    def __init__(self, actor, estimator):
        super().__init__()
        self.actor, self.estimator = actor, estimator

    def forward(self, obs):
        x = obs.clone()
        x[:, 185:194] = self.estimator(x[:, :53])
        return self.actor(x, hist_encoding=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/rbq10_teacher/model_3674.pt')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/sim2sim/teacher')
    args = p.parse_args()
    torch.set_num_threads(1)
    c = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    spec = importlib.util.spec_from_file_location('_rbq_contract', ROOT / 'rbq10_parkour/contract.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_checkpoint_contract(c.get('robot_contract'), module.robot_contract())
    if 'depth_actor_state_dict' in c:
        raise ValueError('This exporter accepts Teacher checkpoints only; Student needs an explicit recurrent deployment contract.')
    actor, estimator = load_original_actor(c)
    policy = TeacherInference(actor, estimator).eval()
    torch.manual_seed(42)
    example = torch.randn(1, 753)
    traced = torch.jit.trace(policy, example)
    max_error = 0.0
    with torch.inference_mode():
        for _ in range(16):
            x = torch.randn(1, 753)
            error = (policy(x) - traced(x)).abs().max().item()
            max_error = max(max_error, error)
        assert max_error < 1e-5, max_error
        # These privileged fields must not affect history+estimator inference.
        changed = example.clone()
        changed[:, 185:223] += 100
        assert torch.allclose(policy(example), policy(changed), atol=1e-6)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / 'teacher.pt'
    traced.save(str(model_path))
    meta = {
        'format': 'rbq10_mujoco_teacher_v1', 'role': 'teacher', 'contract': c['robot_contract'],
        'checkpoint': args.checkpoint.name,
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'iteration': c['iter'], 'input_shape': [1, 753], 'output_shape': [1, 12],
        'estimator_embedded': True, 'history_encoding': True,
        'export_max_abs_error': max_error,
        'scan_semantics': 'IsaacLab 2.3.2 sensor pos_w including offset, original clip(pos_w.z-hit.z-0.3,-1,1)',
        'deployment_status': 'simulation Teacher; privileged terrain and course labels remain',
    }
    (args.output / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
