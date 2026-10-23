"""Verify and import the bundled Teacher into the original runner's resume folder."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
from rbq10_parkour.contract import ROOT, robot_contract, check_checkpoint_contract

source = ROOT / 'checkpoints/rbq10_teacher/model_3674.pt'
manifest = json.loads((source.parent / 'manifest.json').read_text())
if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['sha256']:
    raise ValueError('Bundled Teacher checkpoint hash mismatch')
saved = torch.load(source, map_location='cpu', weights_only=True)
check_checkpoint_contract(saved.get('robot_contract'), robot_contract())
if saved['iter'] != 3674 or 'rbq10_training_state' not in saved:
    raise ValueError('Expected the actual resumable Phase 3 checkpoint at iteration 3674')
target = ROOT / 'logs/rsl_rl/rbq10_extreme_parkour/imported_teacher_3674/model_3674.pt'
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != manifest['sha256']:
    raise FileExistsError(f'Preserving different existing checkpoint: {target}')
if not target.exists():
    shutil.copyfile(source, target)
print(target)
