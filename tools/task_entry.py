"""Register the separate RBQ10 tasks before running an unchanged upstream entry point."""
import argparse
import runpy
import sys
from pathlib import Path
import rbq10_parkour  # Gym registration only; simulator-dependent config stays lazy.

# Reject an explicitly provided wrong-robot checkpoint before starting a GUI.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task")
parser.add_argument("--checkpoint")
selection, _ = parser.parse_known_args(sys.argv[2:])
if selection.task and "Rainbow-RBQ10" in selection.task and selection.checkpoint:
    checkpoint = Path(selection.checkpoint).expanduser()
    if checkpoint.is_file():
        import torch
        from rbq10_parkour.contract import robot_contract, check_checkpoint_contract
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Student distillation initializes from a Teacher; Student playback needs the new camera contract.
        role = "student" if "Student" in selection.task else "teacher"
        initialize = (role == "student" and Path(sys.argv[1]).name == "train.py"
                      and "depth_actor_state_dict" not in saved)
        check_checkpoint_contract(saved.get("robot_contract"), robot_contract(role),
                                  allow_teacher_initialization=initialize)

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
