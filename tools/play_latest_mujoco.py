"""Export the latest saved real RBQ10 Student and start keyboard MuJoCo playback."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--no-keyboard',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.checkpoint:
        checkpoint=args.checkpoint.expanduser().resolve()
    else:
        candidates=list((ROOT/'logs/rsl_rl/rbq10_extreme_parkour').glob('*phase4_student*/model_*.pt'))
        if not candidates:raise SystemExit('No saved RBQ10 Student checkpoint available')
        checkpoint=max(candidates,key=lambda p:p.stat().st_mtime_ns)
    if not checkpoint.is_file():raise SystemExit(f'Checkpoint not found: {checkpoint}')
    output=ROOT/'outputs/sim2sim/live'/checkpoint.parent.name/checkpoint.stem
    print(f'Latest saved Student: {checkpoint}',flush=True)
    subprocess.run([sys.executable,str(ROOT/'tools/export_sim2sim.py'),'--checkpoint',str(checkpoint),'--output',str(output)],check=True,cwd=ROOT)
    python=os.environ.get('RBQ10_MUJOCO_PYTHON',str(ROOT/'.venv-sim2sim/bin/python'))
    command=[python,str(ROOT/'sim2sim/run_student.py'),'--policy-dir',str(output)]
    if not args.no_keyboard and '--autonomous' not in remaining:command.append('--keyboard')
    os.chdir(ROOT)
    os.execv(python,command+remaining)


if __name__=='__main__':main()
