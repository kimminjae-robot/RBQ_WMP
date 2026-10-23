"""Evaluate a frozen five-view Student on fixed original MuJoCo terrain cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    a.policy_dir=a.policy_dir.resolve();a.output=a.output.resolve()
    meta=json.loads((a.policy_dir/"manifest.json").read_text())
    cases=[(f"isaac_{course}",.5,seed) for course in ("flat","hurdle","step","gap","platforms","mixed") for seed in (42,43,44)]
    cases += [(f"isaac_{course}",1.,42) for course in ("gap","step")]
    results=[]
    def run(case):
        course,difficulty,seed=case;name=f"{course}_d{difficulty:g}_s{seed}"
        target=a.output/(name+".json")
        command=[sys.executable,str(ROOT/"sim2sim/run_student.py"),"--policy-dir",str(a.policy_dir),"--course",course,
            "--difficulty",str(difficulty),"--terrain-seed",str(seed),"--autonomous","--seconds","30","--speed","0.6",
            "--terrain-edges","source","--contact-profile","firm","--output",str(target)]
        with (a.output/(name+".log")).open("w") as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")).returncode
        if not target.exists():return dict(course=course,difficulty=difficulty,seed=seed,error="No rollout report",returncode=code)
        report=json.loads(target.read_text())
        if report["checkpoint_sha256"]!=meta["checkpoint_sha256"]:raise ValueError("Checkpoint changed")
        return dict(course=course,difficulty=difficulty,seed=seed,returncode=code,termination=report["termination"],
            success=report["termination"]=="success",seconds=report["simulated_seconds"],goals=report["goals_reached"],
            max_torque_nm=report["max_abs_torque_nm"],max_joint_speed_rad_s=report["max_abs_joint_speed_rad_s"],
            max_raw_action=report['max_abs_raw_action'],clipped_action_components=report['clipped_action_components'],
            displacement_xyz=report["displacement_xyz"],report=str(target))
    def save(complete=False):
        summary={}
        for row in results:
            key=f"{row['course']}_d{row['difficulty']:g}"
            group=summary.setdefault(key,dict(cases=0,successes=0,falls=0,duration=0,errors=0))
            group["cases"]+=1;group["successes"]+=int(row.get("success",False))
            group["falls"]+=int(row.get("termination")=="fall");group["duration"]+=int(row.get("termination")=="duration");group["errors"]+=int("error" in row)
        report=dict(time_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),complete=complete,expected_cases=len(cases),
            checkpoint=meta["checkpoint"],checkpoint_sha256=meta["checkpoint_sha256"],depth_views=meta["configuration"]["view_order"],
            code_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [ROOT/'sim2sim'/name for name in ('run_student.py','multiview_depth.py','student_inputs.py','run_teacher.py','build_scene.py','terrains.py','goal_follower.py')]},
            conditions=dict(speed=.6,seconds=30,contact_profile="firm",terrain_edges="source",automatic_reset=False,
                            direction_source="known_course_current_next_waypoints",velocity_source="MuJoCo_measured_body"),
            scope="Small fixed development set, one episode per condition; not a held-out transfer success rate",summary=summary,cases=results)
        temp=a.output/"summary.json.tmp";temp.write_text(json.dumps(report,indent=2)+"\n");temp.replace(a.output/"summary.json")
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,case) for case in cases]):
            row=future.result();results.append(row);save();print("CASE",row,flush=True)
    save(complete=True)


if __name__=="__main__":main()
