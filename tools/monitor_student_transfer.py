"""Evaluate saved Student snapshots on fixed CPU MuJoCo cases while GPU training runs.

No training mutation/restart, GUI, automatic fall reset, or Teacher fallback.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
COURSES=('isaac_flat','isaac_hurdle','isaac_step','isaac_gap','isaac_platforms','isaac_mixed')
CODE=('sim2sim/run_student.py','sim2sim/run_teacher.py','sim2sim/build_scene.py',
      'sim2sim/student_inputs.py','sim2sim/terrains.py','sim2sim/goal_follower.py',
      'sim2sim/student_config.json','tools/export_sim2sim.py',
      'upstream/Isaaclab_Parkour/parkour_isaaclab/terrains/utils.py',
      'upstream/Isaaclab_Parkour/parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrians.py')


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def code_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CODE}


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except (FileNotFoundError,ProcessLookupError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-file',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interval-updates',type=int,default=500)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--poll-seconds',type=float,default=20.)
    a=p.parse_args()
    if a.interval_updates<=0 or a.poll_seconds<=0:p.error('positive intervals required')
    a.output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    hashes=code_hashes()
    initial=read(a.job_file)
    # Model999 has 1,000 completed updates. Completed-count semantics avoid off-by-one errors.
    baseline_updates=int(a.baseline.stem.split('_')[-1])+1
    state={'pid':os.getpid(),'start_ticks':identity(os.getpid()),'state':'running',
           'training_job':str(a.job_file),'baseline':str(a.baseline),'code_sha256':hashes,
           'target_updates':initial['target_updates'],'baseline_updates':baseline_updates,
           'interval_updates':a.interval_updates,'completed':[],
           'conditions':{'courses':list(COURSES),'seeds':a.seeds,'difficulty':.5,'speed_m_s':.6,
                         'seconds':30,'terrain_edges':'source','contact_profile':'firm','autonomous':True,
                         'episodes_per_case':1,'scope':'Small fixed development set; not held-out success rate or Isaac comparison'}}
    def heartbeat(**updates):
        state.update(updates);state['heartbeat']=datetime.now().astimezone().isoformat()
        write(a.output/'status.json',state)
    def evaluate(checkpoint,threshold):
        if code_hashes()!=hashes:raise RuntimeError('Transfer code changed; preserve baseline and restart with an explicit new benchmark revision')
        bundle=a.output/checkpoint.stem
        bundle.mkdir(exist_ok=True)
        heartbeat(current_checkpoint=str(checkpoint),current_threshold=threshold,current_case='export')
        command=[str(ROOT/'run.sh'),'export-sim2sim','--checkpoint',str(checkpoint),'--output',str(bundle/'export')]
        with (bundle/'export.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=read(bundle/'export/manifest.json')
        rows=[]
        for course in COURSES:
            for seed in a.seeds:
                if code_hashes()!=hashes:raise RuntimeError('Transfer code changed during evaluation')
                name=f'{course}_s{seed}'
                heartbeat(current_case=name)
                path=bundle/(name+'.json')
                cmd=[str(ROOT/'run.sh'),'sim2sim','--policy-dir',str(bundle/'export'),
                     '--autonomous','--course',course,'--difficulty','.5','--terrain-seed',str(seed),
                     '--speed','.6','--seconds','30','--terrain-edges','source','--contact-profile','firm','--output',str(path)]
                started=time.monotonic()
                with (bundle/(name+'.log')).open('w') as log:
                    r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
                if r.returncode not in (0,2) or not path.exists():raise RuntimeError(f'{name} runtime error: exit {r.returncode}')
                result=read(path)
                if result['checkpoint_sha256']!=manifest['checkpoint_sha256']:raise RuntimeError('Checkpoint mismatch')
                if result['termination']=='success' and (result['goals_reached']!=8 or result['fall_count']):raise RuntimeError('Invalid success scoring')
                rows.append({'case':name,'course':course,'seed':seed,'wall_seconds':time.monotonic()-started,
                             **{k:result[k] for k in ['termination','goals_reached','fall_count','simulated_seconds','max_abs_joint_speed_rad_s','max_abs_torque_nm']}})
                write(bundle/'summary.json',{'checkpoint':checkpoint.name,'sha256':manifest['checkpoint_sha256'],
                      'code_sha256':hashes,'conditions':state['conditions'],'complete':False,'cases':rows})
                print('TRANSFER_CASE',checkpoint.name,name,result['termination'],result['goals_reached'],flush=True)
        summary=read(bundle/'summary.json');summary['complete']=True
        summary['successes']=sum(r['termination']=='success' for r in rows);summary['case_count']=len(rows)
        write(bundle/'summary.json',summary)
        state['completed'].append({'checkpoint':checkpoint.name,'sha256':manifest['checkpoint_sha256'],
                                  'threshold_updates':threshold,'successes':summary['successes'],'cases':len(rows)})
        heartbeat(current_case=None)
    heartbeat()
    try:
        evaluate(a.baseline,baseline_updates)
        threshold=baseline_updates+a.interval_updates
        while threshold<=initial['target_updates']:
            job=read(a.job_file)
            directory=Path(job['log_dir']) if job.get('log_dir') else None
            candidates=[] if directory is None else sorted(directory.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
            ready=[p for p in candidates if int(p.stem.split('_')[-1])+1>=threshold]
            if ready:
                evaluate(ready[0],threshold);threshold+=a.interval_updates
                continue
            trainer_alive=identity(job.get('trainer_pid',-1))==job.get('trainer_start_ticks') and bool(job.get('trainer_start_ticks'))
            if job.get('state') in ('failed','stopped','controller_failed') or (not trainer_alive and job.get('state') not in ('queued','starting','running')):
                heartbeat(state='training_finished_before_next_threshold',next_threshold=threshold,training_state=job.get('state'))
                return
            heartbeat(state='waiting_for_checkpoint',next_threshold=threshold,training_state=job.get('state'))
            time.sleep(a.poll_seconds)
            heartbeat(state='running')
        heartbeat(state='completed',current_checkpoint=None,current_case=None)
    except BaseException as error:
        heartbeat(state='failed',error=repr(error));raise


if __name__=='__main__':main()
