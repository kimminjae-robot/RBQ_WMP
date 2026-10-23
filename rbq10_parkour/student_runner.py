"""Original distillation with resumable depth optimizer, progress and safe stop."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
from zoneinfo import ZoneInfo

import torch
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from .contract import ROOT
from .runner import _StopAtUpdateBoundary


class RBQ10StudentRunner(OnPolicyRunnerWithExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_requested = False
        self.teacher_checkpoint = None
        if self.log_dir:
            target = Path(self.log_dir)/'source'
            target.mkdir(parents=True,exist_ok=True)
            shutil.copytree(ROOT/'rbq10_parkour',target/'rbq10_parkour',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
            shutil.copyfile(ROOT/'assets/rbq10/front_depth_camera.json',target/'front_depth_camera.json')
            (target/'upstream.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT/'upstream/Isaaclab_Parkour'))

    def load(self, path, load_optimizer=True):
        checkpoint=torch.load(path,map_location=self.device,weights_only=False)
        initializing='depth_actor_state_dict' not in checkpoint
        result=super().load(path,load_optimizer=False)
        if initializing:
            self.teacher_checkpoint=str(Path(path).resolve())
            self.current_learning_iteration=0
            self.tot_timesteps=0
            self.tot_time=0
            print('[RBQ10 Student] Teacher initialized; Student counter starts at 0.',flush=True)
        else:
            state=checkpoint.get('rbq10_student_state',{})
            if load_optimizer and state:
                self.alg.depth_actor_optimizer.load_state_dict(state['depth_optimizer'])
                self.current_learning_iteration=state['next_iteration']
                self.tot_timesteps=state['total_transitions']
                self.tot_time=state['total_time']
                self.teacher_checkpoint=state['teacher_checkpoint']
        return result

    def learn_vision(self,*args,**kwargs):
        old={}
        def request(signum,frame):
            self.stop_requested=True
            print('[RBQ10 Student] Stop requested; save after current distillation update.',flush=True)
        for sig in (signal.SIGTERM,signal.SIGINT):
            old[sig]=signal.signal(sig,request)
        try:
            return super().learn_vision(*args,**kwargs)
        except _StopAtUpdateBoundary:
            print('[RBQ10 Student] Saved and stopped at update boundary.',flush=True)
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
            if self.writer is not None:self.writer.flush();self.writer.close()

    def save(self,path,infos=None):
        temp=path+'.student.tmp'
        super().save(temp,infos)
        saved=torch.load(temp,map_location='cpu',weights_only=False)
        saved['rbq10_student_state']={
            'depth_optimizer':self.alg.depth_actor_optimizer.state_dict(),
            'next_iteration':self.current_learning_iteration+1,
            'total_transitions':self.tot_timesteps,'total_time':self.tot_time,
            'teacher_checkpoint':self.teacher_checkpoint,
            'direction_source':'joystick_command','velocity_source':'measured_body_xyz',
        }
        torch.save(saved,temp)
        os.replace(temp,path)

    def log_vision(self,locs,*args,**kwargs):
        losses={k:float(v) for k,v in locs['loss_dict'].items()}
        if not all(math.isfinite(v) for v in losses.values()):
            raise FloatingPointError(f'Nonfinite Student losses: {losses}')
        # Upstream log_vision counts 24 rather than the actual 120 distillation steps.
        self.tot_timesteps+=(self.depth_encoder_cfg['num_steps_per_env']-self.num_steps_per_env)*self.env.num_envs*self.gpu_world_size
        super().log_vision(locs,*args,**kwargs)
        metrics=getattr(self.env.unwrapped,'teacher_metrics',None)
        stats=metrics.scalars if metrics is not None else {}
        if self.writer is not None:
            for key,value in stats.items():self.writer.add_scalar(key,value,locs['it'])
        entry={
            'time_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'iteration':locs['it'],'updates_completed':locs['it']+1,
            'target_iteration_exclusive':locs['tot_iter'],
            'total_transitions':self.tot_timesteps,'num_envs':self.env.num_envs,
            'steps_per_env':self.depth_encoder_cfg['num_steps_per_env'],
            'collection_seconds':locs['collection_time'],'learning_seconds':locs['learn_time'],
            'losses':losses,'parkour':stats,'teacher_checkpoint':self.teacher_checkpoint,
        }
        if self.log_dir:
            directory=Path(self.log_dir)
            with (directory/'progress.jsonl').open('a') as stream:stream.write(json.dumps(entry,allow_nan=False)+'\n')
            temp=directory/'latest.json.tmp';temp.write_text(json.dumps(entry,indent=2,allow_nan=False)+'\n');temp.replace(directory/'latest.json')
        if self.stop_requested:
            self.save(str(Path(self.log_dir)/f'model_{locs["it"]}.pt'))
            raise _StopAtUpdateBoundary()
