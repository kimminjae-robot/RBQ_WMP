"""Original Teacher runner with JSON status, complete auxiliary saves and safe stop.

PPO collection, losses, networks and update order remain in the upstream runner.
"""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
from zoneinfo import ZoneInfo
import torch
from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor
from .contract import ROOT


class _StopAtUpdateBoundary(Exception):
    pass


class RBQ10TeacherRunner(OnPolicyRunnerWithExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_requested = False
        self.restored_auxiliary_state = False
        if self.log_dir:
            target = Path(self.log_dir) / 'source'
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ROOT / 'rbq10_parkour', target / 'rbq10_parkour',
                            ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
            shutil.copyfile(ROOT / 'run.sh', target / 'run.sh')
            shutil.copyfile(ROOT / 'assets/rbq10/manifest.json', target / 'asset_manifest.json')
            (target / 'upstream.patch').write_bytes(subprocess.check_output(
                ['git', 'diff', '--binary'], cwd=ROOT / 'upstream/Isaaclab_Parkour'))

    def learn_rl(self, *args, **kwargs):
        old_handlers = {}

        def request_stop(signum, frame):
            self.stop_requested = True
            print(f'[RBQ10] Signal {signum}: finish this PPO update, save, then stop.', flush=True)

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        try:
            return super().learn_rl(*args, **kwargs)
        except _StopAtUpdateBoundary:
            print('[RBQ10] Stopped after saving the completed PPO update.', flush=True)
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

    def log(self, locs, *args, **kwargs):
        losses = {key: float(value) for key, value in locs['loss_dict'].items()}
        if not all(math.isfinite(value) for value in losses.values()):
            raise FloatingPointError(f'Nonfinite Teacher losses: {losses}')
        super().log(locs, *args, **kwargs)
        metrics = self.env.unwrapped.teacher_metrics
        entry = {
            'time_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'iteration': locs['it'], 'updates_completed': locs['it'] + 1,
            'target_iteration_exclusive': locs['tot_iter'],
            'total_transitions': self.tot_timesteps,
            'num_envs': self.env.num_envs, 'steps_per_env': self.num_steps_per_env,
            'collection_seconds': locs['collection_time'], 'learning_seconds': locs['learn_time'],
            'mean_reward': statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
            'mean_episode_length_steps': statistics.mean(locs['lenbuffer']) if locs['lenbuffer'] else None,
            'learning_rate': self.alg.learning_rate, 'losses': losses,
            'parkour': metrics.scalars if metrics is not None else {},
        }
        if self.log_dir:
            directory = Path(self.log_dir)
            with (directory / 'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
            temporary = directory / 'latest.json.tmp'
            temporary.write_text(json.dumps(entry, indent=2, allow_nan=False) + '\n')
            temporary.replace(directory / 'latest.json')
        if self.stop_requested:
            self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))
            raise _StopAtUpdateBoundary()

    def save(self, path, infos=None):
        # Keep the upstream checkpoint format and add auxiliary state that its
        # save() omits. Atomic rename prevents readers loading a partial file.
        temporary = str(path) + f'.{os.getpid()}.tmp'
        try:
            super().save(temporary, infos)
            saved = torch.load(temporary, map_location='cpu', weights_only=False)
            saved['rbq10_training_state'] = {
                'schema': 1, 'completed_iteration': self.current_learning_iteration,
                'ppo_counter': self.alg.counter,
                'estimator_optimizer': self.alg.estimator_optimizer.state_dict(),
                'history_optimizer': self.alg.hist_encoder_optimizer.state_dict(),
                'learning_rate': self.alg.learning_rate,
                'total_transitions': self.tot_timesteps, 'total_training_seconds': self.tot_time,
                'resume_note': 'Optimizer/schedule state restored; simulator episodes restart.',
            }
            torch.save(saved, temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def load(self, path, load_optimizer=True):
        infos = super().load(path, load_optimizer=load_optimizer)
        if load_optimizer:
            saved = torch.load(path, map_location=self.device, weights_only=False)
            state = saved.get('rbq10_training_state')
            if state is None:
                raise ValueError('Teacher resume needs rbq10_training_state; select a Phase 3 checkpoint.')
            self.alg.counter = state['ppo_counter']
            self.alg.estimator_optimizer.load_state_dict(state['estimator_optimizer'])
            self.alg.hist_encoder_optimizer.load_state_dict(state['history_optimizer'])
            self.alg.learning_rate = state['learning_rate']
            self.tot_timesteps = state['total_transitions']
            self.tot_time = state['total_training_seconds']
            self.current_learning_iteration = state['completed_iteration'] + 1
            self.restored_auxiliary_state = True
            print(f'[RBQ10] Auxiliary optimizers/schedule restored; next iteration {self.current_learning_iteration}.', flush=True)
        return infos
