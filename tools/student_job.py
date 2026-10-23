"""Start, inspect or stop one detached RBQ10 Student training job in this project."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'logs/phase4'
EXPERIMENT = ROOT / 'logs/rsl_rl/rbq10_extreme_parkour'
ACTIVE = JOBS / 'active_job.json'


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write(path, data):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(job, role):
    pid = job.get(role + '_pid')
    return bool(pid and job.get(role + '_start_ticks') and
                identity(pid) == job[role + '_start_ticks'])


def worker(path):
    job = read(path)
    job.update(worker_pid=os.getpid(), worker_start_ticks=identity(os.getpid()), state='starting')
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TZ='Asia/Seoul')
    with Path(job['console_log']).open('w') as console:
        process = subprocess.Popen(job['command'], cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        job.update(trainer_pid=process.pid, trainer_start_ticks=identity(process.pid), state='running')
        write(path, job)
        try:
            while True:
                returncode = process.poll()
                # Preserve a concurrent stop request from the CLI.
                job.update(read(path))
                directories = sorted(EXPERIMENT.glob('*_' + job['run_name']))
                if directories:
                    directory = directories[-1]
                    job['log_dir'] = str(directory)
                    checkpoints = sorted(directory.glob('model_*.pt'),
                                         key=lambda p: int(re.fullmatch(r'model_(\d+)\.pt', p.name)[1]))
                    job['latest_checkpoint'] = str(checkpoints[-1]) if checkpoints else None
                job['heartbeat_kst'] = now()
                job['disk_free_gib'] = round(shutil.disk_usage(ROOT).free / 2**30, 2)
                if job['disk_free_gib'] < 2.0 and returncode is None and not job.get('stop_requested'):
                    job.update(stop_requested=True, stop_reason='disk_free_below_2_gib')
                    process.send_signal(signal.SIGTERM)
                if returncode is not None:
                    latest = read(Path(job['log_dir']) / 'latest.json') if job.get('log_dir') else {}
                    completed = latest.get('updates_completed', 0) >= job['target_updates']
                    job.update(returncode=returncode, finished_kst=now(),
                               state='completed' if returncode == 0 and completed else
                                     'stopped' if returncode == 0 and job.get('stop_requested') else 'failed')
                    write(path, job)
                    return
                write(path, job)
                time.sleep(5)
        except BaseException as error:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            job.update(state='controller_failed', error=repr(error), finished_kst=now())
            write(path, job)
            raise


def status():
    pointer = read(ACTIVE)
    if not pointer:
        return {'state': 'not_started'}
    job = read(Path(pointer['job_file']))
    for key in ('worker_pid', 'worker_start_ticks'):
        job.setdefault(key, pointer.get(key))
    job['worker_alive'] = alive(job, 'worker')
    job['trainer_alive'] = alive(job, 'trainer')
    if job.get('log_dir'):
        latest = read(Path(job['log_dir']) / 'latest.json')
        job['progress'] = {key: latest.get(key) for key in (
            'time_kst', 'iteration', 'updates_completed', 'total_transitions', 'mean_reward',
            'mean_episode_length_steps', 'collection_seconds', 'learning_seconds', 'losses',
        )}
        job['terrain_outcomes'] = {key: value for key, value in latest.get('parkour', {}).items()
                                   if key.endswith(('/course_success_rate', '/fall_rate', '/completed_episodes'))}
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    start = commands.add_parser('start')
    start.add_argument('--num_envs', type=int, default=192)
    start.add_argument('--iterations', type=int, default=1000, help='Additional Student updates to run.')
    start.add_argument('--seed', type=int, default=1)
    start.add_argument('--resume', type=Path)
    commands.add_parser('status')
    commands.add_parser('stop')
    run = commands.add_parser('_worker')
    run.add_argument('job_file', type=Path)
    args = parser.parse_args()
    if args.operation == '_worker':
        worker(args.job_file)
        return
    JOBS.mkdir(parents=True, exist_ok=True)
    if args.operation == 'status':
        print(json.dumps(status(), indent=2))
        return
    with (JOBS / 'job.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job = status()
        if args.operation == 'stop':
            if not alive(job, 'trainer'):
                raise SystemExit('No matching live Student process to stop.')
            job_file = Path(read(ACTIVE)['job_file'])
            stored = read(job_file)
            stored.update(stop_requested=True, stop_reason='requested_via_cli', stop_requested_kst=now())
            write(job_file, stored)
            os.kill(job['trainer_pid'], signal.SIGTERM)
            print('Stop requested; the runner saves at the next completed distillation update.')
            return
        if job.get('worker_alive') or job.get('trainer_alive'):
            raise SystemExit('This project already has a live managed Student job. Use status.')
        if args.num_envs <= 0 or args.iterations <= 0:
            parser.error('num_envs and iterations must be positive')
        if shutil.disk_usage(ROOT).free < 2 * 2**30:
            parser.error('Less than 2 GiB free in this project filesystem; free space or use another PC before starting.')
        tag = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
        run_name = f'phase4_student_s{args.seed}_{tag}'
        command = [str(ROOT / 'run.sh'), 'train-rbq10-student', '--num_envs', str(args.num_envs),
                   '--max_iterations', str(args.iterations), '--seed', str(args.seed),
                   '--run_name', run_name, '--logger', 'tensorboard']
        target_updates = args.iterations
        if not args.resume:
            parser.error('Student initialization requires --resume pointing to a real RBQ10 Teacher checkpoint')
        if args.resume:
            checkpoint = args.resume.expanduser().resolve()
            match = re.fullmatch(r'model_(\d+)\.pt', checkpoint.name)
            if not checkpoint.is_file() or not checkpoint.is_relative_to(EXPERIMENT) or not match:
                parser.error('resume must be a model_N.pt inside this project RBQ10 experiment')
            import torch
            checkpoint_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if 'depth_actor_state_dict' in checkpoint_state:
                target_updates += checkpoint_state.get('rbq10_student_state', {}).get('next_iteration', int(match[1])+1)
            command += ['--resume', '--load_run', checkpoint.parent.name, '--checkpoint', checkpoint.name]
        job_file = JOBS / f'job_{tag}.json'
        spec = {'created_kst': now(), 'run_name': run_name, 'command': command, 'state': 'queued',
                'target_updates': target_updates, 'num_envs': args.num_envs, 'seed': args.seed,
                'console_log': str(JOBS / f'student_{tag}.log'), 'job_file': str(job_file)}
        write(job_file, spec)
        with (JOBS / f'controller_{tag}.log').open('w') as output:
            controller = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker', str(job_file)],
                                          cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                          start_new_session=True)
        write(ACTIVE, {'job_file': str(job_file), 'worker_pid': controller.pid,
                       'worker_start_ticks': identity(controller.pid)})
        print(json.dumps(spec, indent=2))


if __name__ == '__main__':
    main()
