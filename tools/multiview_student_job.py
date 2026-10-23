"""Independent five-view training queue; never interrupts the front-only job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import student_job as jobs

ROOT = jobs.ROOT
FRONT_ACTIVE = jobs.ACTIVE
jobs.JOBS = ROOT / "logs/phase4_multiview"
jobs.ACTIVE = jobs.JOBS / "active_job.json"
jobs.EXPERIMENT = ROOT / "logs/rsl_rl/rbq10_extreme_parkour_multiview"


def front_job():
    pointer = jobs.read(FRONT_ACTIVE)
    return jobs.read(Path(pointer["job_file"])) if pointer else {}


def worker(path):
    job = jobs.read(path)
    job.update(worker_pid=os.getpid(), worker_start_ticks=jobs.identity(os.getpid()))
    jobs.write(path, job)
    try:
        while True:
            job.update(jobs.read(path))
            job["heartbeat_kst"] = jobs.now()
            if job.get("stop_requested"):
                job.update(state="cancelled", finished_kst=jobs.now())
                jobs.write(path, job)
                return
            front = front_job()
            if jobs.alive(front, "trainer") or jobs.alive(front, "worker"):
                job.update(state="waiting_for_front_training", waiting_for_job=front.get("job_file"))
            else:
                memory = subprocess.check_output([
                    "nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
                ], text=True).strip()
                job["gpu_free_mib"] = int(memory)
                if int(memory) >= job["minimum_gpu_free_mib"]:
                    # Reuse the existing process identity, disk guard, progress and graceful-stop controller.
                    job.update(state="ready", ready_kst=jobs.now())
                    jobs.write(path, job)
                    jobs.worker(path)
                    return
                job["state"] = "waiting_for_gpu_memory"
            jobs.write(path, job)
            time.sleep(10)
    except BaseException as error:
        job.update(state="controller_failed", error=repr(error), finished_kst=jobs.now())
        jobs.write(path, job)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    start = commands.add_parser("start")
    start.add_argument("--checkpoint", type=Path, required=True, help="Explicit front warm start or five-view resume checkpoint")
    start.add_argument("--num_envs", type=int, default=192)
    start.add_argument("--iterations", type=int, default=2000, help="Additional updates in the five-view branch")
    start.add_argument("--seed", type=int, default=1)
    start.add_argument("--minimum_gpu_free_mib", type=int, default=12000)
    commands.add_parser("status")
    commands.add_parser("stop")
    run = commands.add_parser("_worker")
    run.add_argument("job_file", type=Path)
    args = parser.parse_args()
    if args.operation == "_worker":
        worker(args.job_file)
        return
    jobs.JOBS.mkdir(parents=True, exist_ok=True)
    if args.operation == "status":
        print(json.dumps(jobs.status(), indent=2))
        return
    with (jobs.JOBS / "job.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = jobs.status()
        if args.operation == "stop":
            if not (previous.get("worker_alive") or previous.get("trainer_alive")):
                raise SystemExit("No live five-view job")
            path = Path(jobs.read(jobs.ACTIVE)["job_file"])
            stored = jobs.read(path)
            stored.update(stop_requested=True, stop_reason="requested_via_cli", stop_requested_kst=jobs.now())
            jobs.write(path, stored)
            if jobs.alive(previous, "trainer"):
                os.kill(previous["trainer_pid"], signal.SIGTERM)
            print("Five-view stop requested. The front-only job is unaffected.")
            return
        if previous.get("worker_alive") or previous.get("trainer_alive"):
            raise SystemExit("A five-view job is already queued or running; use status")
        if min(args.num_envs, args.iterations, args.minimum_gpu_free_mib) <= 0:
            parser.error("num_envs, iterations and minimum_gpu_free_mib must be positive")
        import torch
        from rbq10_parkour.multiview_contract import validate_multiview_checkpoint
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        mode = validate_multiview_checkpoint(saved, allow_front=True)
        next_iteration = saved["rbq10_student_state"]["next_iteration"] if mode == "multiview_resume" else 0
        del saved
        tag = jobs.datetime.now(jobs.ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
        path = jobs.JOBS / f"job_{tag}.json"
        run_name = f"depth5_student_s{args.seed}_{tag}"
        command = [str(ROOT / "run.sh"), "train-rbq10-multiview", "--checkpoint", str(checkpoint),
                   "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
                   "--seed", str(args.seed), "--run_name", run_name, "--device", "cuda:0"]
        job = dict(created_kst=jobs.now(), run_name=run_name, command=command, state="queued",
                   target_updates=next_iteration + args.iterations, num_envs=args.num_envs, seed=args.seed,
                   initialization_mode=mode, source_checkpoint=str(checkpoint),
                   minimum_gpu_free_mib=args.minimum_gpu_free_mib,
                   console_log=str(jobs.JOBS / f"student_{tag}.log"), job_file=str(path))
        jobs.write(path, job)
        with (jobs.JOBS / f"controller_{tag}.log").open("w") as output:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker", str(path)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        pointer = dict(job_file=str(path), worker_pid=process.pid, worker_start_ticks=jobs.identity(process.pid))
        jobs.write(jobs.ACTIVE, pointer)
        print(json.dumps({**job, **pointer}, indent=2))


if __name__ == "__main__":
    main()
