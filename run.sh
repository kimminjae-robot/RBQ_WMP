#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
upstream_root="$project_root/upstream/Isaaclab_Parkour"
export PYTHONPATH="$project_root:$upstream_root/parkour_tasks:$upstream_root"
export PYTHONDONTWRITEBYTECODE=1
python_bin="${RBQ10_PYTHON:-$project_root/.venv/bin/python}"
mode="${1:-play-student}"
if (($#)); then shift; fi
cd "$project_root"
case "$mode" in
  play-student)
    exec "$python_bin" "$upstream_root/scripts/rsl_rl/play.py" --task Isaac-Extreme-Parkour-Student-Unitree-Go2-Play-v0 --checkpoint "$project_root/checkpoints/go2_student.pt" --num_envs 16 --disable_depth_debug "$@" ;;
  play-teacher)
    exec "$python_bin" "$upstream_root/scripts/rsl_rl/play.py" --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --checkpoint "$project_root/checkpoints/go2_teacher.pt" --num_envs 16 "$@" ;;
  train-teacher)
    exec "$python_bin" "$upstream_root/scripts/rsl_rl/train.py" --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0 --seed 1 --headless "$@" ;;
  train-student)
    exec "$python_bin" "$upstream_root/scripts/rsl_rl/train.py" --task Isaac-Extreme-Parkour-Student-Unitree-Go2-v0 --seed 1 --headless "$@" ;;
  play-rbq10-teacher)
    exec "$python_bin" "$project_root/tools/task_entry.py" "$upstream_root/scripts/rsl_rl/play.py" --task Isaac-Extreme-Parkour-Teacher-Rainbow-RBQ10-Play-v0 --num_envs 16 "$@" ;;
  play-rbq10-student)
    exec "$python_bin" "$project_root/tools/task_entry.py" "$upstream_root/scripts/rsl_rl/play.py" --task Isaac-Extreme-Parkour-Student-Rainbow-RBQ10-Play-v0 --num_envs 16 --disable_depth_debug "$@" ;;
  train-rbq10-teacher)
    exec "$python_bin" "$project_root/tools/task_entry.py" "$upstream_root/scripts/rsl_rl/train.py" --task Isaac-Extreme-Parkour-Teacher-Rainbow-RBQ10-v0 --seed 1 --headless "$@" ;;
  train-rbq10-student)
    exec "$python_bin" "$project_root/tools/task_entry.py" "$upstream_root/scripts/rsl_rl/train.py" --task Isaac-Extreme-Parkour-Student-Rainbow-RBQ10-v0 --seed 1 --headless "$@" ;;
  validate-rbq10)
    exec "$python_bin" "$project_root/tools/validate_rbq10.py" "$@" ;;
  compare-policies)
    exec "$python_bin" "$project_root/tools/compare_teacher_student.py" "$@" ;;
  evaluate-teacher)
    exec "$python_bin" "$project_root/tools/evaluate_teacher.py" "$@" ;;
  export-sim2sim)
    exec "$python_bin" "$project_root/tools/export_sim2sim.py" "$@" ;;
  sim2sim-latest)
    exec "$python_bin" "$project_root/tools/play_latest_mujoco.py" "$@" ;;
  sim2sim)
    exec "${RBQ10_MUJOCO_PYTHON:-$project_root/.venv-sim2sim/bin/python}" "$project_root/sim2sim/run_student.py" "$@" ;;
  import-teacher)
    exec "$python_bin" "$project_root/tools/import_teacher_checkpoint.py" "$@" ;;
  mujoco-map)
    exec "${RBQ10_MUJOCO_PYTHON:-$project_root/.venv-sim2sim/bin/python}" "$project_root/tools/build_mujoco_terrain_map.py" "$@" ;;
  student-job)
    exec "$python_bin" "$project_root/tools/student_job.py" "$@" ;;
  train-rbq10-multiview)
    exec "$python_bin" "$project_root/tools/train_multiview_student.py" --headless "$@" ;;
  multiview-student-job)
    exec "$python_bin" "$project_root/tools/multiview_student_job.py" "$@" ;;
  teacher-job)
    exec "$python_bin" "$project_root/tools/teacher_job.py" "$@" ;;
  list-rbq10)
    exec "$python_bin" "$project_root/tools/task_entry.py" "$upstream_root/list_envs.py" "$@" ;;
  list)
    exec "$python_bin" "$upstream_root/list_envs.py" "$@" ;;
  probe-student|probe-teacher)
    kind=Student
    if [[ "$mode" == probe-teacher ]]; then kind=Teacher; fi
    exec "$python_bin" "$project_root/tools/probe_go2.py" --kind "$kind" --output "$project_root/logs/$mode.json" "$@" ;;
  *)
    printf 'Usage: %s {play-student|play-teacher|train-teacher|train-student|list|probe-student|probe-teacher|validate-rbq10|teacher-job|list-rbq10|play-rbq10-teacher|play-rbq10-student|train-rbq10-teacher|train-rbq10-student} [upstream arguments]\n' "$0" >&2
    exit 2 ;;
esac
