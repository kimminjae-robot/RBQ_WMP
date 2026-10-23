#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
viewer_args=(--viewer)
if [[ "${1:-}" == --headless ]]; then viewer_args=(); shift; fi
exec "$project_root/run.sh" sim2sim-latest \
  --checkpoint "$project_root/checkpoints/rbq10_student/model_941.pt" \
  --autonomous "${viewer_args[@]}" --course isaac_platforms --difficulty 0.5 \
  --terrain-seed 42 --speed 0.6 --seconds 30 \
  --terrain-edges source --contact-profile firm \
  --output "$project_root/outputs/sim2sim/autonomous_demo.json" "$@"
