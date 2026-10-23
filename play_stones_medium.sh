#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$project_root/run.sh" sim2sim-latest --course isaac_platforms --difficulty 0.5 --terrain-seed 42 --terrain-label parkour "$@"
