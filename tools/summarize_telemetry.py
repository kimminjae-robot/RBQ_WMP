"""Summarize measured Go2 replay data without inferring a task success rate."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
rows = [json.loads(line) for line in args.input.read_text().splitlines()]
if not rows:
    raise SystemExit("No simulation measurements found")
summary = {
    "source": str(args.input),
    "sample_count": len(rows),
    "sample_interval_policy_steps": 10,
    "last_sample_policy_step": rows[-1]["step"],
    "last_sample_time_s": rows[-1]["time_s"],
    "all_sampled_actions_finite": all(row["actions_finite"] for row in rows),
    "note": "Sampled progress and contact measurements; no formal episode success rate is inferred.",
    "environments": [],
}
for env in range(len(rows[0]["terrain"])):
    settled = [row for row in rows if row["time_s"] > 1.0]
    if not settled:
        settled = rows
    summary["environments"].append({
        "env_index": env,
        "terrain": rows[0]["terrain"][env],
        "max_sampled_waypoint_index": max(row["goal_index"][env] for row in rows),
        "min_base_z_world_m": min(row["root_pos_w"][env][2] for row in settled),
        "max_base_z_world_m": max(row["root_pos_w"][env][2] for row in settled),
        "observed_reset_intervals": sum(
            row["episode_length"][env] < previous["episode_length"][env]
            for previous, row in zip(rows, rows[1:])
        ),
        "all_feet_below_2N_sample_times_s": [
            row["time_s"] for row in settled if max(row["foot_force_norm"][env]) < 2.0
        ],
    })
args.output.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "environments"}, indent=2))
