"""Initialize the upstream Go2 environment without constructing or running a policy."""
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--kind', choices=('Teacher', 'Student'), default='Student')
parser.add_argument('--num_envs', type=int, default=4)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import torch
    import parkour_tasks
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

    tasks = sorted(t for t in gym.registry if t.startswith('Isaac-Extreme-Parkour-'))
    print('REGISTERED_TASKS', json.dumps(tasks), flush=True)
    for task in tasks:
        parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
        agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
        print('CONFIG_OK', task, agent.algorithm.class_name, flush=True)
    task = f'Isaac-Extreme-Parkour-{args.kind}-Unitree-Go2-Play-v0'
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 1
    if args.kind == 'Student':
        cfg.observations.depth_camera.depth_cam.params['debug_vis'] = False
    env = gym.make(task, cfg=cfg)
    obs, extras = env.reset(seed=1)
    report = {
        'task': task, 'registered_tasks': tasks, 'num_envs': args.num_envs,
        'gui': env.unwrapped.sim.has_gui(),
        'physics_dt': env.unwrapped.physics_dt, 'policy_dt': env.unwrapped.step_dt,
        'action_dimension': env.unwrapped.action_manager.total_action_dim,
        'observations': {k: {'shape': list(v.shape), 'finite': bool(torch.isfinite(v).all())} for k, v in obs.items()},
        'joint_names': env.unwrapped.scene['robot'].joint_names,
        'terrain_types': sorted(set(env.unwrapped.scene.terrain.terrain_generator_class.terrain_names.reshape(-1).tolist())),
        'policy_loaded': False, 'policy_steps': 0,
        'note': 'Environment initialization only. No policy, behavior, or task success is tested.'
    }
    assert len(tasks) == 6
    assert report['action_dimension'] == 12
    assert all(x['finite'] for x in report['observations'].values())
    args.output.write_text(json.dumps(report, indent=2))
    print('ENVIRONMENT_INITIALIZATION_OK', json.dumps(report), flush=True)
    for _ in range(30):
        app.update()
    env.close()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    from isaaclab.sim import SimulationContext
    SimulationContext.clear_instance()
    app.close()
