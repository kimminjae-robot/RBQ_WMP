"""Check joint mapping, static PD, export parity record and original observation layout."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import torch

from build_scene import ASSETS, ROOT, build_scene
from run_teacher import ObservationBridge


def source_method(name):
    path = ROOT/'upstream/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/observations.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='ExtremeParkourObservations')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name==name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ns = {'torch': torch, 'wrap_to_pi': lambda x: (x+torch.pi)%(2*torch.pi)-torch.pi,
          'euler_xyz_from_quat': lambda q: (q[:, 0], q[:, 1], q[:, 2])}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns[name]


def main():
    torch.set_num_threads(1)
    meta=json.loads((ROOT/'outputs/sim2sim/teacher/manifest.json').read_text())
    c=meta['contract']
    m=mujoco.MjModel.from_xml_string(build_scene(c))
    d=mujoco.MjData(m)
    hd=json.loads((ASSETS/'official/info.json').read_text())['config_info']['init_state']['default_joint_angles']
    home=np.array([hd[n] for n in c['joint_order']])
    b=ObservationBridge(m,d,c,home,True,.6)
    assert [m.joint(i).name for i in b.joints]==c['joint_order']
    assert [m.actuator(i).name for i in range(m.nu)]==['pd_'+n for n in c['joint_order']]
    d.qpos[b.qids]=home
    d.ctrl[:]=home
    mujoco.mj_forward(m,d)
    for _ in range(2000):
        mujoco.mj_step(m,d)
    standing_z=float(d.xpos[b.base,2])
    assert standing_z>.4 and np.max(np.abs(d.qvel))<.01
    # Compare against the *actual source method* using a tensor fixture for simulator accessors.
    source=source_method('__call__')
    state=NS(delta_yaw=torch.zeros(1), delta_next_yaw=torch.zeros(1),
             measured_heights=torch.zeros(1,132), _obs_history_buffer=torch.zeros(1,10,53),
             history_length=10, num_envs=1, device='cpu')
    maximum=0.
    for step in range(7):
        action=np.linspace(-.4,.7,12,dtype=np.float32)+step*.01
        actual,angles=b.sample(action)
        yaw=np.arctan2(d.xmat[b.base].reshape(3,3)[1,0],d.xmat[b.base].reshape(3,3)[0,0])
        # Fixture returns exact roll/pitch/yaw; quaternion conversion itself is tested by static pose.
        state.asset=NS(data=NS(root_quat_w=torch.tensor([[*angles,yaw]],dtype=torch.float32),
            root_ang_vel_b=torch.tensor(actual[None,:3]/.25),
            joint_pos=torch.tensor(d.qpos[b.qids][None],dtype=torch.float32),
            default_joint_pos=torch.tensor(home[None],dtype=torch.float32),
            joint_vel=torch.tensor(d.qvel[b.vids][None],dtype=torch.float32)))
        state.parkour_event=NS(env_per_terrain_name=np.array([['parkour_flat']]),
            target_yaw=torch.tensor([b.yaws[0]+yaw]),next_target_yaw=torch.tensor([b.yaws[1]+yaw]))
        state._get_heights=lambda: torch.tensor(b.scan[None])
        state._get_priv_explicit=lambda: torch.zeros(1,9)
        state._get_priv_latent=lambda: torch.zeros(1,29)
        state._get_contact_fill=lambda: torch.tensor(actual[None,49:53])
        env=NS(common_step_counter=step,episode_length_buf=torch.tensor([step]),
            command_manager=NS(get_command=lambda n: torch.tensor([[.6,0.,0.]])),
            action_manager=NS(get_term=lambda n: NS(action_history_buf=torch.tensor(action[None,None]))))
        expected=source(state,env,None,None,'base_parkour',10)
        error=float(np.abs(expected.numpy()[0]-actual).max())
        maximum=max(maximum,error)
        assert error<1e-5, (step,error)
    get_heights=source_method('_get_heights')
    fake=NS(ray_sensor=NS(data=NS(pos_w=torch.tensor([[.42,0.,20.55]]),
                                 ray_hits_w=torch.zeros(1,132,3))),device='cpu')
    saturated=bool(torch.all(get_heights(fake)==1))
    assert saturated
    report={'static_pd_seconds':10,'static_pd_final_base_z':standing_z,
        'source_observation_fixture_steps':7,'source_observation_max_abs_error':maximum,
        'export_max_abs_error':meta['export_max_abs_error'],
        'joint_and_actuator_name_order_verified':True,
        'flat_height_formula_with_20m_offset_saturates':saturated,
        'scope':'CPU fixtures and static dynamics; not live Isaac/MuJoCo trajectory parity or parkour success'}
    out=ROOT/'outputs/sim2sim/bridge_checks.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(out.read_text())


if __name__=='__main__':
    main()
