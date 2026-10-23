"""Compare MuJoCo gains/torque law with the actual RBQ10 Isaac actuator config.

No Isaac process. Stores an open-loop CPU MuJoCo trajectory for future PhysX
replay; matching config/instantaneous law does not prove implicit solver parity.
"""
import argparse,ast,json,re,sys
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim2sim'))
from build_scene import build_scene,ASSETS
from run_teacher import ObservationBridge


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    contract=json.loads(a.manifest.read_text())['contract']
    tree=ast.parse((ROOT/'rbq10_parkour/robot.py').read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RBQ10_CFG' for t in n.targets))
    actuators=next(k.value for k in assignment.value.keywords if k.arg=='actuators')
    groups=[]
    for call in actuators.values:
        groups.append({k.arg:ast.literal_eval(k.value) for k in call.keywords})
    expected=[]
    for name in contract['joint_order']:
        matches=[g for g in groups if any(re.fullmatch(pattern,name) for pattern in g['joint_names_expr'])]
        assert len(matches)==1
        expected.append(matches[0])
    m=mujoco.MjModel.from_xml_string(build_scene(contract,'flat',contact_profile='firm'))
    d=mujoco.MjData(m)
    hd=json.loads((ASSETS/'official/info.json').read_text())['config_info']['init_state']['default_joint_angles']
    home=np.array([hd[n] for n in contract['joint_order']])
    b=ObservationBridge(m,d,contract,home,True,.6)
    kp=np.array([g['stiffness'] for g in expected]);kd=np.array([g['damping'] for g in expected])
    limits=np.array([g['effort_limit_sim'] for g in expected]);velocity_limits=np.array([g['velocity_limit_sim'] for g in expected])
    armature=np.array([g['armature'] for g in expected])
    assert np.allclose(m.actuator_gainprm[:,0],kp)
    assert np.allclose(m.actuator_biasprm[:,1],-kp) and np.allclose(m.actuator_biasprm[:,2],-kd)
    assert np.allclose(m.actuator_forcerange,np.c_[-limits,limits])
    assert np.allclose(m.dof_armature[b.vids],armature)
    errors=[]
    for speed in [0.,5.,45.,60.]:
        mujoco.mj_resetData(m,d);d.qpos[2]=2.;d.qpos[b.qids]=home
        d.qvel[b.vids]=np.linspace(-speed,speed,12)
        d.ctrl[:]=home+np.linspace(-.15,.15,12)
        mujoco.mj_forward(m,d)
        expected_force=np.clip(kp*(d.ctrl-d.qpos[b.qids])-kd*d.qvel[b.vids],-limits,limits)
        error=float(abs(d.actuator_force-expected_force).max());assert error<1e-8
        errors.append({'max_joint_speed_rad_s':speed,'torque_law_max_error_nm':error})
    mujoco.mj_resetData(m,d);d.qpos[2]=2.;d.qpos[b.qids]=home;d.ctrl[:]=home
    mujoco.mj_forward(m,d)
    target=home.copy();target[contract['joint_order'].index('joint7_FRP')]+=.1
    rows={'qpos':[d.qpos.copy()],'qvel':[d.qvel.copy()],'force':[d.actuator_force.copy()],'time':[d.time]}
    for _ in range(40):
        d.ctrl[:]=target;mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        assert d.ncon==0,'Free-space fixture should remain above the floor'
        for name,value in [('qpos',d.qpos),('qvel',d.qvel),('force',d.actuator_force),('time',d.time)]:rows[name].append(np.copy(value))
    np.savez_compressed(a.output/'mujoco_free_space_step.npz',**{k:np.asarray(v) for k,v in rows.items()},target=target,home=home)
    result={'source':'rbq10_parkour/robot.py','joint_order':contract['joint_order'],'kp':kp.tolist(),'kd':kd.tolist(),
            'armature':armature.tolist(),'effort_limit_nm':limits.tolist(),'isaac_velocity_limit_rad_s':velocity_limits.tolist(),
            'config_matches':True,'instantaneous_torque_law':errors,'mujoco_velocity_limit_enforced':False,
            'total_urdf_import_mass_kg':float(m.body_mass.sum()),'open_loop_trace':'mujoco_free_space_step.npz',
            'scope':'Actual Isaac config vs MuJoCo configuration and algebraic torque law. Live PhysX open-loop/contact replay pending; not dynamics equivalence.'}
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
