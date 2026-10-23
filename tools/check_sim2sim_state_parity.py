"""Replay live Isaac states into MuJoCo to verify robot kinematics and velocity frames."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim2sim'))
from build_scene import build_scene
from run_teacher import ObservationBridge,body_velocity
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trace',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
z=np.load(a.trace);c=json.loads(a.manifest.read_text())['contract']
m=mujoco.MjModel.from_xml_string(build_scene(c,'flat'));d=mujoco.MjData(m)
b=ObservationBridge(m,d,c,np.zeros(12),False,.6)
errors=[];old_errors=[];kinematics=[]
for i in range(len(z['joint_pos'])):
    d.qpos[:3]=z['root_link_pos_w'][i]-z['origin'][i]
    d.qpos[3:7]=z['root_link_quat_w'][i]
    d.qpos[b.qids]=z['joint_pos'][i];d.qvel[b.vids]=z['joint_vel'][i]
    mujoco.mj_forward(m,d)
    R=d.xmat[b.base].reshape(3,3)
    d.qvel[3:6]=R.T@z['root_ang_vel_w'][i]
    d.qvel[:3]=z['root_lin_vel_w'][i]-np.cross(z['root_ang_vel_w'][i],R@m.body_ipos[b.base])
    mujoco.mj_forward(m,d)
    expected=np.r_[R.T@z['root_ang_vel_w'][i],R.T@z['root_lin_vel_w'][i]]
    errors.append(np.abs(body_velocity(m,d,b.base)-expected))
    old=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b.base,old,1)
    old_errors.append(np.abs(old-expected))
    kinematics.append(max(np.linalg.norm(d.xpos[m.body(n).id]-(z['body_pos_w'][i,j]-z['origin'][i])) for j,n in enumerate(z['body_names'][i])))
errors=np.array(errors);old_errors=np.array(old_errors)
r={'samples':len(errors),'body_position_max_error_m':max(kinematics),
   'old_angular_velocity_max_error_rad_s':float(old_errors[:,:3].max()),
   'old_linear_velocity_max_error_m_s':float(old_errors[:,3:].max()),
   'fixed_angular_velocity_max_error_rad_s':float(errors[:,:3].max()),
   'fixed_linear_velocity_max_error_m_s':float(errors[:,3:].max()),
   'isaac_joint_limits':z['joint_pos_limits'][0].tolist(),'mujoco_joint_limits':m.jnt_range[b.joints].tolist(),
   'scope':'Identical recorded states; tests frames/kinematics, not contact/dynamics equivalence'}
assert errors.max()<1e-6 and max(kinematics)<1e-5
assert np.allclose(z['joint_pos_limits'][0],m.jnt_range[b.joints],atol=1e-6)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
