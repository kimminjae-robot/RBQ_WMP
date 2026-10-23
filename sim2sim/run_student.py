"""RBQ10 Vision Student Sim2Sim: real depth weights, joystick and measured velocity."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from build_scene import ROOT, ASSETS, build_scene
from terrains import COURSES, generate
from goal_follower import TerrainGoalFollower
from run_teacher import ObservationBridge, body_velocity
from student_inputs import CommandInput, DepthInput, student_observation
from keyboard import KeyboardInput


def validate_export(directory):
    meta = json.loads((directory/'manifest.json').read_text())
    if meta.get('format')!='rbq10_mujoco_student_v1' or meta.get('role')!='student':
        raise ValueError('Student Sim2Sim requires a real RBQ10 Student export; Teacher exports are rejected')
    for name in ['student.pt','depth.pt']:
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=meta['files'][name]:
            raise ValueError(f'Export hash mismatch: {name}')
    contract=meta['contract']
    if contract.get('robot')!='Rainbow_RBQ10_without_arm':
        raise ValueError('Go2 exports cannot be deployed on RBQ10')
    if hashlib.sha256((ASSETS/'resources/urdf/rbq10.urdf').read_bytes()).hexdigest()!=contract['urdf_sha256']:
        raise ValueError('URDF does not match the Student checkpoint')
    if contract['policy_dt']!=.02 or contract['physics_dt']!=.005:
        raise ValueError('Unsupported timing contract')
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir',type=Path,default=ROOT/'outputs/sim2sim/student')
    p.add_argument('--config',type=Path,default=ROOT/'sim2sim/student_config.json')
    p.add_argument('--course',choices=['flat','hurdle','step',*COURSES],default='isaac_flat')
    p.add_argument('--difficulty',type=float,default=.5)
    p.add_argument('--terrain-seed',type=int,default=42)
    p.add_argument('--terrain-edges',choices=['source','hfield'],default='source')
    p.add_argument('--contact-profile',choices=['default','firm'],default='firm')
    p.add_argument('--height',type=float,default=.10)
    p.add_argument('--seconds',type=float,default=None)
    p.add_argument('--speed',type=float,default=None,help='Reference vx m/s; original training is forward walking')
    p.add_argument('--yaw-rate',type=float,default=0.,help='Reference body yaw rate rad/s')
    p.add_argument('--joystick',help='Linux joystick device, e.g. /dev/input/js0; axis1 forward, axis0 turn')
    p.add_argument('--input-json',type=Path,help='Live reference command and optional measured body velocity')
    p.add_argument('--terrain-label',choices=['flat','parkour'],help='Manual label for known experiment layout, not automatic terrain truth')
    p.add_argument('--velocity-source',choices=['mujoco','external'])
    p.add_argument('--direction-source',choices=['joystick','vision'])
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--hold-seconds',type=float,default=3.,help='Hold final autonomous viewer frame without advancing physics')
    p.add_argument('--autonomous',action='store_true',help='Use original terrain current/next waypoint headings; no keyboard required')
    p.add_argument('--keyboard',action='store_true',help='Viewer key presses: W/S speed, A/D turn, space stop, R reset, Q quit')
    p.add_argument('--video',type=Path)
    p.add_argument('--screenshot',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/sim2sim/student_rollout.json')
    args=p.parse_args()
    args.seconds=args.seconds if args.seconds is not None else (3600. if args.keyboard else 30. if args.autonomous else 10.)
    args.speed=args.speed if args.speed is not None else (0. if args.keyboard else .6)
    if args.autonomous:
        if args.keyboard or args.joystick or args.input_json:p.error('Autonomous terrain goals cannot be combined with manual input')
        if not args.course.startswith('isaac_'):p.error('Autonomous mode requires an isaac_* course with original goals')
        if args.speed<=0:p.error('Autonomous mode requires positive speed')
    if args.keyboard:
        args.viewer=True
        if args.joystick:p.error('Choose keyboard or joystick')
    if args.seconds<=0 or args.height<=0 or not 0<=args.difficulty<=1:
        p.error('seconds and height must be positive; difficulty must be in [0,1]')
    if not (args.policy_dir/'manifest.json').is_file():
        p.error('No RBQ10 Student export. Train/distill a real Student, then run export-sim2sim --checkpoint STUDENT.pt. No Teacher fallback.')
    meta=validate_export(args.policy_dir)
    cfg=json.loads(args.config.read_text())
    for config_key, contract_key in [('position_m','camera_position_m'),('quaternion_wxyz_ros','camera_quaternion_wxyz_ros')]:
        if contract_key not in meta['contract'] or not np.allclose(cfg['camera'][config_key],meta['contract'][contract_key],atol=1e-7):
            p.error('Camera pose differs from the trained Student contract; use the matching camera configuration')
    for key in ['terrain_label','velocity_source','direction_source']:
        value=getattr(args,key)
        if value is not None:
            cfg[key]=value
    if args.autonomous:
        cfg['direction_source']='terrain_goals'
        cfg['terrain_label']='flat' if args.course=='isaac_flat' else 'parkour'
    if cfg['terrain_label'] not in ['flat','parkour'] or cfg['velocity_source'] not in ['mujoco','external'] or cfg['direction_source'] not in ['joystick','vision','terrain_goals']:
        p.error('Invalid Student input configuration')
    if cfg['depth_update_policy_steps']!=5 or cfg['action_delay_policy_steps']!=1:
        p.error('This runner retains original Student timing: depth 5 steps, action delay 1 step')
    if cfg['yaw_rate_to_error_seconds']<=0:
        p.error('yaw_rate_to_error_seconds must be positive')
    if cfg['velocity_source']=='external' and not args.input_json:
        p.error('External measured velocity requires --input-json')
    torch.set_num_threads(1)
    command=CommandInput(args.speed,args.yaw_rate,args.input_json,args.joystick)
    # Validate command/measurement before opening a simulator or viewer.
    try:
        initial=command.read()
        if cfg['velocity_source']=='external' and initial[2] is None:
            raise ValueError('External measured_velocity_body_m_s is required; reference vx cannot substitute for it')
        c=meta['contract']
        model=mujoco.MjModel.from_xml_string(build_scene(c,args.course,args.height,args.difficulty,args.terrain_seed,args.terrain_edges,args.contact_profile,args.autonomous))
        data=mujoco.MjData(model)
        follower=TerrainGoalFollower(generate(args.course,args.difficulty,args.terrain_seed)) if args.autonomous else None
        hd=json.loads((ASSETS/'official/info.json').read_text())['config_info']['init_state']['default_joint_angles']
        home=np.array([hd[n] for n in c['joint_order']])
        bridge=ObservationBridge(model,data,c,home,cfg['terrain_label']=='flat',args.speed)
        bridge.terrain_scan_enabled=False  # Student has no Teacher height scan; optional goal commands are supplied separately.
        data.qpos[bridge.qids]=home
        mujoco.mj_forward(model,data)
        camera=DepthInput(model,data,cfg['camera'])
        policy=torch.jit.load(str(args.policy_dir/'student.pt'),map_location='cpu').eval()
        depth=torch.jit.load(str(args.policy_dir/'depth.pt'),map_location='cpu').eval()
        keyboard=KeyboardInput(args.speed,args.yaw_rate) if args.keyboard else None
        viewer=None
        capture=None
        if args.viewer:
            from mujoco import viewer as mj_viewer
            viewer=mj_viewer.launch_passive(model,data,key_callback=keyboard.on_key if keyboard else None)
            viewer.cam.distance=3.4;viewer.cam.azimuth=125;viewer.cam.elevation=-24
        if args.video or args.screenshot:
            from visualization import Capture
            capture=Capture(model,args.video,args.screenshot)
        if keyboard:print('KEYBOARD: W/S speed +/-0.1; A/D turn +/-0.2; SPACE zero; R reset; Q/ESC close. Click the viewer first.',flush=True)
        records=[]; action=np.zeros(12,dtype=np.float32)
        termination='duration'; max_torque=0.; max_speed=0.
        start=data.xpos[bridge.base].copy()
        initial_qpos=data.qpos.copy()
        paused_on_fall=False;episode=0;policy_step=0;fall_count=0
        try:
            for step in range(int(args.seconds/.02)):
                wall=time.monotonic()
                vx,yaw_rate,external=command.read()
                if keyboard:
                    vx,yaw_rate,reset,quit_requested=keyboard.read()
                    if quit_requested or not viewer.is_running():
                        termination='viewer_closed';break
                    if reset:
                        mujoco.mj_resetData(model,data);data.qpos[:]=initial_qpos;data.ctrl[:]=home
                        mujoco.mj_forward(model,data)
                        bridge=ObservationBridge(model,data,c,home,cfg['terrain_label']=='flat',vx)
                        bridge.terrain_scan_enabled=False
                        camera=DepthInput(model,data,cfg['camera'])
                        depth.hidden_states=torch.zeros(1,0,512)
                        action=np.zeros(12,dtype=np.float32)
                        paused_on_fall=False;episode+=1;policy_step=0
                        print('KEYBOARD RESET',episode,flush=True)
                    if paused_on_fall:
                        viewer.sync();time.sleep(.02);continue
                    if not keyboard.active:
                        viewer.cam.lookat[:]=data.xpos[bridge.base]+np.array([.25,0.,.15])
                        viewer.sync();time.sleep(.02);continue
                measured=body_velocity(model,data,bridge.base)
                measured_velocity=measured[3:] if cfg['velocity_source']=='mujoco' else external
                if measured_velocity is None:
                    raise ValueError('External measured velocity missing; no fallback to reference command')
                bridge.speed=vx
                obs,angles=bridge.sample(action)
                obs=student_observation(obs,vx,yaw_rate,measured_velocity,cfg)
                if follower:
                    rotation=data.xmat[bridge.base].reshape(3,3)
                    yaw=np.arctan2(rotation[1,0],rotation[0,0])
                    previous_goal=follower.index
                    obs[6:8]=follower.update(data.xpos[bridge.base,:2],yaw)
                    if follower.index!=previous_goal:print(f'AUTONOMOUS t={data.time:.2f}s goal {follower.index}/{len(follower.goals)}',flush=True)
                    for marker in range(len(follower.goals)):
                        model.site_rgba[model.site(f'goal_{marker}').id]=([.2,.8,.2,.7] if marker<follower.index else [.1,1.,1.,.9] if marker==follower.index else [.9,.5,.1,.7])
                    if follower.complete:
                        termination='success'
                        if capture:capture.frame(data,bridge.base,meta['checkpoint'],args.course,vx,yaw_rate,f'SUCCESS {follower.index}/{len(follower.goals)}',autonomous=True)
                        break
                with torch.inference_mode():
                    if policy_step%5==0:
                        image=camera.sample()
                        prop=torch.from_numpy(obs[None,:53].copy())
                        prop[:,6:8]=0
                        features=depth(image,prop)
                        latent=features[:,:32]
                        predicted_yaw=1.5*features[:,32:]
                    if cfg['direction_source']=='vision':
                        obs[6:8]=predicted_yaw.numpy()[0]
                    new_action=policy(torch.from_numpy(obs[None]),latent).numpy()[0]
                if not np.isfinite(new_action).all():
                    raise FloatingPointError('Non-finite Student action')
                # Original Student one-policy-step actuation delay.
                data.ctrl[:]=home+.25*action
                action=new_action.copy()
                for _ in range(4):
                    mujoco.mj_step(model,data)
                    max_torque=max(max_torque,float(np.max(np.abs(data.actuator_force))))
                    max_speed=max(max_speed,float(np.max(np.abs(data.qvel[bridge.vids]))))
                mujoco.mj_forward(model,data)  # Derived poses, contacts and velocities now match current qpos/qvel.
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError('Non-finite simulation state')
                records.append({'episode':episode,'time':float(data.time),'base_xyz':data.xpos[bridge.base].tolist(),
                    'reference_vx':vx,'reference_yaw_rate':yaw_rate,'measured_velocity_body_m_s':np.asarray(measured_velocity).tolist(),
                    'actor_direction_errors':obs[6:8].tolist(),'goals_reached':follower.index if follower else None,'foot_contacts':bridge.current_contacts.tolist(),
                    'depth_range_m':[float(camera.latest_raw.min()),float(camera.latest_raw.max())]})
                rotation=data.xmat[bridge.base].reshape(3,3)
                post_angles=(np.arctan2(rotation[2,1],rotation[2,2]),np.arcsin(np.clip(-rotation[2,0],-1,1)))
                fallen=max(map(abs,post_angles))>1.5 or data.xpos[bridge.base,2]<.15
                if capture and (step%2==0 or fallen):
                    capture.frame(data,bridge.base,meta['checkpoint'],args.course,vx,yaw_rate,'FALL - STOPPED' if fallen else f'GOAL {follower.index}/{len(follower.goals)}' if follower else 'RUNNING',autonomous=args.autonomous)
                policy_step+=1
                if fallen:
                    fall_count+=1
                    if keyboard:
                        paused_on_fall=True
                        print('FALL: simulation paused. Press R to reset or Q to quit.',flush=True)
                    else:
                        termination='fall';break
                if viewer:
                    if not viewer.is_running():
                        termination='viewer_closed';break
                    viewer.cam.lookat[:]=data.xpos[bridge.base]+np.array([.25,0.,.15])
                    viewer.sync()
                    time.sleep(max(0,.02-(time.monotonic()-wall)))
        finally:
            if viewer and follower and termination in ['success','fall']:
                until=time.monotonic()+max(0.,args.hold_seconds)
                while viewer.is_running() and time.monotonic()<until:
                    viewer.sync();time.sleep(.02)
            if capture:capture.finish(hold_last=(termination in ['fall','success']))
            if viewer:viewer.close()
        result={'role':'student','checkpoint':meta['checkpoint'],'checkpoint_sha256':meta['checkpoint_sha256'],
            'terrain_profile':'upstream_wrapper_v2', 'terrain_edges':args.terrain_edges,'contact_profile':args.contact_profile,'configuration':cfg,'course':args.course,'height':args.height,'difficulty':args.difficulty,'terrain_seed':args.terrain_seed,'engine':mujoco.__version__,
            'simulated_seconds':float(data.time),'termination':termination,'fall_count':fall_count,'episodes_started':episode+1,'keyboard':args.keyboard,'autonomous':args.autonomous,'goals_reached':follower.index if follower else None,'total_goals':len(follower.goals) if follower else None,'video':str(args.video) if args.video else None,
            'displacement_xyz':(data.xpos[bridge.base]-start).tolist(),
            'max_abs_torque_nm':max_torque,'max_abs_joint_speed_rad_s':max_speed,
            'limitation':'Terrain-goal mode uses known map waypoints; motor velocity cap and complete contact/depth parity remain pending. Single-rollout success is not a transfer success rate.',
            'records':records}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    finally:
        command.close()
    return 2 if args.autonomous and termination!='success' else 0


if __name__=='__main__':
    raise SystemExit(main())
