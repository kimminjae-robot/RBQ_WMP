"""Compare actual Teacher/Student on identical fixed-level Isaac terrain and inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for directory in [ROOT,ROOT/"upstream/Isaaclab_Parkour",ROOT/"upstream/Isaaclab_Parkour/parkour_tasks"]:
    sys.path.insert(0,str(directory))

from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--teacher',type=Path,required=True)
p.add_argument('--student',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--levels',type=int,nargs='+',default=[4])
p.add_argument('--roles',choices=['teacher','student'],nargs='+',default=['student'])
p.add_argument('--num_envs',type=int,default=20)
p.add_argument('--episodes',type=int,default=3)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--speed',type=float,default=.6)
p.add_argument('--video',type=Path)
p.add_argument('--video-env',type=int,help='Optional explicit environment index of the chosen terrain')
p.add_argument('--video-terrain',choices=['parkour','parkour_hurdle','parkour_gap','parkour_step','parkour_flat'],default='parkour')
p.add_argument('--trace',type=Path)
p.add_argument('--record-seconds',type=float,default=20.)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.video:a.enable_cameras=True
if a.num_envs%10 or a.num_envs<10 or any(l<0 or l>9 for l in a.levels):p.error('Use num_envs divisible by 10 and levels 0..9')
if len(a.levels)*len(a.roles)!=1:p.error('Run one role/level per process; use separate invocations for independent Isaac contexts')
a.output.parent.mkdir(parents=True,exist_ok=True)
app=AppLauncher(a).app


def main():
    import numpy as np
    import torch
    from isaaclab.utils.math import euler_xyz_from_quat,wrap_to_pi
    from rbq10_parkour.config import RBQ10StudentEnvCfg
    from rbq10_parkour.evaluation import fixed_level_config,FixedLevelEvaluationEnv
    from rbq10_parkour.contract import check_checkpoint_contract,robot_contract
    from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper
    from export_sim2sim import load_original_actor,load_original_student
    cfg=RBQ10StudentEnvCfg()
    reference=fixed_level_config(a.levels[0],a.num_envs,a.seed)
    cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device
    cfg.scene.terrain=reference.scene.terrain
    gen=cfg.scene.terrain.terrain_generator
    gen.horizontal_scale=.1
    for name,sub in gen.sub_terrains.items():
        sub.use_simplified=True;sub.horizontal_scale=.1
        if name!='parkour':sub.y_range=(-.1,.1)
    cfg.parkours=reference.parkours;cfg.commands=reference.commands
    cfg.commands.base_velocity.ranges.lin_vel_x=(a.speed,a.speed)
    cfg.episode_length_s=20.
    cfg.events.push_by_setting_velocity=None
    cfg.events.randomize_rigid_body_mass=None;cfg.events.randomize_rigid_body_com=None
    cfg.events.random_camera_position=None
    cfg.events.physics_material.params['friction_range']=(.9,.9)
    cfg.training_metrics_enabled=False
    cfg.viewer=reference.viewer
    if a.video:
        cfg.viewer.eye=(.5,3.2,1.6)
        cfg.viewer.lookat=(.3,0.,.4)
    raw=FixedLevelEvaluationEnv(cfg=cfg,render_mode='rgb_array' if a.video else None)
    env=ParkourRslRlVecEnvWrapper(raw)
    tc=torch.load(a.teacher,map_location='cpu',weights_only=False)
    sc=torch.load(a.student,map_location='cpu',weights_only=False)
    check_checkpoint_contract(tc['robot_contract'],robot_contract('teacher'))
    check_checkpoint_contract(sc['robot_contract'],robot_contract('student'))
    teacher,_=load_original_actor(tc);teacher.to(raw.device)
    student,depth=load_original_student(sc);student.to(raw.device);depth.to(raw.device)
    robot=raw.scene['robot'];parkour=raw.parkour_manager.get_term('base_parkour')
    report={'teacher':a.teacher.name,'student':a.student.name,
        'teacher_sha256':hashlib.sha256(a.teacher.read_bytes()).hexdigest(),
        'student_sha256':hashlib.sha256(a.student.read_bytes()).hexdigest(),
        'conditions':{'seed':a.seed,'num_envs':a.num_envs,'episodes_per_env':a.episodes,'terrain_profile':'Student horizontal_scale=0.1/simplified, original five types equally represented',
        'velocity_input':'Both measured body velocity (matches distillation input contract)',
        'direction_input':'Both commanded goal heading; no vision direction replacement',
        'action_delay_policy_steps':1,'vx_command_m_s':a.speed,'random_push_mass_com_camera':False,
        'scope':'Nominal single-seed Isaac comparison; not MuJoCo/hardware or confidence interval'},'results':[]}
    try:
        for level in a.levels:
            for role in a.roles:
                counts=np.zeros(raw.num_envs,dtype=int);records=[]
                peaks=torch.zeros(raw.num_envs,device=raw.device)
                raw.episode_end_observer=None
                parkour.cfg.fixed_terrain_level=level
                raw.common_step_counter=0
                raw.reset(seed=a.seed)
                obs,extras=env.get_observations()
                names=parkour.env_per_terrain_name.reshape(-1).copy()
                depth.hidden_states=torch.zeros(1,0,512,device=raw.device)
                def sample():
                    roll,pitch,_=euler_xyz_from_quat(robot.data.root_quat_w)
                    peaks.copy_(torch.maximum(peaks,torch.maximum(wrap_to_pi(roll).abs(),wrap_to_pi(pitch).abs())))
                def finish(ids):
                    sample()
                    for i in ids.cpu().tolist():
                        if counts[i]>=a.episodes:peaks[i]=0;continue
                        assert int(raw.scene.terrain.terrain_levels[i])==level
                        goal=int(parkour.cur_goal_idx[i])
                        fall=bool(peaks[i]>1.5 or robot.data.root_pos_w[i,2]<-.25)
                        result='fall' if fall else 'success' if goal>=parkour.num_goals else 'timeout'
                        records.append({'env_id':i,'episode':int(counts[i]),'terrain':str(names[i]),'outcome':result,'goals':goal})
                        counts[i]+=1;peaks[i]=0
                raw.episode_end_observer=finish
                tick=0
                trace=[]
                writer=None
                if a.video:
                    import cv2
                    a.video.parent.mkdir(parents=True,exist_ok=True)
                    writer=cv2.VideoWriter(str(a.video),cv2.VideoWriter_fourcc(*'mp4v'),25.,cfg.viewer.resolution)
                    assert writer.isOpened()
                    camera_id=a.video_env if a.video_env is not None else int(np.where(names==a.video_terrain)[0][0])
                    assert names[camera_id]==a.video_terrain
                    video_states=[]
                    foot_ids,_=raw.scene['contact_forces'].find_bodies(['HR_foot','HL_foot','FR_foot','FL_foot'],preserve_order=True)
                    raw.viewport_camera_controller.set_view_env_index(camera_id)
                with torch.inference_mode():
                    while not (counts>=a.episodes).all() and tick<a.episodes*raw.max_episode_length+50:
                        assert bool((raw.scene.terrain.terrain_levels==level).all())
                        sample()
                        if role=='teacher':
                            action=teacher(obs,hist_encoding=True)
                        else:
                            if raw.common_step_counter%5==0 or tick==0:
                                prop=obs[:,:53].clone();prop[:,6:8]=0
                                latent=depth(extras['observations']['depth_camera'],prop)[:,:32]
                            action=student(obs,latent)
                        if a.trace and tick<200:
                            i=0
                            row={}
                            for key in ['root_link_pos_w','root_link_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel','body_pos_w','joint_pos_limits','joint_vel_limits','applied_torque']:
                                row[key]=getattr(robot.data,key)[i].detach().cpu().numpy()
                            row['obs']=obs[i].cpu().numpy();row['action']=action[i].cpu().numpy()
                            row['body_names']=np.array(robot.body_names)
                            row['origin']=raw.scene.env_origins[i].cpu().numpy()
                            trace.append(row)
                        if not torch.isfinite(action).all():raise FloatingPointError('Nonfinite evaluation action')
                        obs,_,_,extras=env.step(action);tick+=1
                        if writer is not None and tick%2==0 and tick*raw.step_dt<=a.record_seconds:
                            force=raw.scene['contact_forces'].data.net_forces_w[camera_id,foot_ids].norm(dim=-1)
                            contacts=(force>2).cpu().numpy()
                            airborne=not bool(contacts.any())
                            video_states.append({'t':tick*raw.step_dt,'airborne':airborne,'foot_contacts':contacts.tolist(),'base_z':float(robot.data.root_pos_w[camera_id,2]),'goals':int(parkour.cur_goal_idx[camera_id])})
                            frame=cv2.cvtColor(raw.render(),cv2.COLOR_RGB2BGR)
                            cv2.rectangle(frame,(0,0),(960,58),(20,20,20),-1)
                            cv2.putText(frame,f'Isaac Sim | RBQ10 Student {a.student.stem} | {a.video_terrain} level {level}/9',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                            cv2.putText(frame,f'Actual simulation {tick*raw.step_dt:.2f}s | '+('AIRBORNE (all feet off contact)' if airborne else 'GROUND CONTACT'),(12,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1,cv2.LINE_AA)
                            writer.write(frame)
                if writer is not None:
                    writer.release()
                    a.video.with_suffix('.states.json').write_text(json.dumps(video_states,indent=2)+'\n')
                if a.trace:
                    a.trace.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(a.trace,**{k:np.stack([r[k] for r in trace]) for k in trace[0]})
                if not (counts>=a.episodes).all():raise RuntimeError('Incomplete scored episodes')
                summaries={}
                for name in sorted(set(names.tolist())):
                    rows=[r for r in records if r['terrain']==name]
                    summaries[name]={'episodes':len(rows),**{k:sum(x['outcome']==k for x in rows) for k in ['success','fall','timeout']}}
                successes=sum(r['outcome']=='success' for r in records)
                result={'role':role,'level':level,'episodes':len(records),'successes':successes,'success_rate':successes/len(records),'summary':summaries,'records':records}
                report['results'].append(result)
                a.output.write_text(json.dumps(report,indent=2)+'\n')
                print('COMPARISON_RESULT',role,level,successes,'/',len(records),flush=True)
    finally:
        raw.episode_end_observer=None
        env.close()


try:main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:app.close()
