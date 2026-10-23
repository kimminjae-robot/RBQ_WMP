"""CPU comparison of MuJoCo depth against the installed Isaac Lab ray-camera code.

Executes the actual pinhole, intrinsics, Warp raycast kernel and original parkour
preprocessing. This is a source-pipeline comparison, not a second live Isaac Sim
process or a proof of GPU/PhysX dynamics equivalence. Training remains untouched.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace as NS

import mujoco
import numpy as np
import torch
import torchvision.transforms as T
import trimesh
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim2sim'))
from build_scene import build_scene,ASSETS
from student_inputs import DepthInput,student_observation
from run_teacher import ObservationBridge
from terrains import generate,SOURCE


def functions(path,names=None,namespace=None,class_name=None):
    tree=ast.parse(path.read_text())
    if class_name:tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and (names is None or n.name in names)]
    for node in nodes:node.decorator_list=[]
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[]))
    ns={} if namespace is None else namespace
    exec(compile(module,str(path),'exec'),ns)
    return ns


def stats(x):
    x=np.abs(np.asarray(x))
    return {'mae':float(x.mean()),'max':float(x.max()),'p99':float(np.quantile(x,.99))}


def original_mesh(model,data,course,terrain_functions):
    if course.startswith('isaac_'):
        terrain=generate(course,.5,42)
        raw=np.rint(terrain['heights']/terrain['vertical_scale']).astype(np.int16)
        vertices,faces,_=terrain_functions['convert_height_field_to_mesh'](raw,terrain['spacing'],terrain['vertical_scale'],terrain['slope_threshold'])
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces)
        mesh=mesh.simplify_quadric_decimation(face_count=int(.65*len(faces)),aggression=3)
        mesh.vertices[:,:2]+=terrain['origin_xy']
        return mesh
    meshes=[]
    for i in range(model.ngeom):
        if model.geom_group[i]!=0:continue
        kind=model.geom_type[i]
        if kind==mujoco.mjtGeom.mjGEOM_PLANE:
            mesh=trimesh.Trimesh(vertices=[[-50,-50,0],[50,-50,0],[50,50,0],[-50,50,0]],faces=[[0,1,2],[0,2,3]],process=False)
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            mesh=trimesh.creation.box(extents=2*model.geom_size[i])
        else:raise ValueError(f'Unexpected analytic geometry {kind}')
        mesh.vertices[:]=mesh.vertices@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir',type=Path,required=True)
    p.add_argument('--isaaclab-source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    source=a.isaaclab_source
    if source is None:
        spec=importlib.util.find_spec('isaaclab')
        if spec is None:p.error('Provide --isaaclab-source /path/to/IsaacLab/source/isaaclab/isaaclab; CPU MuJoCo venv may not install Isaac Lab')
        source=Path(spec.origin).parent
    sources={'camera':source/'sensors/ray_caster/ray_caster_camera.py',
             'pattern':source/'sensors/ray_caster/patterns/patterns.py',
             'math':source/'utils/math.py','ops':source/'utils/warp/ops.py','kernels':source/'utils/warp/kernels.py',
             'preprocessing':ROOT/'upstream/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/observations.py',
             'terrain':SOURCE/'utils.py'}
    wp.config.quiet=True;wp.init()
    spec=importlib.util.spec_from_file_location('_rbq_transfer_isaac_kernels',sources['kernels'])
    kernels=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernels;spec.loader.exec_module(kernels)
    ns={'torch':torch,'np':np,'math':math,'wp':wp,'kernels':kernels}
    math_ops=functions(sources['math'],namespace=ns.copy())
    intrinsic=functions(sources['camera'],['_compute_intrinsic_matrices'],ns.copy(),'RayCasterCamera')['_compute_intrinsic_matrices']
    pattern=functions(sources['pattern'],['pinhole_camera_pattern'],ns.copy())['pinhole_camera_pattern']
    warp_ops=functions(sources['ops'],['raycast_mesh','convert_to_warp_mesh'],ns.copy())
    prep=functions(sources['preprocessing'],['reset','__call__','_process_depth_image','_crop_depth_image','_normalize_depth_image'],ns.copy(),'image_features')
    terrain_fns=functions(sources['terrain'],['convert_height_field_to_mesh'],{'np':np})
    meta=json.loads((a.policy_dir/'manifest.json').read_text());contract=meta['contract']
    config=json.loads((ROOT/'sim2sim/student_config.json').read_text());camcfg=config['camera']
    pat=NS(**{k:camcfg[k] for k in ['width','height','focal_length','horizontal_aperture','vertical_aperture']},horizontal_aperture_offset=0.,vertical_aperture_offset=0.)
    intrinsic_data=NS(cfg=NS(pattern_cfg=pat),_data=NS(intrinsic_matrices=torch.eye(3)[None]))
    intrinsic(intrinsic_data);K=intrinsic_data._data.intrinsic_matrices
    _,source_rays=pattern(pat,K,'cpu')
    qworld=math_ops['convert_camera_frame_orientation_convention'](torch.tensor([camcfg['quaternion_wxyz_ros']]),origin='ros',target='world')
    R_offset=math_ops['matrix_from_quat'](qworld)[0].numpy()
    home_info=json.loads((ASSETS/'official/info.json').read_text())['config_info']['init_state']['default_joint_angles']
    home=np.array([home_info[name] for name in contract['joint_order']])
    actor=torch.jit.load(str(a.policy_dir/'student.pt'),map_location='cpu').eval()
    ref_encoder=torch.jit.load(str(a.policy_dir/'depth.pt'),map_location='cpu').eval()
    mj_encoder=torch.jit.load(str(a.policy_dir/'depth.pt'),map_location='cpu').eval()
    report={'scope':'CPU execution of installed Isaac source ray camera, Warp kernel and original preprocessing. Analytic scenes isolate pipeline; source simplified terrain compares geometry. Not live GPU camera/PhysX parity.',
            'checkpoint':meta['checkpoint'],'checkpoint_sha256':meta['checkpoint_sha256'],
            'source_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in sources.items()},
            'warp_device':'cpu','policy_device':'cpu','source_intrinsics':K[0].tolist(),'cases':[]}
    a.output.mkdir(parents=True,exist_ok=True)
    poses=[(0.,0.,.55,0.,0.,0.),(.18,.03,.52,.08,-.1,.12),(.4,-.04,.6,-.1,.12,-.2),(.65,0.,.5,0.,-.06,0.)]
    for course in ['flat','hurdle','step','isaac_platforms','isaac_gap','isaac_mixed']:
        model=mujoco.MjModel.from_xml_string(build_scene(contract,course,height=.25,difficulty=.5,terrain_seed=42,contact_profile='firm'))
        data=mujoco.MjData(model)
        bridge=ObservationBridge(model,data,contract,home,course=='flat',.6);bridge.terrain_scan_enabled=False
        data.qpos[bridge.qids]=home;mujoco.mj_forward(model,data)
        mesh=original_mesh(model,data,course,terrain_fns)
        reference_mesh=warp_ops['convert_to_warp_mesh'](np.asarray(mesh.vertices),np.asarray(mesh.faces),'cpu')
        camera=DepthInput(model,data,camcfg)
        source_local=source_rays[0].numpy()@R_offset.T
        mj_local=camera.rays@camera.rotation.T
        ray_error=float(abs(source_local-mj_local).max())
        assert ray_error<2e-6,ray_error
        processed_ref=NS(num_envs=1,device='cpu',clipping_range=camcfg['max_distance_m'],buffer_len=2,debug_vis=False,
                         depth_buffer=torch.zeros(1,2,58,87),resize_transform=T.Resize((58,87),interpolation=T.InterpolationMode.BICUBIC))
        for name in ['_process_depth_image','_crop_depth_image','_normalize_depth_image']:setattr(processed_ref,name,MethodType(prep[name],processed_ref))
        ref_encoder.hidden_states=torch.zeros(1,0,512);mj_encoder.hidden_states=torch.zeros(1,0,512)
        samples=[];arrays={}
        for tick,pose in enumerate(poses):
            x,y,z,roll,pitch,yaw=pose
            data.qpos[:3]=[x+(1. if course.startswith('isaac_') else 0.),y,z]
            data.qpos[3:7]=math_ops['quat_from_euler_xyz'](*[torch.tensor([v]) for v in [roll,pitch,yaw]])[0].numpy()
            data.qvel[:]=0;mujoco.mj_forward(model,data)
            R=data.xmat[bridge.base].reshape(3,3)
            origin=data.xpos[bridge.base]+R@np.asarray(camcfg['position_m'])
            directions=(source_local@R.T).astype(np.float32)
            starts=torch.from_numpy(np.repeat(origin[None],len(directions),axis=0).astype(np.float32))[None]
            _,distance,_,_=warp_ops['raycast_mesh'](starts,torch.from_numpy(directions)[None],reference_mesh,return_distance=True)
            raw_ref=distance.clamp(max=camcfg['max_distance_m']).reshape(camcfg['height'],camcfg['width'])
            processed_ref.camera_sensor=NS(data=NS(output={'distance_to_camera':raw_ref[None,:,:,None]}))
            if tick==0:prep['reset'](processed_ref)
            delayed_ref=prep['__call__'](processed_ref,NS(common_step_counter=tick*5,device="cpu"),None,(58,87),2,False)
            delayed_mj=camera.sample()
            raw_mj=camera.latest_raw
            same_preprocess=processed_ref._process_depth_image(torch.from_numpy(raw_mj.copy()))
            # Compare normalization/resize on identical raw input, independent of geometry.
            import torch.nn.functional as F
            mj_current=F.interpolate(torch.from_numpy(raw_mj[:-2,4:-4].copy())[None,None],size=(58,87),mode='bicubic',align_corners=False,antialias=True)[0,0]/2-.5
            preprocess_error=float((same_preprocess-mj_current).abs().max())
            assert preprocess_error<1e-6
            obs,_=bridge.sample(np.zeros(12,dtype=np.float32))
            cfg=dict(config,terrain_label='flat' if course=='flat' else 'parkour',direction_source='joystick')
            obs=student_observation(obs,.6,0.,np.zeros(3),cfg)
            prop=torch.from_numpy(obs[None,:53].copy());prop[:,6:8]=0
            with torch.inference_mode():
                latent_ref=ref_encoder(delayed_ref,prop)[:,:32]
                latent_mj=mj_encoder(delayed_mj,prop)[:,:32]
                action_ref=actor(torch.from_numpy(obs[None]),latent_ref)
                action_mj=actor(torch.from_numpy(obs[None]),latent_mj)
            raw_error=raw_mj-raw_ref.numpy();visible=(raw_mj<1.999)|(raw_ref.numpy()<1.999)
            sample={'policy_step':tick*5,'pose_xyz_rpy':data.qpos[:3].tolist()+list(pose[3:]),
                    'raw_depth_m':stats(raw_error),'visible_raw_depth_m':stats(raw_error[visible]) if visible.any() else None,
                    'fraction_pixels_error_over_2cm':float((abs(raw_error)>.02).mean()),
                    'preprocess_same_raw_max':preprocess_error,'delayed_depth':stats((delayed_ref-delayed_mj).numpy()),
                    'latent':stats((latent_ref-latent_mj).numpy()),'raw_action':stats((action_ref-action_mj).numpy()),
                    'scope':'Same proprioception, different ray/geometry pipeline; latent/action sensitivity is not a rollout success rate'}
            assert np.isfinite(raw_error).all()
            if not course.startswith('isaac_'):assert sample['raw_depth_m']['max']<2e-4,sample
            samples.append(sample)
            arrays[f'source_{tick}']=raw_ref.numpy();arrays[f'mujoco_{tick}']=raw_mj.copy()
        np.savez_compressed(a.output/(course+'.npz'),**arrays)
        report['cases'].append({'course':course,'ray_direction_max_abs_error':ray_error,'samples':samples})
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print('DEPTH_CHECK',course,'max_m',max(s['raw_depth_m']['max'] for s in samples),'mae_m',np.mean([s['raw_depth_m']['mae'] for s in samples]),flush=True)
    report['completed']=True
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
