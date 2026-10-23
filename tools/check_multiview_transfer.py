"""Execute installed Isaac ray/preprocessing code against all five MuJoCo cameras."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import mujoco
import numpy as np
import torch
import torchvision
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"sim2sim"))
from build_scene import build_scene, ASSETS
from multiview_depth import MultiViewDepthInput, add_cameras, validate_configuration
from run_teacher import ObservationBridge
from student_inputs import student_observation
from check_transfer_depth import functions, original_mesh, stats
from terrains import SOURCE


def load_class(path, name, namespace):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[]))
    exec(compile(module,str(path),"exec"),namespace)
    return namespace[name]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--isaaclab-source",type=Path,default=Path("/home/user/isaac/IsaacLab/source/isaaclab/isaaclab"))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);wp.config.quiet=True;wp.init()
    source=a.isaaclab_source
    paths=dict(camera=source/"sensors/ray_caster/ray_caster_camera.py",pattern=source/"sensors/ray_caster/patterns/patterns.py",
        math=source/"utils/math.py",ops=source/"utils/warp/ops.py",kernels=source/"utils/warp/kernels.py",
        preprocessing=ROOT/"upstream/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/observations.py",
        ground_preprocessing=ROOT/"rbq10_parkour/multiview_observations.py")
    spec=importlib.util.spec_from_file_location("_rbq_multi_kernels",paths["kernels"])
    kernels=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernels;spec.loader.exec_module(kernels)
    ns=dict(torch=torch,np=np,math=math,wp=wp,kernels=kernels)
    ops=functions(paths["ops"],["raycast_mesh","convert_to_warp_mesh"],ns.copy())
    math_ops=functions(paths["math"],namespace=ns.copy())
    intrinsics=functions(paths["camera"],["_compute_intrinsic_matrices"],ns.copy(),"RayCasterCamera")["_compute_intrinsic_matrices"]
    pattern=functions(paths["pattern"],["pinhole_camera_pattern"],ns.copy())["pinhole_camera_pattern"]
    terrain_functions=functions(SOURCE/"utils.py",["convert_height_field_to_mesh"],dict(np=np))
    class_ns=dict(torch=torch,torchvision=torchvision,np=np,ManagerTermBase=object)
    front_class=load_class(paths["preprocessing"],"image_features",class_ns)
    ground_class=load_class(paths["ground_preprocessing"],"GroundDepthImageFeatures",class_ns)
    meta=json.loads((a.policy_dir/"manifest.json").read_text());cfg=meta["configuration"];contract=meta["contract"]
    validate_configuration(cfg,contract)
    for mutation in ("order","pose"):
        wrong=copy.deepcopy(cfg)
        if mutation=="order":wrong["cameras"][1],wrong["cameras"][2]=wrong["cameras"][2],wrong["cameras"][1]
        else:wrong["cameras"][1]["position_m"][0]+=.001
        try:validate_configuration(wrong,contract)
        except ValueError:pass
        else:raise AssertionError("Mismatched camera configuration accepted")
    report=dict(scope="CPU execution of actual installed Isaac camera math, Warp kernels and source front/ground preprocessing; not live GPU/PhysX dynamics equivalence",
        checkpoint=meta["checkpoint"],checkpoint_sha256=meta["checkpoint_sha256"],view_order=cfg["view_order"],
        source_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
        wrong_pose_and_order_rejected=True,cases=[])
    home_dict=json.loads((ASSETS/"official/info.json").read_text())["config_info"]["init_state"]["default_joint_angles"]
    home=np.array([home_dict[n] for n in contract["joint_order"]])
    actor=torch.jit.load(str(a.policy_dir/"student.pt")).eval()
    poses=[(0.,0.,.55,0.,0.,0.),(.18,.03,.52,.08,-.1,.12),(.4,-.04,.6,-.1,.12,-.2),(.65,0.,.5,0.,-.06,0.)]
    for course in ("flat","hurdle","step","isaac_platforms","isaac_gap","isaac_mixed"):
        xml=add_cameras(build_scene(contract,course,height=.25,difficulty=.5,terrain_seed=42,contact_profile="firm"),cfg["cameras"])
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        b=ObservationBridge(m,d,contract,home,course=="flat",.6);b.terrain_scan_enabled=False
        d.qpos[b.qids]=home;mujoco.mj_forward(m,d)
        camera=MultiViewDepthInput(m,d,cfg)
        mesh=original_mesh(m,d,course,terrain_functions)
        warp_mesh=ops["convert_to_warp_mesh"](np.asarray(mesh.vertices),np.asarray(mesh.faces),"cpu")
        references=[];view_reports=[];source_local=[]
        for index,view in enumerate(cfg["cameras"]):
            pat=NS(**{k:view[k] for k in ("width","height","focal_length","horizontal_aperture","vertical_aperture")},horizontal_aperture_offset=0.,vertical_aperture_offset=0.)
            cam=NS(cfg=NS(pattern_cfg=pat),_data=NS(intrinsic_matrices=torch.eye(3)[None]));intrinsics(cam)
            _,rays=pattern(pat,cam._data.intrinsic_matrices,"cpu")
            q=math_ops["convert_camera_frame_orientation_convention"](torch.tensor([view["quaternion_wxyz_ros"]]),origin="ros",target="world")
            rot=math_ops["matrix_from_quat"](q)[0].numpy();source_local.append(rays[0].numpy()@rot.T)
            cls=front_class if index==0 else ground_class;ref=cls.__new__(cls)
            ref.num_envs=1;ref.device="cpu";ref.clipping_range=2.;ref.buffer_len=2;ref.debug_vis=False
            ref.depth_buffer=torch.zeros(1,2,58,87)
            ref.resize_transform=torchvision.transforms.Resize((58,87),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            if index:ref.optical_z_cosine=rays[0,:,0].reshape(60,106)
            references.append(ref)
            R=np.zeros(9);mujoco.mju_quat2Mat(R,np.array(view["quaternion_wxyz_ros"]))
            ray_error=float(abs(source_local[-1]-camera.rays[index]@R.reshape(3,3).T).max());assert ray_error<2e-6
            # Invalid/near-range handling and source torchvision resize are tested on identical raw input.
            probe=np.random.default_rng(123).uniform(.05,2.,(5,60,106)).astype(np.float32)
            if index:probe[index,10,20]=np.nan
            processed=camera.preprocess(probe)[0,index]
            assert torch.isfinite(processed).all()
            preprocess_error=float((processed-ref._process_depth_image(torch.from_numpy(probe[index]))).abs().max())
            assert preprocess_error<2e-6
            view_reports.append(dict(view=view["name"],ray_max_error=ray_error,same_raw_preprocess_max_error=preprocess_error,intrinsic_matrix=cam._data.intrinsic_matrices[0].tolist(),samples=[]))
        ref_encoder=torch.jit.load(str(a.policy_dir/"depth.pt")).eval();mj_encoder=torch.jit.load(str(a.policy_dir/"depth.pt")).eval()
        samples=[]
        for tick,pose in enumerate(poses):
            x,y,z,roll,pitch,yaw=pose
            d.qpos[:3]=[x+(1. if course.startswith("isaac_") else 0.),y,z]
            d.qpos[3:7]=math_ops["quat_from_euler_xyz"](*[torch.tensor([v]) for v in (roll,pitch,yaw)])[0].numpy()
            d.qvel[:]=0;mujoco.mj_forward(m,d)
            R=d.xmat[b.base].reshape(3,3);raw_refs=[]
            for index,view in enumerate(cfg["cameras"]):
                origin=d.xpos[b.base]+R@np.array(view["position_m"])
                directions=(source_local[index]@R.T).astype(np.float32)
                starts=torch.from_numpy(np.repeat(origin[None],len(directions),axis=0).astype(np.float32))[None]
                _,distance,_,_=ops["raycast_mesh"](starts,torch.from_numpy(directions)[None],warp_mesh,return_distance=True)
                raw=distance.clamp(max=2.).reshape(60,106);raw_refs.append(raw)
                ref=references[index];ref.camera_sensor=NS(data=NS(output={"distance_to_camera":raw[None,:,:,None]}))
                if tick==0:ref.reset()
            delayed_ref=torch.stack([ref(NS(common_step_counter=tick*5,device="cpu"),None,(58,87),2,False) for ref in references],dim=1)
            delayed_mj=camera.sample()
            assert tuple(delayed_mj.shape)==(1,5,58,87)
            for index,ref in enumerate(references):
                raw_error=camera.latest_raw[index]-raw_refs[index].numpy()
                row=dict(raw_depth_m=stats(raw_error),delayed_input=stats((delayed_ref[:,index]-delayed_mj[:,index]).numpy()),
                         camera_origin_max_error_m=float(abs(camera.latest_origins[index]-(d.xpos[b.base]+R@np.array(cfg["cameras"][index]["position_m"]))).max()))
                if not course.startswith("isaac_"):assert row["raw_depth_m"]["max"]<2e-4,(course,index,row)
                assert row["camera_origin_max_error_m"]<1e-12
                view_reports[index]["samples"].append(row)
                held=ref(NS(common_step_counter=tick*5+1,device="cpu"),None,(58,87),2,False)
                assert torch.equal(held,delayed_ref[:,index]),"Depth buffer advanced between scheduled updates"
            obs,_=b.sample(np.zeros(12,dtype=np.float32))
            obs=student_observation(obs,.6,0.,np.zeros(3),dict(cfg,terrain_label="flat" if course=="flat" else "parkour"))
            prop=torch.from_numpy(obs[None,:53].copy());prop[:,6:8]=0
            with torch.inference_mode():
                lf=ref_encoder(delayed_ref,prop)[:,:32];lm=mj_encoder(delayed_mj,prop)[:,:32]
                af=actor(torch.from_numpy(obs[None]),lf);am=actor(torch.from_numpy(obs[None]),lm)
            samples.append(dict(pose_xyz_rpy=list(pose),latent=stats((lf-lm).numpy()),action=stats((af-am).numpy())))
        case=dict(course=course,views=view_reports,policy_sensitivity=samples)
        report["cases"].append(case)
        (a.output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
        print("DEPTH5_CHECK",course,"max_ray_m",max(r["raw_depth_m"]["max"] for v in view_reports for r in v["samples"]),flush=True)
    report["completed"]=True
    (a.output/"results.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
