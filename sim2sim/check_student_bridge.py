"""Input/depth/serialization checks; Go2 weights are only an offline network fixture."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

import mujoco
import numpy as np
import torch
import torchvision

from build_scene import ROOT, ASSETS, build_scene
from run_teacher import ObservationBridge
from student_inputs import CommandInput, DepthInput, student_observation, joystick_axis
from run_student import validate_export


def main():
    torch.set_num_threads(1)
    spec=importlib.util.spec_from_file_location('student_export',ROOT/'tools/export_sim2sim.py')
    export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
    cfg=json.loads((ROOT/'sim2sim/student_config.json').read_text())
    c=torch.load(ROOT/'checkpoints/rbq10_teacher/model_3674.pt',map_location='cpu',weights_only=False)
    rejected=False
    try:export.load_original_student(c)
    except ValueError:rejected=True
    assert rejected
    m=mujoco.MjModel.from_xml_string(build_scene(c['robot_contract']))
    d=mujoco.MjData(m)
    hd=json.loads((ASSETS/'official/info.json').read_text())['config_info']['init_state']['default_joint_angles']
    home=np.array([hd[n] for n in c['robot_contract']['joint_order']])
    b=ObservationBridge(m,d,c['robot_contract'],home,False,.6)
    b.terrain_scan_enabled=False
    d.qpos[b.qids]=home;d.ctrl[:]=home
    mujoco.mj_forward(m,d)
    for _ in range(400):mujoco.mj_step(m,d)
    obs,_=b.sample(np.zeros(12,dtype=np.float32))
    x=student_observation(obs,.6,-.4,[.2,-.1,.03],cfg)
    np.testing.assert_allclose(x[185:194],[.4,-.2,.06,0,0,0,0,0,0],atol=1e-6)
    np.testing.assert_allclose(x[6:8],[-.2,-.2])
    assert x[10]==np.float32(.6) and list(x[11:13])==[1,0]
    assert not x[53:185].any() and not x[194:223].any()
    assert b.current_contacts.all(), b.current_contacts
    # The terrain label comes from experiment config even on a different scene.
    assert x[11]==1 and cfg['terrain_label']=='parkour'
    depth=DepthInput(m,d,cfg['camera'])
    image=depth.sample()
    assert image.shape==(1,58,87) and torch.isfinite(image).all()
    assert depth.latest_raw.min()<1.99, 'Camera appears not to see ground'
    transform=torchvision.transforms.Resize((58,87),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    original=transform(torch.from_numpy(depth.latest_raw[:-2,4:-4].copy())[None])[0]/2.-.5
    preprocessing_error=float((image[0]-original).abs().max())
    assert preprocessing_error<1e-6
    np.testing.assert_allclose([joystick_axis(0),joystick_axis(32767),joystick_axis(-32767)],[0,1,-1])
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/'input.json'
        p.write_text(json.dumps({'command':{'vx':.5,'yaw_rate':-.3},'measured_velocity_body_m_s':[.1,.02,0]}))
        cmd=CommandInput(path=p)
        vx,wz,measured=cmd.read();cmd.close()
        assert vx==.5 and wz==-.3
        np.testing.assert_allclose(measured,[.1,.02,0])
    report={'teacher_checkpoint_rejected':True,'measured_velocity_separate_from_reference':True,
        'joystick_yaw_rate_and_manual_label_checked':True,'mujoco_standing_contacts_all_four':True,
        'camera_sees_ground':True,'depth_resize_vs_original_max_abs_error':preprocessing_error,
        'policy_execution_status':'No trained RBQ10 Student available; no RBQ10 Student rollout performed'}
    fixture=ROOT/'checkpoints/go2_student.pt'
    if fixture.is_file():
        # Verify actual pretrained Student networks without ever controlling RBQ10 with Go2 weights.
        ckpt=torch.load(fixture,map_location='cpu',weights_only=False)
        policy,encoder=export.load_original_student(ckpt)
        actor_jit=torch.jit.trace(policy,(torch.randn(1,753),torch.randn(1,32)))
        encoder_jit=torch.jit.script(encoder)
        reference=copy.deepcopy(encoder)
        errors=[]
        torch.manual_seed(42)
        with torch.inference_mode():
            for _ in range(8):
                im,prop=torch.rand(1,58,87)-.5,torch.randn(1,53)
                errors.append(float((reference(im,prop)-encoder_jit(im,prop)).abs().max()))
                xx,z=torch.randn(1,753),torch.randn(1,32)
                errors.append(float((policy(xx,z)-actor_jit(xx,z)).abs().max()))
                changed=xx.clone();changed[:,53:185]+=100;changed[:,194:223]+=100
                assert torch.allclose(policy(xx,z),policy(changed,z),atol=1e-6)
        assert max(errors)<1e-5
        encoder_jit.hidden_states=torch.zeros(1,0,512)
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'depth.pt';encoder_jit.save(str(file))
            restored=torch.jit.load(str(file))
            assert restored.hidden_states.shape==(1,0,512)
            assert restored(torch.zeros(1,58,87),torch.zeros(1,53)).shape==(1,34)
        report['offline_serialization_fixture']='Actual public Go2 Student weights; NOT an RBQ10 policy, never used for MuJoCo control'
        report['offline_network_sequence_steps']=8
        report['offline_network_max_abs_error']=max(errors)
    out=ROOT/'evaluations/sim2sim_student_interface/checks.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(out.read_text())


if __name__=='__main__':
    main()
