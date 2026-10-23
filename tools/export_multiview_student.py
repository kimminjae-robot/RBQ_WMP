"""Export real five-view Student CNN/fusion/GRU and actor, with strict camera metadata."""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"sim2sim"))
from export_sim2sim import load_original_actor, StudentInference
from rbq10_parkour.multiview_contract import validate_multiview_checkpoint
from rbq10_parkour.multiview_network import MultiViewDepthCNN
from multiview_depth import configuration, FORMAT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    validate_multiview_checkpoint(saved)
    actor,_=load_original_actor(saved)
    actor.load_state_dict(saved["depth_actor_state_dict"],strict=True)
    policy=StudentInference(actor).eval()
    module=importlib.import_module("_rbq_original_networks.feature_extractors.depth_backbone")
    depth=module.RecurrentDepthBackbone(MultiViewDepthCNN(module.DepthOnlyFCBackbone58x87(32,512)),{"num_prop":53})
    depth.load_state_dict(saved["depth_encoder_state_dict"],strict=True)
    depth.eval()
    traced=torch.jit.trace(policy,(torch.randn(1,753),torch.randn(1,32)))
    scripted=torch.jit.script(depth)
    errors=[]
    torch.manual_seed(42)
    with torch.inference_mode():
        for _ in range(12):
            obs=torch.randn(1,753);prop=obs[:,:53].clone();prop[:,6:8]=0
            images=torch.rand(1,5,58,87)-.5
            expected=depth(images,prop);actual=scripted(images,prop)
            errors.append(float((expected-actual).abs().max()))
            errors.append(float((policy(obs,expected[:,:32])-traced(obs,actual[:,:32])).abs().max()))
        changed=obs.clone();changed[:,53:185]+=100;changed[:,194:223]+=100
        assert torch.allclose(traced(obs,actual[:,:32]),traced(changed,actual[:,:32]),atol=1e-6)
        # Verify each actual trained ground-view fusion block affects inference.
        sensitivity={}
        for index,name in enumerate(("front","ground_0","ground_1","ground_2","ground_3")):
            left,right=copy.deepcopy(depth),copy.deepcopy(depth)
            left.hidden_states=right.hidden_states=torch.zeros(1,0,512)
            altered=images.clone();altered[:,index]=.5
            sensitivity[name]=float((left(images,prop)-right(altered,prop)).abs().max())
        assert all(value>0 for value in sensitivity.values()),sensitivity
    if max(errors)>1e-5:raise AssertionError(errors)
    scripted.hidden_states=torch.zeros(1,0,512)
    args.output.mkdir(parents=True,exist_ok=True)
    traced.save(str(args.output/"student.pt"));scripted.save(str(args.output/"depth.pt"))
    cfg=configuration(saved["robot_contract"])
    (args.output/"configuration.json").write_text(json.dumps(cfg,indent=2)+"\n")
    result=dict(format=FORMAT,role="student",checkpoint=args.checkpoint.name,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        iteration=saved["iter"],updates_completed=saved["rbq10_student_state"]["next_iteration"],
        contract=saved["robot_contract"],configuration=cfg,export_max_abs_error=max(errors),
        camera_input_sensitivity=sensitivity,privileged_scan_and_physics_truth_ignored=True,
        files={name:hashlib.sha256((args.output/name).read_bytes()).hexdigest() for name in ("student.pt","depth.pt","configuration.json")},
        recurrent_state="Original stateful GRU; loaded fresh for each rollout",depth_input_shape=[1,5,58,87])
    (args.output/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("contract","configuration")},indent=2))


if __name__=="__main__":main()
