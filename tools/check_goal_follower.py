"""Check the NumPy waypoint adapter against the actual upstream Torch method."""
import ast,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim2sim'))
from goal_follower import TerrainGoalFollower
from terrains import generate
source=ROOT/'upstream/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/parkours/parkour_event.py'
tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ParkourEvent')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_command')
ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),ns)
terrain=generate('isaac_platforms',.5,42);follower=TerrainGoalFollower(terrain)
goals=torch.tensor(np.pad(follower.goals,((0,2),(0,1)),mode='edge')[None],dtype=torch.float64)
goals[0,-2:]=goals[0,-3]
position=torch.zeros(1,3,dtype=torch.float64)
ref=NS(reach_goal_timer=torch.zeros(1),reach_goal_delay=.1,simulation_time=.02,debug_vis=False,
       cur_goal_idx=torch.zeros(1,dtype=torch.long),env_origins=torch.zeros(1,3),
       next_goal_threshold=.2,cur_goals=goals[:,0].clone(),next_goals=goals[:,1].clone(),
       robot=NS(data=NS(root_pos_w=position)),env=NS(scene=NS(terrain=NS(cfg=NS(terrain_generator=NS(size=(16.,4.)))))),_reset_offset=3.)
ref.terrain=ref.env.scene.terrain
ref.device="cpu"
ref._gather_cur_goals=lambda future=0:goals[:,int(ref.cur_goal_idx[0])+future]
maximum=0.;ticks=0
# Traverse every segment including the switch delay, with changing yaw.
for target in follower.goals:
    start=position[0,:2].numpy().copy()
    for alpha in np.r_[np.linspace(0,1,65),np.ones(12)]:
        pos=start*(1-alpha)+target*alpha;position[0,:2]=torch.tensor(pos)
        yaw=.2*np.sin(ticks*.1)
        ns['_update_command'](ref)
        actual=follower.update(pos,yaw)
        if ticks%5==0:expected=np.array([ref.target_yaw.item(),ref.next_target_yaw.item()])-yaw
        maximum=max(maximum,float(np.abs(actual-expected).max()))
        assert maximum<1e-7,(ticks,actual,expected)
        assert follower.index==int(ref.cur_goal_idx[0])
        ticks+=1
        if follower.complete:break
    if follower.complete:break
assert follower.complete
out={'upstream_source':str(source.relative_to(ROOT)),'steps':ticks,'goals_completed':follower.index,'max_heading_error_rad':maximum,'timer_and_goal_indices_match':True}
print(json.dumps(out,indent=2))
p=ROOT/'evaluations/autonomous/goal_adapter_check.json';p.write_text(json.dumps(out,indent=2)+'\n')
