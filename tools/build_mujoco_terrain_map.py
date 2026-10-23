"""Build and render 18 actual MuJoCo courses from original Isaac parkour height functions."""
import os
import sys
if '--viewer' not in sys.argv:
    os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim2sim'))
from terrains import generate,add_heightfield,NAMES,SOURCE

COLORS=[(.36,.63,.52,1),(.86,.58,.27,1),(.48,.65,.86,1),(.77,.43,.40,1),(.64,.51,.82,1),(.42,.71,.76,1)]
LABELS=['ROUGH FLAT','HURDLES / JUMP','STEPS / CLIMB','GAPS / LEAP','STEPPING PLATFORMS','MIXED DEMO']


def make_atlas(seed=42):
    root=ET.Element('mujoco',model='Isaac_parkour_terrain_atlas')
    ET.SubElement(root,'option',gravity='0 0 -9.81')
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1800',offheight='1200',orthographic='true',fovy='38')
    ET.SubElement(visual,'quality',shadowsize='2048')
    ET.SubElement(visual,'headlight',ambient='.4 .4 .4',diffuse='.5 .5 .5')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='15 0 35',dir='-.3 .2 -1',directional='true',castshadow='true',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='catch_floor',type='plane',pos='0 0 -2.5',size='100 100 .1',rgba='.09 .12 .16 1')
    metadata=[];terrains=[]
    for row,name in enumerate(NAMES):
        for col,level in enumerate([0,4,9]):
            difficulty=level/9
            terrain=generate(name,difficulty,seed)
            origin=(col*18.,(5-row)*5.)
            record=add_heightfield(root,terrain,f'tile_{row}_{col}',origin,COLORS[row])
            record.update(level=level,row=row,column=col,grid_height_sha256=hashlib.sha256(terrain['heights'].tobytes()).hexdigest())
            # Goal markers are sites: they do not affect collision or depth ray tests.
            for index,(point,z) in enumerate(zip(terrain['goals'],terrain['goal_heights'])):
                if 0<=point[0]<=15.9 and 0<=point[1]<=3.9:
                    ET.SubElement(world,'site',name=f'goal_{row}_{col}_{index}',type='sphere',size='.055',
                        pos=f'{origin[0]+point[0]} {origin[1]+point[1]} {z+.10}',rgba='1 .95 .6 1')
            metadata.append(record);terrains.append(terrain)
    return root,metadata,terrains


def check_heights(model,data,records,terrains):
    maximum=0.;tested=0
    group=np.array([1,0,0,0,0,0],dtype=np.uint8)
    for k,(record,terrain) in enumerate(zip(records,terrains)):
        z=terrain['heights'];ox,oy=record['origin_xy']
        # Interior grid vertices: detects transpose, scale and missing gap/filled pit mistakes.
        for ix in range(5,z.shape[0]-5,9):
            for iy in [5,13,20,30]:
                origin=np.array([ox+ix*.1,oy+iy*.1,5.])
                hit=np.array([-1],dtype=np.int32)
                distance=mujoco.mj_ray(model,data,origin,np.array([0.,0.,-1.]),group,1,-1,hit)
                if distance<0:raise AssertionError('Missing heightfield collision')
                error=abs(5-distance-z[ix,iy]);maximum=max(maximum,error);tested+=1
                assert error<2e-5,(record['name'],record['level'],ix,iy,error)
    return {'ray_height_samples':tested,'maximum_absolute_height_error_m':maximum}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--viewer',action='store_true',help='Open interactive full-map view after saving the overview')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/mujoco_terrain_overview.png')
    args=parser.parse_args()
    root,records,terrains=make_atlas(args.seed)
    output_dir=ROOT/'outputs/mujoco_terrain_map';output_dir.mkdir(parents=True,exist_ok=True)
    xml=ET.tostring(root,encoding='unicode')
    (output_dir/'atlas.xml').write_text(xml)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    validation=check_heights(model,data,records,terrains)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[26,14,0]
    camera.distance=53;camera.azimuth=90;camera.elevation=-66;camera.orthographic=1
    width,height=1800,1200
    with mujoco.Renderer(model,height=height,width=width) as renderer:
        renderer.update_scene(data,camera=camera)
        image=Image.fromarray(renderer.render().copy())
        gl=renderer.scene.camera[0]
        position=np.array(gl.pos);forward=np.array(gl.forward);up=np.array(gl.up)
        right=np.cross(forward,up)
        half_height=float(gl.frustum_top-gl.frustum_bottom)/2
        def project(point):
            delta=np.asarray(point)-position
            x=float(delta@right);y=float(delta@up)
            span=half_height if gl.orthographic else half_height*float(delta@forward)/gl.frustum_near
            return (width/2+x/(2*span*width/height)*width,height/2-y/(2*span)*height)
        draw=ImageDraw.Draw(image)
        font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        title_font=ImageFont.truetype(bold,31);font=ImageFont.truetype(bold,18);small=ImageFont.truetype(font_path,18)
        draw.rectangle((0,0,width,112),fill=(15,23,35))
        draw.text((32,16),'MUJOCO | ISAAC PARKOUR TERRAIN ATLAS',font=title_font,fill='white')
        draw.text((34,66),'6 terrain types x 3 difficulties  |  16 x 4 m source courses  |  seed 42  |  actual MuJoCo render',font=small,fill=(190,210,228))
        for record in records:
            row,col=record['row'],record['column']
            ox,oy=record['origin_xy']
            x,y=project([ox+7.95,oy+4.5,.1])
            text=f'{LABELS[row]}  |  L{record["level"]}/9'
            bbox=draw.textbbox((0,0),text,font=font);tw=bbox[2]
            box=(x-tw/2-10,y-11,x+tw/2+10,y+18)
            draw.rounded_rectangle(box,radius=5,fill=(18,26,38))
            draw.text((x-tw/2,y-7),text,font=font,fill='white')
        draw.rectangle((0,height-64,width,height),fill=(15,23,35))
        draw.text((32,height-54),'Left: level 0    |    Middle: level 4    |    Right: level 9    |    Yellow dots: original course goals',font=small,fill=(225,230,240))
        draw.text((32,height-29),'Original height functions reused; MuJoCo heightfield contacts differ from Isaac triangle meshes. No policy performance implied.',font=ImageFont.truetype(font_path,15),fill=(160,180,199))
        args.output.parent.mkdir(parents=True,exist_ok=True);image.save(args.output)
    report={'engine':mujoco.__version__,'seed':args.seed,'courses':records,'validation':validation,
        'original_source':str(SOURCE.relative_to(ROOT)),
        'source_files_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'extreme_parkour/extreme_parkour_terrians.py',SOURCE/'extreme_parkour/extreme_parkour_terrains_cfg.py',SOURCE/'extreme_parkour/config/parkour.py',SOURCE/'parkour_terrain_generator_cfg.py']},
        'adapter':'Original raw height generator, Student 0.1m horizontal scale, 0.005m vertical scale; replaced only Isaac mesh decorator/config construction',
        'limitation':'Heightfield retains sampled heights but not Isaac slope-threshold vertical-edge correction or triangle simplification; gaps have bottoms, no overhang/Crawl added',
        'image':str(args.output.relative_to(ROOT))}
    (ROOT/'evaluations/mujoco_terrain_map').mkdir(parents=True,exist_ok=True)
    (ROOT/'evaluations/mujoco_terrain_map/manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'image':str(args.output),'courses':len(records),'validation':validation},indent=2))
    if args.viewer:
        import time
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model,data) as view:
            view.cam.lookat[:]=camera.lookat
            view.cam.distance=camera.distance;view.cam.azimuth=90;view.cam.elevation=-66
            while view.is_running():
                view.sync();time.sleep(.05)


if __name__=='__main__':main()
