"""Reuse original Isaac parkour raw height functions without launching Isaac Sim."""
import ast
import copy
import hashlib
from pathlib import Path
import random
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'upstream/Isaaclab_Parkour/parkour_isaaclab/terrains'
NAMES = ('flat','hurdle','step','gap','platforms','mixed')
COURSES = tuple('isaac_'+n for n in NAMES)
KEYS = dict(zip(NAMES,('parkour_flat','parkour_hurdle','parkour_step','parkour_gap','parkour','parkour_demo')))


class _Config:
    def __init__(self, **kwargs):
        for cls in reversed(type(self).__mro__):
            for key,value in vars(cls).items():
                if not key.startswith('_'):
                    setattr(self,key,copy.deepcopy(value))
        self.__dict__.update(kwargs)


def _execute(path, namespace, classes_only=False):
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if classes_only:
            if isinstance(node,ast.ClassDef):body.append(node)
        elif isinstance(node,ast.ImportFrom) and (node.level or (node.module or '').startswith('parkour_isaaclab')):
            continue  # The mesh decorator/import requires Isaac. Raw height function stays unchanged.
        elif isinstance(node,ast.If):
            continue  # TYPE_CHECKING imports only.
        else:
            body.append(node)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    body=[node for node in body if not (isinstance(node,ast.ImportFrom) and node.module=='__future__')]
    code=ast.fix_missing_locations(ast.Module(body=[future,*body],type_ignores=[]))
    exec(compile(code,str(path),'exec'),namespace)


def original_config():
    namespace={'configclass':lambda cls:cls,'HfTerrainBaseCfg':_Config,'TerrainGeneratorCfg':_Config,
               'parkour_field_to_mesh':lambda function:function}
    _execute(SOURCE/'extreme_parkour/extreme_parkour_terrians.py',namespace)
    namespace['extreme_parkour_terrians']=SimpleNamespace(**namespace.copy())
    _execute(SOURCE/'parkour_terrain_generator_cfg.py',namespace,classes_only=True)
    _execute(SOURCE/'extreme_parkour/extreme_parkour_terrains_cfg.py',namespace,classes_only=True)
    _execute(SOURCE/'extreme_parkour/config/parkour.py',namespace)
    return namespace['EXTREME_PARKOUR_TERRAINS_CFG']


def generate(name,difficulty=.5,seed=42):
    if name.startswith('isaac_'):name=name[6:]
    if name not in NAMES or not 0<=difficulty<=1:
        raise ValueError('Expected known course and difficulty in [0,1]')
    root=original_config()
    cfg=copy.deepcopy(root.sub_terrains[KEYS[name]])
    cfg.size=root.size
    full_size=cfg.size
    cfg.horizontal_scale=.1  # Original Student scene setting.
    cfg.vertical_scale=root.vertical_scale
    if KEYS[name]!='parkour':cfg.y_range=(-.1,.1)  # Original Student override.
    np_state,py_state=np.random.get_state(),random.getstate()
    try:
        np.random.seed(seed);random.seed(seed)
        # Match parkour_field_to_mesh: it always inserts a one-pixel border,
        # including when border_width=0. The previous bridge skipped this wrapper.
        border=int(cfg.border_width/cfg.horizontal_scale)+1
        shape=tuple(int(dim/cfg.horizontal_scale)+1 for dim in full_size)
        cfg.size=tuple((n-2*border)*cfg.horizontal_scale for n in shape)
        inner,goals,goal_heights=cfg.function(float(difficulty),cfg,root.num_goals)
        raw=np.zeros(shape,dtype=np.int16)
        raw[border:-border,border:-border]=inner
        # Upstream centers goals using the inner size, meshes using the full size.
        goals=np.asarray(goals)+(np.asarray(full_size)-np.asarray(cfg.size))/2
        # Step function alone returns height units; its goal heights are unused by policy.
        if name=='step':goal_heights=np.asarray(goal_heights)*cfg.vertical_scale
    finally:
        np.random.set_state(np_state);random.setstate(py_state)
    heights=np.asarray(raw,dtype=np.float64)*cfg.vertical_scale
    if not np.isfinite(heights).all():raise ValueError('Invalid original height field')
    return {'name':name,'difficulty':difficulty,'seed':seed,'heights':heights,
            'spacing':cfg.horizontal_scale,'vertical_scale':cfg.vertical_scale,'origin_xy':np.array([0.,-full_size[1]/2]),'profile':'upstream_wrapper_v2','goals':np.asarray(goals),'goal_heights':np.asarray(goal_heights),
            'slope_threshold':cfg.slope_threshold,'source_sha256':hashlib.sha256((SOURCE/'extreme_parkour/extreme_parkour_terrians.py').read_bytes()).hexdigest()}


def add_heightfield(root, terrain, name='terrain', origin=(0.,0.), rgba=(1.,1.,1.,1.)):
    """MuJoCo hfield rows are Y, columns X; source raw array is X,Y."""
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    world=root.find('worldbody')
    z=terrain['heights'];spacing=terrain['spacing']
    width,length=(z.shape[0]-1)*spacing,(z.shape[1]-1)*spacing
    zmin,zmax=float(z.min()),float(z.max())
    scale=max(zmax-zmin,1e-6)
    # MuJoCo XML elevation loader flips image-style rows on compilation.
    # Reverse rows here so the compiled array and world Y preserve source X,Y.
    elevation=((z-zmin)/scale).T[::-1].ravel()
    ET.SubElement(asset,'hfield',name=name,nrow=str(z.shape[1]),ncol=str(z.shape[0]),
                  size=f'{width/2} {length/2} {scale} 0.15',
                  elevation=' '.join(f'{x:.9g}' for x in elevation))
    from viewer_style import heightfield_material
    material=heightfield_material(asset,name,z,spacing)
    ET.SubElement(world,'geom',name=name,type='hfield',hfield=name,group='0',
                  pos=f'{origin[0]+width/2} {origin[1]+length/2} {zmin}',
                  rgba=' '.join(map(str,rgba)),material=material,friction='0.9 0.005 0.0001')
    return {'name':terrain['name'],'difficulty':terrain['difficulty'],'seed':terrain['seed'],
            'origin_xy':list(origin),'extent_xy':[width,length],'grid_shape_xy':list(z.shape),
            'height_range_m':[zmin,zmax],'goals_xy':(terrain['goals']+np.asarray(origin)).tolist(),
            'goal_heights_m':terrain['goal_heights'].tolist(),'source_sha256':terrain['source_sha256']}


def refine_source_edges(terrain, spacing=.02):
    """Rasterize the unchanged upstream slope-corrected triangle surface.

    MuJoCo hfields cannot have truly vertical walls. A 2 cm raster limits the
    remaining transition width to 2 cm instead of the previous 10 cm ramp.
    This is a geometry conversion, not a change of obstacle heights or goals.
    """
    from scipy.ndimage import distance_transform_edt
    source=SOURCE/'utils.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='convert_height_field_to_mesh')
    module=ast.Module(body=[node],type_ignores=[])
    namespace={'np':np}
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
    z=terrain['heights']; original_spacing=terrain['spacing']
    vertical_scale=terrain['vertical_scale']
    raw=np.rint(z/vertical_scale).astype(np.int16)
    vertices,triangles,_=namespace[node.name](raw,original_spacing,vertical_scale,terrain['slope_threshold'])
    shape=tuple(int(round((n-1)*original_spacing/spacing))+1 for n in z.shape)
    raster=np.full(shape,-np.inf)
    for face in triangles:
        p=vertices[face].astype(float)
        determinant=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(determinant)<1e-10:continue  # Vertical wall projects to zero area.
        lo=np.maximum(np.ceil(p[:,:2].min(axis=0)/spacing-1e-5).astype(int),0)
        hi=np.minimum(np.floor(p[:,:2].max(axis=0)/spacing+1e-5).astype(int),np.array(shape)-1)
        if (hi<lo).any():continue
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1)*spacing,np.arange(lo[1],hi[1]+1)*spacing,indexing='ij')
        a=((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/determinant
        b=((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/determinant
        inside=(a>=-1e-5)&(b>=-1e-5)&(a+b<=1+1e-5)
        values=np.where(inside,a*p[0,2]+b*p[1,2]+(1-a-b)*p[2,2],-np.inf)
        target=raster[lo[0]:hi[0]+1,lo[1]:hi[1]+1]
        np.maximum(target,values,out=target)
    missing=~np.isfinite(raster)
    if missing.any():
        indices=distance_transform_edt(missing,return_distances=False,return_indices=True)
        raster[missing]=raster[tuple(i[missing] for i in indices)]
    result=dict(terrain,heights=raster,spacing=spacing)
    result['profile']='upstream_wrapper_v2_slope_mesh_2cm'
    return result
