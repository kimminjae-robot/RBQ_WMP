"""Compile the official URDF; add a floating base, PD servos and small test courses."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets/rbq10'


def build_scene(contract, course='flat', height=0.10, difficulty=.5, terrain_seed=42):
    source = ET.parse(ASSETS / 'resources/urdf/rbq10.urdf').getroot()
    for link in source.findall('link'):
        for kind in ('visual', 'collision'):
            for i, geom in enumerate(link.findall(kind)):
                geom.set('name', f"{link.get('name')}_{kind}_{i}")
    for mesh in source.findall('.//mesh'):
        mesh.set('filename', str((ASSETS / 'resources/urdf' / mesh.get('filename')).resolve()))
    extension = ET.SubElement(source, 'mujoco')
    ET.SubElement(extension, 'compiler', discardvisual='false', fusestatic='false', strippath='false')
    # Use MuJoCo's URDF importer; do not hand-copy masses, inertia or joint axes.
    with tempfile.TemporaryDirectory(prefix='rbq10_mj_') as temp:
        urdf = Path(temp) / 'robot.urdf'
        urdf.write_bytes(ET.tostring(source))
        imported = mujoco.MjModel.from_xml_path(str(urdf))
        mjcf = Path(temp) / 'robot.xml'
        mujoco.mj_saveLastXML(str(mjcf), imported)
        root = ET.parse(mjcf).getroot()
    ET.SubElement(root, 'option', timestep='0.005', gravity='0 0 -9.81', integrator='implicitfast')
    default = root.find('default')
    if default is None:
        default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction='0.9 0.005 0.0001', condim='3')
    world = root.find('worldbody')
    trunk = world.find("body[@name='trunk']")
    if trunk is None:
        raise ValueError('URDF importer did not preserve trunk body')
    trunk.set('pos', '0 0 0.55')
    ET.SubElement(trunk, 'freejoint', name='floating_base')
    ET.SubElement(world, 'light', pos='0 -3 5', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='100 20 0.1',
                  group='0', rgba='0.35 0.38 0.4 1')
    # Static terrain group 0; robot geoms group 1, excluded by terrain ray casting.
    for geom in trunk.findall('.//geom'):
        geom.set('group', '1')
    if course.startswith('isaac_'):
        from terrains import generate, add_heightfield
        terrain=generate(course,difficulty,terrain_seed)
        # No plane at z=0: it would fill the negative-height gaps/pits.
        world.remove(world.find("geom[@name='ground']"))
        ET.SubElement(world,'geom',name='catch_floor',type='plane',pos='0 0 -2.5',
                      size='100 20 .1',group='0',rgba='.15 .17 .19 1')
        add_heightfield(root,terrain,origin=(0.,-1.95))
        # Put the robot on the original initial platform, not on the hfield boundary.
        initial_z=float(terrain['heights'][10,20])+.55
        trunk.set('pos',f'1 0 {initial_z}')
    elif course == 'hurdle':
        for i, x in enumerate((2.0, 4.0, 6.0)):
            ET.SubElement(world, 'geom', name=f'hurdle_{i}', type='box',
                          pos=f'{x} 0 {height/2}', size=f'0.12 1.5 {height/2}', rgba='0.8 0.5 0.2 1')
    elif course == 'step':
        ET.SubElement(world, 'geom', name='platform', type='box',
                      pos=f'4 0 {height/2}', size=f'2.0 1.5 {height/2}', rgba='0.8 0.5 0.2 1')
    elif course != 'flat':
        raise ValueError(course)
    actuators = ET.SubElement(root, 'actuator')
    for name in contract['joint_order']:
        j = trunk.find(f".//joint[@name='{name}']")
        if j is None:
            raise ValueError(f'Missing joint: {name}')
        knee = name.endswith('K')
        j.set('armature', str(0.0214816 if knee else 0.014058))
        j.set('damping', '0')
        limit = 150 if knee else 100
        ET.SubElement(actuators, 'position', name=f'pd_{name}', joint=name,
                      kp=str(127.77215 if knee else 123.39138), kv='2.5',
                      forcelimited='true', forcerange=f'-{limit} {limit}')
    return ET.tostring(root, encoding='unicode')
