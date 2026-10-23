"""Display colors only; never adds geometry or changes terrain samples."""
import hashlib
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


GOAL_PENDING = (1., .2, .65, 1.)
GOAL_CURRENT = (.05, 1., 1., 1.)
GOAL_REACHED = (.15, .85, .3, .9)


def heightfield_material(asset, name, heights, spacing):
    """Bake a height palette and real discontinuity outlines into a 2D texture."""
    z = np.asarray(heights)
    colors = np.array([[74, 91, 112], [239, 183, 65], [233, 133, 51],
                       [215, 81, 59], [166, 87, 158], [119, 82, 166]], dtype=np.uint8)
    levels = np.digitize(z, [.075, .23, .43, .68, 1.05])
    rgb = colors[levels].copy()
    rgb[z < -.075] = [15, 28, 48]
    # Distinguish walkable stones from the surrounding pit even at height zero.
    if z.min() < -.15:
        rgb[(z >= -.075) & (z < .075)] = [55, 156, 172]
    x, y = np.indices(z.shape)
    grid = ((np.mod(x * spacing, .5) < spacing * .65) |
            (np.mod(y * spacing, .5) < spacing * .65)) & (z >= -.075)
    rgb[grid] = np.minimum(rgb[grid].astype(int) + 14, 255)
    edge = np.zeros(z.shape, dtype=bool)
    dx, dy = np.abs(np.diff(z, axis=0)) > .075, np.abs(np.diff(z, axis=1)) > .075
    edge[:-1] |= dx; edge[1:] |= dx
    edge[:, :-1] |= dy; edge[:, 1:] |= dy
    rgb[edge] = [255, 225, 70]
    # MuJoCo maps S to +X and T to -Y; PNG rows run toward +Y.
    image = np.ascontiguousarray(rgb.transpose(1, 0, 2))
    digest = hashlib.sha256(image.tobytes()).hexdigest()[:24]
    cache = Path(tempfile.gettempdir()) / "rbq10_mujoco_viewer_textures"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"height_{digest}.png"
    if not path.exists():
        # Separate concurrent writers, then atomically publish the complete PNG.
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            Image.fromarray(image).save(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    texture, material = f"{name}_height_colors", f"{name}_surface"
    ET.SubElement(asset, "texture", name=texture, type="2d", file=str(path))
    ET.SubElement(asset, "material", name=material, texture=texture, texrepeat="1 1",
                  texuniform="false", rgba="1 1 1 1", specular=".05", shininess=".1")
    return material


def robot_colors(trunk):
    for body in trunk.iter("body"):
        name = body.get("name", "")
        color = ".92 .95 1 1" if name == "trunk" else ".08 .38 .88 1"
        if "foot" in name:
            color = ".95 .12 .18 1"
        for geom in body.findall("geom"):
            geom.set("rgba", color)
