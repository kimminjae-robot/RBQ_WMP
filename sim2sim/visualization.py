"""Actual MuJoCo offscreen frames with checkpoint, command and termination labels."""
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont


class Capture:
    def __init__(self,model,video=None,screenshot=None):
        self.renderer=mujoco.Renderer(model,height=640,width=960)
        self.camera=mujoco.MjvCamera()
        self.camera.distance=3.4;self.camera.azimuth=125;self.camera.elevation=-24
        self.video=video;self.screenshot=screenshot;self.writer=None;self.last=None
        self.frames=0
        if video:
            import imageio.v2 as imageio
            Path(video).parent.mkdir(parents=True,exist_ok=True)
            self.writer=imageio.get_writer(str(video),fps=25,codec='libx264',macro_block_size=16)
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)

    def frame(self,data,base,checkpoint,course,vx,wz,state='RUNNING',autonomous=False):
        self.camera.lookat[:]=data.xpos[base]+np.array([.25,0.,.15])
        self.renderer.update_scene(data,camera=self.camera)
        frame=Image.fromarray(self.renderer.render().copy())
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,960,70),fill=(15,23,35))
        draw.text((12,9),f'RBQ10 Student {checkpoint} | {course} | t={data.time:.2f}s | {state}',font=self.font,fill='white')
        draw.text((12,38),f'vx={vx:+.2f} m/s  yaw_rate={wz:+.2f} rad/s | actual MuJoCo + Depth inference',font=self.font,fill=(200,215,230))
        draw.rectangle((0,607,960,640),fill=(15,23,35))
        legend='AUTONOMOUS: original terrain waypoints | no manual input | no reset on fall' if autonomous else 'W/S: speed  A/D: turn  SPACE: zero command  R: reset  Q/ESC: close'
        draw.text((12,614),legend,font=self.font,fill='white')
        self.last=np.asarray(frame)
        if self.writer:self.writer.append_data(self.last);self.frames+=1
        if self.screenshot and self.frames==1:
            Path(self.screenshot).parent.mkdir(parents=True,exist_ok=True);frame.save(self.screenshot)

    def finish(self,hold_last=False):
        if self.writer:
            if hold_last and self.last is not None:
                for _ in range(50):self.writer.append_data(self.last)
            self.writer.close()
        if self.screenshot and self.last is not None:
            Path(self.screenshot).parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(self.last).save(self.screenshot)
        self.renderer.close()
