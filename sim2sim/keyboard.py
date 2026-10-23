"""MuJoCo viewer key-press commands; no joystick dependency."""
from threading import Lock


class KeyboardInput:
    def __init__(self, vx=0., yaw_rate=0.):
        self.vx=float(vx);self.yaw_rate=float(yaw_rate)
        self.reset_requested=False;self.quit_requested=False
        self.active=bool(vx or yaw_rate)
        self.lock=Lock()

    def on_key(self,key):
        with self.lock:
            if key in (87,265,83,264,65,263,68,262):self.active=True
            if key in (87,265):self.vx=min(.8,self.vx+.1)  # W / up
            elif key in (83,264):self.vx=max(-.8,self.vx-.1)  # S / down
            elif key in (65,263):self.yaw_rate=min(1.,self.yaw_rate+.2)  # A / left
            elif key in (68,262):self.yaw_rate=max(-1.,self.yaw_rate-.2)  # D / right
            elif key==32:self.vx=0.;self.yaw_rate=0.  # space
            elif key==82:
                self.vx=0.;self.yaw_rate=0.;self.reset_requested=True;self.active=False
            elif key in (81,256):self.quit_requested=True  # Q / escape

    def read(self):
        with self.lock:
            state=(self.vx,self.yaw_rate,self.reset_requested,self.quit_requested)
            self.reset_requested=False
        return state
