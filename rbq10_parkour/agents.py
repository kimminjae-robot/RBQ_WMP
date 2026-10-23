"""Original PPO/distillation settings with separate RBQ10 logs and identity."""
from isaaclab.utils import configclass
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import UnitreeGo2ParkourTeacherPPORunnerCfg
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_ppo_cfg import UnitreeGo2ParkourStudentPPORunnerCfg
from .contract import robot_contract


@configclass
class RBQ10TeacherRunnerCfg(UnitreeGo2ParkourTeacherPPORunnerCfg):
    experiment_name = "rbq10_extreme_parkour"
    robot_contract: dict = robot_contract()


@configclass
class RBQ10StudentRunnerCfg(UnitreeGo2ParkourStudentPPORunnerCfg):
    student_direction_source: str = "joystick"
    experiment_name = "rbq10_extreme_parkour"
    robot_contract: dict = robot_contract("student")
