"""RBQ10 task registration; safe to import before launching Isaac Sim."""
import gymnasium as gym

for role in ("Teacher", "Student"):
    for suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY"), ("-Eval", "_EVAL")):
        gym.register(
            id=f"Isaac-Extreme-Parkour-{role}-Rainbow-RBQ10{suffix}-v0",
            entry_point="rbq10_parkour.env:RBQ10ParkourEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"rbq10_parkour.config:RBQ10{role}EnvCfg{cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"rbq10_parkour.agents:RBQ10{role}RunnerCfg",
            },
        )

# Separate registration and entry point: never replace the existing front-only task.
gym.register(
    id="Isaac-Extreme-Parkour-Student-MultiDepth-Rainbow-RBQ10-v0",
    entry_point="rbq10_parkour.env:RBQ10ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "rbq10_parkour.multiview_config:RBQ10MultiViewStudentEnvCfg",
        "rsl_rl_cfg_entry_point": "rbq10_parkour.multiview_config:RBQ10MultiViewStudentRunnerCfg",
    },
)
