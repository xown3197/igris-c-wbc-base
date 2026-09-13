"""Gym registrations. Import only after Isaac Sim AppLauncher starts."""

import gymnasium as gym


_TASKS = {
    "WBC-Base-Flat-IGRISC-v0": "wbc_base.env_cfg:WBCBaseFlatEnvCfg",
    "WBC-Base-Flat-IGRISC-Nominal-v0": "wbc_base.env_cfg:WBCBaseFlatEnvCfg_NOMINAL",
    "WBC-Base-Flat-IGRISC-Play-v0": "wbc_base.env_cfg:WBCBaseFlatEnvCfg_PLAY",
    "WBC-Base-Flat-IGRISC-CustomHand-v0": "wbc_base.env_cfg:WBCBaseCustomHandFlatEnvCfg",
    "WBC-Base-Flat-IGRISC-CustomHand-Nominal-v0": "wbc_base.env_cfg:WBCBaseCustomHandFlatEnvCfg_NOMINAL",
    "WBC-Base-Flat-IGRISC-CustomHand-Play-v0": "wbc_base.env_cfg:WBCBaseCustomHandFlatEnvCfg_PLAY",
}

for task_id, env_cfg_entry_point in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": env_cfg_entry_point,
                "rsl_rl_cfg_entry_point": "wbc_base.agent_cfg:WBCBasePPORunnerCfg",
            },
        )


__all__ = ["_TASKS"]
