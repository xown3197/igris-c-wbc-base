"""Observation terms that preserve the 58-value WBC actor frame."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv


def velocity_body_pose_command(
    env: ManagerBasedEnv,
    command_name: str,
    body_height_m: float,
) -> torch.Tensor:
    """Expand manufacturer `[vx, vy, wz]` to the WBC 7D command contract."""

    velocity = env.command_manager.get_command(command_name)
    if velocity.shape[-1] != 3:
        raise ValueError(
            f"{command_name!r} must provide [vx, vy, wz], got {velocity.shape}"
        )
    body_targets = velocity.new_tensor(
        [body_height_m, 0.0, 0.0, 0.0]
    ).expand(velocity.shape[0], 4)
    return torch.cat((velocity, body_targets), dim=-1)


__all__ = ["velocity_body_pose_command"]

