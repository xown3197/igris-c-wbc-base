"""Zero-policy-dimension upper-body default-pose hold."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class DefaultJointPositionAction(ActionTerm):
    """Hold named joints at the manufacturer's articulation default positions."""

    cfg: DefaultJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: DefaultJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        joint_ids, joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        if tuple(joint_names) != tuple(cfg.joint_names):
            raise ValueError(
                "upper-body hold joints did not resolve in the requested order: "
                f"expected {tuple(cfg.joint_names)}, got {tuple(joint_names)}"
            )
        self._joint_ids = joint_ids
        self._raw_actions = torch.empty((self.num_envs, 0), device=self.device)
        self._processed_actions = self._asset.data.default_joint_pos[
            :, self._joint_ids
        ].clone()

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 0:
            raise ValueError(f"expected zero-width hold action, got {actions.shape}")
        self._processed_actions.copy_(
            self._asset.data.default_joint_pos[:, self._joint_ids]
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._processed_actions[env_ids] = self._asset.data.default_joint_pos[
            env_ids
        ][:, self._joint_ids]


@configclass
class DefaultJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DefaultJointPositionAction
    joint_names: list[str] = []


__all__ = ["DefaultJointPositionAction", "DefaultJointPositionActionCfg"]

