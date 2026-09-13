"""Manufacturer torso-root Isaac Lab environment overlay for WBC_base."""

from __future__ import annotations

from pathlib import Path

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg

from robros_lab_public.tasks.locomotion.velocity.config.igris_c.base.flat_env_cfg import (
    IGRISCFlatEnvCfg,
)
from robros_lab_public.tasks.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg as ManufacturerObservationsCfg,
)

from .contract import CONTRACT, POLICY_JOINT_NAMES, UPPER_BODY_HOLD_JOINT_NAMES
from .mdp import DefaultJointPositionActionCfg, velocity_body_pose_command


CUSTOM_HAND_USD = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "igris_c_custom_hand_torso"
    / "usd"
    / "igris_c_custom_hand_torso.usd"
)


@configclass
class WBCBaseActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(POLICY_JOINT_NAMES),
        preserve_order=True,
        use_default_offset=True,
        scale=CONTRACT.action_scale,
    )
    upper_body_hold = DefaultJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(UPPER_BODY_HOLD_JOINT_NAMES),
    )


@configclass
class WBCBasePolicyCfg(ObsGroup):
    """58-value torso frame with two frames of history."""

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.03, n_max=0.03),
    )
    command = ObsTerm(
        func=velocity_body_pose_command,
        params={
            "command_name": "base_velocity",
            "body_height_m": CONTRACT.command_body_height_m,
        },
    )
    joint_pos_rel = ObsTerm(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(POLICY_JOINT_NAMES),
                preserve_order=True,
            )
        },
        noise=NoiseModelWithAdditiveBiasCfg(
            noise_cfg=Unoise(n_min=-0.03, n_max=0.03),
            bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
        ),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(POLICY_JOINT_NAMES),
                preserve_order=True,
            )
        },
        noise=Unoise(n_min=-1.0, n_max=1.0),
    )
    previous_raw_action = ObsTerm(
        func=mdp.last_action,
        params={"action_name": "joint_pos"},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = True
        self.concatenate_terms = True
        self.flatten_history_dim = True
        self.history_length = CONTRACT.history_length


@configclass
class WBCBaseObservationsCfg(ManufacturerObservationsCfg):
    policy: WBCBasePolicyCfg = WBCBasePolicyCfg()


@configclass
class WBCBaseFlatEnvCfg(IGRISCFlatEnvCfg):
    """Scratch-training config; inherits the exact manufacturer public plant."""

    actions: WBCBaseActionsCfg = WBCBaseActionsCfg()
    observations: WBCBaseObservationsCfg = WBCBaseObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 50.0
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.35, 0.35)
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.15
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.35
        self.rewards.dof_pos_limits.weight = -8.0
        if self.sim.dt != CONTRACT.physics_dt_s or self.decimation != 4:
            raise ValueError("manufacturer 5 ms / decimation-4 timing changed")


@configclass
class WBCBaseFlatEnvCfg_NOMINAL(WBCBaseFlatEnvCfg):
    """64-env randomization-light diagnostic; not a robustness result.

    The manufacturer's DelayedPD 0--2 step sampling intentionally remains.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        for name in (
            "add_base_mass",
            "add_link_mass",
            "base_com",
            "robot_physics_material",
            "robot_joint_stiffness_and_damping",
            "robot_joint_params",
            "push_robot",
        ):
            setattr(self.events, name, None)
        self.events.reset_robot_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_base.params["velocity_range"] = {
            axis: (0.0, 0.0)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.curriculum.command_levels = None


@configclass
class WBCBaseFlatEnvCfg_PLAY(WBCBaseFlatEnvCfg_NOMINAL):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 20.0
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class WBCBaseCustomHandFlatEnvCfg(WBCBaseFlatEnvCfg):
    """Full-randomization task using the corrected fixed custom-hand visuals."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(CUSTOM_HAND_USD)


@configclass
class WBCBaseCustomHandFlatEnvCfg_NOMINAL(WBCBaseFlatEnvCfg_NOMINAL):
    """Randomization-light custom-hand smoke/training task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(CUSTOM_HAND_USD)


@configclass
class WBCBaseCustomHandFlatEnvCfg_PLAY(WBCBaseFlatEnvCfg_PLAY):
    """One-environment zero-command custom-hand placement task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.spawn.usd_path = str(CUSTOM_HAND_USD)


__all__ = [
    "WBCBaseActionsCfg",
    "WBCBaseCustomHandFlatEnvCfg",
    "WBCBaseCustomHandFlatEnvCfg_NOMINAL",
    "WBCBaseCustomHandFlatEnvCfg_PLAY",
    "WBCBaseFlatEnvCfg",
    "WBCBaseFlatEnvCfg_NOMINAL",
    "WBCBaseFlatEnvCfg_PLAY",
    "WBCBaseObservationsCfg",
    "WBCBasePolicyCfg",
]
