"""CPU-testable policy, frame, and upstream contracts."""

from __future__ import annotations

from dataclasses import dataclass


POLICY_JOINT_NAMES = (
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Yaw_Left",
    "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Roll_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Right",
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
)

UPPER_BODY_HOLD_JOINT_NAMES = (
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
)

OBSERVATION_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("command", 7),
    ("joint_pos_rel", 15),
    ("joint_vel", 15),
    ("previous_raw_action", 15),
)


@dataclass(frozen=True)
class WBCBaseContract:
    upstream_commit: str = "b72d87190ee6893cf6f3db8dc417d1750518040a"
    upstream_asset_sha256: str = (
        "ef96bb9d078d311233c08e08d2334fd0d154ac89627111bf77bbb9c969966eea"
    )
    articulation_root: str = "base_link"
    root_semantic: str = "torso"
    pelvis_body: str = "Link_Waist_Pitch"
    quaternion_order: str = "wxyz"
    physics_dt_s: float = 0.005
    policy_dt_s: float = 0.02
    action_scale: float = 0.25
    history_length: int = 2
    history_reset_semantic: str = "repeat_first_observation"
    action_offset_source: str = "manufacturer_public_default_pose"
    command_body_height_m: float = 0.94

    @property
    def action_dim(self) -> int:
        return len(POLICY_JOINT_NAMES)

    @property
    def actor_frame_dim(self) -> int:
        return sum(size for _, size in OBSERVATION_TERMS)

    @property
    def actor_observation_dim(self) -> int:
        return self.actor_frame_dim * self.history_length

    def validate(self) -> None:
        if self.articulation_root != "base_link" or self.root_semantic != "torso":
            raise ValueError("WBC_base requires the manufacturer torso-root contract")
        if self.pelvis_body != "Link_Waist_Pitch":
            raise ValueError("WBC_base pelvis alias must be Link_Waist_Pitch")
        if self.quaternion_order != "wxyz":
            raise ValueError("WBC_base quaternion order must be wxyz")
        if len(set(POLICY_JOINT_NAMES)) != 15:
            raise ValueError("policy joint order must contain 15 unique joints")
        if set(POLICY_JOINT_NAMES) & set(UPPER_BODY_HOLD_JOINT_NAMES):
            raise ValueError("policy and upper-body hold joints must be disjoint")
        if self.actor_frame_dim != 58 or self.actor_observation_dim != 116:
            raise ValueError("actor observation must be 58 x 2 = 116")
        if self.history_reset_semantic != "repeat_first_observation":
            raise ValueError("deployment must reproduce IsaacLab first-observation history fill")
        if self.action_offset_source != "manufacturer_public_default_pose":
            raise ValueError("action offset must remain the public Isaac default pose")
        if round(self.policy_dt_s / self.physics_dt_s) != 4:
            raise ValueError("policy period must equal four physics steps")


CONTRACT = WBCBaseContract()
CONTRACT.validate()


__all__ = [
    "CONTRACT",
    "OBSERVATION_TERMS",
    "POLICY_JOINT_NAMES",
    "UPPER_BODY_HOLD_JOINT_NAMES",
    "WBCBaseContract",
]
