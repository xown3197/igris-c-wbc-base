from wbc_base.contract import (
    CONTRACT,
    OBSERVATION_TERMS,
    POLICY_JOINT_NAMES,
    UPPER_BODY_HOLD_JOINT_NAMES,
)


def test_policy_contract_dimensions_and_timing():
    CONTRACT.validate()
    assert CONTRACT.action_dim == 15
    assert CONTRACT.actor_frame_dim == 58
    assert CONTRACT.actor_observation_dim == 116
    assert CONTRACT.policy_dt_s / CONTRACT.physics_dt_s == 4


def test_frame_term_order_is_deployment_compatible():
    assert OBSERVATION_TERMS == (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("command", 7),
        ("joint_pos_rel", 15),
        ("joint_vel", 15),
        ("previous_raw_action", 15),
    )


def test_policy_and_hold_partition_the_manufacturer_23_axes():
    assert len(POLICY_JOINT_NAMES) == 15
    assert len(UPPER_BODY_HOLD_JOINT_NAMES) == 8
    assert not set(POLICY_JOINT_NAMES) & set(UPPER_BODY_HOLD_JOINT_NAMES)
    assert len(set(POLICY_JOINT_NAMES + UPPER_BODY_HOLD_JOINT_NAMES)) == 23


def test_root_is_native_torso_not_pelvis_alias():
    assert CONTRACT.articulation_root == "base_link"
    assert CONTRACT.root_semantic == "torso"
    assert CONTRACT.pelvis_body == "Link_Waist_Pitch"


def test_deployment_reset_does_not_change_policy_input_contract():
    assert CONTRACT.history_reset_semantic == "repeat_first_observation"
    assert CONTRACT.action_offset_source == "manufacturer_public_default_pose"
