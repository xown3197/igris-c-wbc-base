from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = (ROOT / "src/wbc_base/env_cfg.py").read_text(encoding="utf-8")
ACTION_SOURCE = (ROOT / "src/wbc_base/mdp/actions.py").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
TASK_SOURCE = (ROOT / "src/wbc_base/tasks.py").read_text(encoding="utf-8")


def test_env_uses_manufacturer_config_and_explicit_policy_order():
    assert "IGRISCFlatEnvCfg" in ENV_SOURCE
    assert "joint_names=list(POLICY_JOINT_NAMES)" in ENV_SOURCE
    assert "preserve_order=True" in ENV_SOURCE
    assert "history_length = CONTRACT.history_length" in ENV_SOURCE


def test_upper_body_hold_consumes_no_policy_dimension():
    assert "return 0" in ACTION_SOURCE
    assert "default_joint_pos" in ACTION_SOURCE


def test_no_custom_pelvis_asset_or_duplicate_delay_is_configured():
    forbidden = (
        "igris_c_corrected.unlocked.usd",
        "foot_override.usda",
        "IsaacTorsoObservationHook",
        "TargetPipeline",
    )
    assert not any(token in ENV_SOURCE for token in forbidden)
    assert "고정 5 ms custom target delay" in README


def test_custom_hand_task_changes_only_the_robot_asset_entrypoint():
    assert "self.scene.robot.spawn.usd_path = str(CUSTOM_HAND_USD)" in ENV_SOURCE
    assert "WBC-Base-Flat-IGRISC-CustomHand-Nominal-v0" in TASK_SOURCE
    assert "igris_c_custom_hand_torso.usd" in ENV_SOURCE
