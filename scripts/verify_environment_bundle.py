#!/usr/bin/env python3
"""CPU-only integrity check for a copied WBC_base environment bundle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wbc_base.contract import CONTRACT  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> int:
    manifest = json.loads((ROOT / "environment_manifest.json").read_text())
    contract = manifest["policy_contract"]
    expected_contract = {
        "action_dim": CONTRACT.action_dim,
        "actor_frame_dim": CONTRACT.actor_frame_dim,
        "history_length": CONTRACT.history_length,
        "actor_observation_dim": CONTRACT.actor_observation_dim,
        "quaternion_order": CONTRACT.quaternion_order,
        "articulation_root": CONTRACT.articulation_root,
        "root_semantic": CONTRACT.root_semantic,
        "pelvis_body": CONTRACT.pelvis_body,
        "physics_dt_s": CONTRACT.physics_dt_s,
        "policy_dt_s": CONTRACT.policy_dt_s,
        "action_scale": CONTRACT.action_scale,
        "action_offset_source": CONTRACT.action_offset_source,
        "history_reset_semantic": CONTRACT.history_reset_semantic,
    }
    for key, expected in expected_contract.items():
        require_equal(f"policy_contract.{key}", contract[key], expected)

    for path_key, hash_key in (
        ("custom_hand_usd", "custom_hand_usd_sha256"),
        ("custom_hand_usd_base_layer", "custom_hand_usd_base_layer_sha256"),
        ("hybrid_urdf", "hybrid_urdf_sha256"),
    ):
        path = ROOT / manifest["assets"][path_key]
        require_equal(hash_key, sha256(path), manifest["assets"][hash_key])

    for path_key, hash_key in (
        ("env_cfg", "env_cfg_sha256"),
        ("task_registry", "task_registry_sha256"),
    ):
        path = ROOT / manifest["source_contract"][path_key]
        require_equal(hash_key, sha256(path), manifest["source_contract"][hash_key])

    asset_manifest = json.loads(
        (ROOT / "assets/igris_c_custom_hand_torso/manifest.json").read_text()
    )
    require_equal(
        "movable_joint_count",
        asset_manifest["movable_joint_count"],
        manifest["assets"]["movable_joint_count"],
    )
    require_equal(
        "custom_hand_links_per_side",
        asset_manifest["custom_hand_links_per_side"],
        manifest["assets"]["custom_hand_links_per_side"],
    )

    task_source = (ROOT / manifest["source_contract"]["task_registry"]).read_text()
    for role, task_id in manifest["tasks"].items():
        if task_id not in task_source:
            raise RuntimeError(f"tasks.{role}: {task_id!r} is not registered")

    print(
        "environment bundle OK: "
        f"{len(manifest['tasks'])} custom-hand tasks, "
        f"obs={CONTRACT.actor_observation_dim}, action={CONTRACT.action_dim}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

