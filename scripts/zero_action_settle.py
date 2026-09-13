#!/usr/bin/env python3
"""Run the 1-env, zero-command/zero-action settling gate and save evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument(
    "--task",
    default="WBC-Base-Flat-IGRISC-Play-v0",
    help="Registered task to probe; manufacturer Play is used as the A/B control",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    os.fspath(project_root / "src"),
    os.fspath(args.upstream_root.resolve() / "source" / "robros_lab_public"),
]

from wbc_base.compat import install_action_term_cfg_alias  # noqa: E402

install_action_term_cfg_alias()

import robros_lab_public.tasks  # noqa: E402,F401
import wbc_base.tasks  # noqa: E402,F401

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from wbc_base.contract import CONTRACT  # noqa: E402


def tensor_row(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor[0].detach().cpu().tolist()]


def main() -> int:
    task_id = args.task
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=1)
    env = gym.make(
        task_id,
        cfg=env_cfg,
    )
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action = torch.zeros(
        (1, unwrapped.action_manager.total_action_dim), device=unwrapped.device
    )
    body_ids, body_names = robot.find_bodies(
        [CONTRACT.articulation_root, CONTRACT.pelvis_body], preserve_order=True
    )
    if tuple(body_names) != (CONTRACT.articulation_root, CONTRACT.pelvis_body):
        raise RuntimeError(f"body resolution mismatch: {body_names}")

    initial_height = float(robot.data.root_pos_w[0, 2].item())
    min_height = initial_height
    max_abs_ang_vel = 0.0
    first_done_step: int | None = None
    first_done_terms: dict[str, float] = {}
    final_observation_shape: list[int] | None = None

    for step in range(args.steps):
        observation, _, terminated, truncated, _ = env.step(action)
        policy_obs = observation["policy"] if isinstance(observation, dict) else observation
        final_observation_shape = list(policy_obs.shape)
        root_height = float(robot.data.root_pos_w[0, 2].item())
        min_height = min(min_height, root_height)
        max_abs_ang_vel = max(
            max_abs_ang_vel,
            float(robot.data.root_ang_vel_b[0].abs().max().item()),
        )
        if first_done_step is None and bool((terminated | truncated)[0].item()):
            first_done_step = step + 1
            first_done_terms = {
                name: float(values[0])
                for name, values in unwrapped.termination_manager.get_active_iterable_terms(0)
                if float(values[0]) != 0.0
            }

    result = {
        "schema_version": "wbc_base_zero_action_settle/v1",
        "task": task_id,
        "steps_requested": args.steps,
        "policy_dt_s": CONTRACT.policy_dt_s,
        "duration_s": args.steps * CONTRACT.policy_dt_s,
        "action_dim": int(action.shape[-1]),
        "actor_observation_shape": final_observation_shape,
        "initial_root_height_m": initial_height,
        "minimum_root_height_m": min_height,
        "final_root_pos_w": tensor_row(robot.data.root_pos_w),
        "final_root_quat_w": tensor_row(robot.data.root_quat_w),
        "final_root_ang_vel_b": tensor_row(robot.data.root_ang_vel_b),
        "max_abs_root_ang_vel_rad_s": max_abs_ang_vel,
        "final_torso_pos_w": tensor_row(robot.data.body_pos_w[:, body_ids[0]]),
        "final_pelvis_pos_w": tensor_row(robot.data.body_pos_w[:, body_ids[1]]),
        "first_done_step": first_done_step,
        "first_done_terms": first_done_terms,
        "completed_without_done": first_done_step is None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    env.close()
    return 0 if first_done_step is None else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
