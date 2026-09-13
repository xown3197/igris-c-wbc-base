#!/usr/bin/env python3
"""Emit a compact, machine-readable audit of one RSL-RL training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCALAR_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/base_height_low",
    "Episode_Termination/orientation",
    "Episode_Termination/time_out",
)


def tensor_leaves(value: Any, path: str = "root"):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, f"{path}[{index}]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 100, 200, 300, 400, 499])
    args = parser.parse_args()

    checkpoints = sorted(args.run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    events = list(args.run_dir.glob("events.out.tfevents.*"))
    if not checkpoints or len(events) != 1:
        raise SystemExit("expected at least one checkpoint and exactly one TensorBoard event file")

    final_checkpoint = checkpoints[-1]
    payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    leaves = list(tensor_leaves(payload))
    nonfinite = [path for path, value in leaves if not torch.isfinite(value).all()]

    accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))
    scalar_samples: dict[str, dict[str, float]] = {}
    for tag in SCALAR_TAGS:
        if tag not in available_tags:
            continue
        by_step = {sample.step: sample.value for sample in accumulator.Scalars(tag)}
        scalar_samples[tag] = {
            str(step): by_step[step]
            for step in args.steps
            if step in by_step
        }

    optimizer = payload.get("optimizer_state_dict") or {}
    result = {
        "run_dir": str(args.run_dir),
        "checkpoints": [path.name for path in checkpoints],
        "final_checkpoint": final_checkpoint.name,
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "checkpoint_iter": payload.get("iter"),
        "tensor_leaves": len(leaves),
        "tensor_values": sum(value.numel() for _, value in leaves),
        "nonfinite_tensor_paths": nonfinite,
        "optimizer_state_entries": len(optimizer.get("state", {})),
        "event_file": events[0].name,
        "requested_steps": args.steps,
        "scalars": scalar_samples,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if nonfinite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
