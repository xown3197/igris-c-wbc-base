#!/usr/bin/env python3
"""Load the torso-root custom-hand USD in Isaac Sim and render one proof image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--asset", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd
import torch
from PIL import Image
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("RENDER_STAGE simulation_context", flush=True)
    sim = SimulationContext(SimulationCfg(dt=0.005, device=args.device))
    ground_cfg = sim_utils.GroundPlaneCfg(color=(0.18, 0.20, 0.23))
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=700.0, color=(0.92, 0.95, 1.0))
    light_cfg.func("/World/DomeLight", light_cfg)
    key_cfg = sim_utils.DistantLightCfg(intensity=550.0, color=(1.0, 0.88, 0.76), angle=1.0)
    key_cfg.func("/World/KeyLight", key_cfg, translation=(2.0, 2.0, 4.0))

    robot_cfg = sim_utils.UsdFileCfg(
        usd_path=str(args.asset),
        activate_contact_sensors=False,
    )
    print("RENDER_STAGE spawn_robot", flush=True)
    robot_cfg.func("/World/Robot", robot_cfg, translation=(0.0, 0.0, 0.98))
    camera = Camera(
        CameraCfg(
            prim_path="/World/AssetCamera",
            update_period=0.0,
            height=960,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=48.0,
                focus_distance=400.0,
                horizontal_aperture=36.0,
                clipping_range=(0.1, 100.0),
            ),
        )
    )
    print("RENDER_STAGE reset", flush=True)
    sim.reset()

    print("RENDER_STAGE camera", flush=True)
    camera.set_world_poses_from_view(
        torch.tensor([[2.8, 0.0, 1.45]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.82]], device=sim.device),
    )

    print("RENDER_STAGE capture", flush=True)
    pixels = None
    for _ in range(20):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
        rgb = camera.data.output.get("rgb")
        if rgb is not None and rgb.numel() > 0:
            pixels = rgb[0, :, :, :3].detach().cpu().numpy()
            break
    if pixels is None:
        raise RuntimeError("Camera sensor did not produce RGB data")
    Image.fromarray(pixels.astype("uint8"), mode="RGB").save(args.output)
    with Image.open(args.output) as captured:
        image_size = [captured.width, captured.height]
    print("RENDER_STAGE saved", flush=True)

    stage = omni.usd.get_context().get_stage()
    revolute_joints = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint)
    ]
    hand_visual_prims = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if "hand" in prim.GetName().lower() and prim.GetTypeName() in ("Mesh", "Xform")
    ]
    audit = {
        "asset": str(args.asset),
        "output": str(args.output),
        "image_size": image_size,
        "revolute_joint_count": len(revolute_joints),
        "revolute_joints": revolute_joints,
        "hand_visual_prim_count": len(hand_visual_prims),
        "has_left_custom_hand": any("l_hand" in path for path in hand_visual_prims),
        "has_right_custom_hand": any("r_hand" in path for path in hand_visual_prims),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
    simulation_app.close()
