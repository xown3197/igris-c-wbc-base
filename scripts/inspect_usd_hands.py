#!/usr/bin/env python3
"""Inspect hand/wrist prims and physics metadata in a USD asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd", type=Path)
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise SystemExit(f"failed to open {args.usd}")

    selected = []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if not any(token in name for token in ("hand", "wrist", "palm", "finger", "mate_connector")):
            continue
        xformable = UsdGeom.Xformable(prim)
        local = None
        if xformable:
            transform = xformable.GetLocalTransformation()
            matrix = transform[0] if isinstance(transform, tuple) else transform
            local = [list(row) for row in matrix]
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        joint = UsdPhysics.Joint(prim)
        selected.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "active": prim.IsActive(),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "local_transform": local,
                "mass": mass,
                "joint_body0": [str(path) for path in joint.GetBody0Rel().GetTargets()] if joint else [],
                "joint_body1": [str(path) for path in joint.GetBody1Rel().GetTargets()] if joint else [],
            }
        )

    result = {
        "asset": str(args.usd.resolve()),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "selected_prims": selected,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    simulation_app.close()


if __name__ == "__main__":
    main()
