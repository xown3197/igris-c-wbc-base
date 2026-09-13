#!/usr/bin/env python3
"""Report composed hand transforms from a converted USD without launching Kit."""

from __future__ import annotations

import argparse
import json

from pxr import Usd, UsdGeom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd)
    if stage is None:
        raise SystemExit(f"failed to open {args.usd}")

    wanted = {"Left_Hand", "Right_Hand", "l_hand_root", "r_hand_root"}
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    result = {}
    for prim in stage.Traverse():
        if prim.GetName() not in wanted or not prim.IsA(UsdGeom.Xformable):
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = matrix.ExtractTranslation()
        aligned = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        result[str(prim.GetPath())] = {
            "translation": [translation[0], translation[1], translation[2]],
            "bound_min": [minimum[0], minimum[1], minimum[2]],
            "bound_max": [maximum[0], maximum[1], maximum[2]],
            "matrix": [[matrix[row][column] for column in range(4)] for row in range(4)],
        }

    for side in ("l_hand", "r_hand"):
        mesh_bounds = []
        for prim in stage.Traverse():
            if side not in str(prim.GetPath()) or not prim.IsA(UsdGeom.Mesh):
                continue
            aligned = bounds.ComputeWorldBound(prim).ComputeAlignedBox()
            minimum = aligned.GetMin()
            maximum = aligned.GetMax()
            if minimum[0] > maximum[0]:
                continue
            mesh_bounds.append(
                {
                    "path": str(prim.GetPath()),
                    "min": [minimum[0], minimum[1], minimum[2]],
                    "max": [maximum[0], maximum[1], maximum[2]],
                }
            )
        result[f"{side}_mesh_bounds"] = mesh_bounds
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
