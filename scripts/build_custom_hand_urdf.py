#!/usr/bin/env python3
"""Build a torso-root manufacturer URDF with the custom hand visuals fixed on."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


HAND_SIDES = {
    # The manufacturer hand frame already contains roll=pi.  Conjugating the
    # source wrist-to-hand rotation by that frame reverses the yaw sign.
    "l_hand": ("Left_Hand", "1.5707963267948966"),
    "r_hand": ("Right_Hand", "-1.5707963267948966"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manufacturer-urdf", type=Path, required=True)
    parser.add_argument("--manufacturer-meshes", type=Path, required=True)
    parser.add_argument("--custom-urdf", type=Path, required=True)
    parser.add_argument("--custom-hand-meshes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_urdf_dir = args.output_root / "urdf"
    output_mesh_dir = args.output_root / "meshes"
    output_urdf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.manufacturer_meshes, output_mesh_dir, dirs_exist_ok=True)
    shutil.copytree(
        args.custom_hand_meshes,
        output_mesh_dir / "igris_c_rcv_hand",
        dirs_exist_ok=True,
    )

    manufacturer = ET.parse(args.manufacturer_urdf).getroot()
    custom = ET.parse(args.custom_urdf).getroot()
    manufacturer.set("name", "IGRIS_C_custom_hand_torso")

    for custom_prefix, (manufacturer_hand, yaw) in HAND_SIDES.items():
        stock_link = next(link for link in manufacturer.findall("link") if link.get("name") == manufacturer_hand)
        for visual in list(stock_link.findall("visual")):
            stock_link.remove(visual)

        for link in custom.findall("link"):
            if not link.get("name", "").startswith(custom_prefix):
                continue
            copied = copy.deepcopy(link)
            for mesh in copied.findall(".//mesh"):
                filename = mesh.get("filename", "")
                marker = "package://igris_c_description/meshes/"
                if not filename.startswith(marker):
                    raise ValueError(f"unexpected custom mesh URI: {filename}")
                mesh.set("filename", "../meshes/" + filename.removeprefix(marker))
            manufacturer.append(copied)

        root_joint = ET.Element("joint", {"name": f"{custom_prefix}_visual_mount", "type": "fixed"})
        ET.SubElement(root_joint, "parent", {"link": manufacturer_hand})
        ET.SubElement(root_joint, "child", {"link": f"{custom_prefix}_root"})
        ET.SubElement(root_joint, "origin", {"xyz": "0 0 0", "rpy": f"0 0 {yaw}"})
        manufacturer.append(root_joint)

        for joint in custom.findall("joint"):
            name = joint.get("name", "")
            if not name.startswith(custom_prefix) or name == custom_prefix:
                continue
            copied = copy.deepcopy(joint)
            copied.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller"):
                for child in list(copied.findall(tag)):
                    copied.remove(child)
            manufacturer.append(copied)

    ET.indent(manufacturer, space="  ")
    output_urdf = output_urdf_dir / "igris_c_custom_hand_torso.urdf"
    ET.ElementTree(manufacturer).write(output_urdf, encoding="utf-8", xml_declaration=True)

    parsed = ET.parse(output_urdf).getroot()
    links = parsed.findall("link")
    joints = parsed.findall("joint")
    movable = [joint.get("name") for joint in joints if joint.get("type") not in ("fixed", None)]
    manifest = {
        "schema_version": "igris_c_custom_hand_torso/v1",
        "manufacturer_urdf_sha256": sha256(args.manufacturer_urdf),
        "custom_urdf_sha256": sha256(args.custom_urdf),
        "output_urdf_sha256": sha256(output_urdf),
        "link_count": len(links),
        "joint_count": len(joints),
        "movable_joint_count": len(movable),
        "movable_joints": movable,
        "custom_hand_links_per_side": 55,
        "physics_surrogate": {
            "mass_and_collision": "manufacturer Left_Hand and Right_Hand links retained",
            "visuals": "custom hand subtree",
            "custom_subtree_joints": "fixed",
            "training_readiness": "visual alignment only; exact custom mass and collision not validated"
        }
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
