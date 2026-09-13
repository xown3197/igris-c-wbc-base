import math
import xml.etree.ElementTree as ET
from pathlib import Path


ASSET_URDF = (
    Path(__file__).parents[1]
    / "assets"
    / "igris_c_custom_hand_torso"
    / "urdf"
    / "igris_c_custom_hand_torso.urdf"
)


def test_custom_hands_are_mounted_on_their_own_sides() -> None:
    root = ET.parse(ASSET_URDF).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    expected = {
        "l_hand_visual_mount": ("Left_Hand", "l_hand_root", math.pi / 2),
        "r_hand_visual_mount": ("Right_Hand", "r_hand_root", -math.pi / 2),
    }
    for name, (parent, child, yaw) in expected.items():
        joint = joints[name]
        assert joint.find("parent").get("link") == parent
        assert joint.find("child").get("link") == child
        actual_yaw = float(joint.find("origin").get("rpy").split()[2])
        assert actual_yaw == yaw
