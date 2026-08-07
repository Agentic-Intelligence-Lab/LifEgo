#!/usr/bin/env python3
"""Build a MuJoCo MJCF scene for the AgileX Nero arm setup."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_ROOT = Path("/home/ymq/code/agx_arm_urdf")
NERO_ROOT = URDF_ROOT / "nero"

LINK_COLORS = {
    "base_link": "0.18 0.18 0.20 1",
    "link1": "0.82 0.84 0.86 1",
    "link2": "0.75 0.77 0.80 1",
    "link3": "0.82 0.84 0.86 1",
    "link4": "0.75 0.77 0.80 1",
    "link5": "0.82 0.84 0.86 1",
    "link6": "0.75 0.77 0.80 1",
    "link7": "0.20 0.20 0.22 1",
    "gripper_flange": "0.10 0.10 0.12 1",
    "gripper_base": "0.12 0.12 0.14 1",
    "gripper_link1": "0.08 0.08 0.09 1",
    "gripper_link2": "0.08 0.08 0.09 1",
}

ZERO_QPOS = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
}


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_vec(text: str | None, default: str = "0 0 0") -> list[float]:
    return [float(v) for v in (text or default).split()]


def fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def rpy_to_quat(rpy: list[float]) -> list[float]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll), returned as w x y z.
    return [
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        sy * cp * sr + cy * sp * cr,
        sy * cp * cr - cy * sp * sr,
    ]


def matrix_to_quat(R: np.ndarray) -> list[float]:
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def mesh_name_from_link(link: str) -> str:
    return f"{link}_mesh"


def load_urdf_model() -> tuple[dict, dict, dict[str, list[dict]]]:
    base_tree = ET.parse(NERO_ROOT / "urdf" / "nero_description.urdf")
    root = base_tree.getroot()
    flange_tree = ET.parse(NERO_ROOT / "urdf" / "nero_with_gripper_flange_description.xacro")
    flange_root = flange_tree.getroot()
    gripper_tree = ET.parse(NERO_ROOT / "urdf" / "nero_with_gripper_description.xacro")
    gripper_root = gripper_tree.getroot()

    links: dict[str, ET.Element] = {}
    joints: dict[str, ET.Element] = {}
    for node in list(root) + list(flange_root) + list(gripper_root):
        tag = node.tag.split("}", 1)[-1]
        if tag == "link":
            links[node.attrib["name"]] = node
        elif tag == "joint":
            joints[node.attrib["name"]] = node

    children: dict[str, list[dict]] = {}
    for name, joint in joints.items():
        if joint.attrib.get("type") == "fixed" and name == "world_to_base_link":
            continue
        # URDF uses this as the mimic master only; the visible fingers are
        # gripper_joint1 and gripper_joint2, which MuJoCo controls directly.
        if name == "gripper":
            continue
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append({"name": name, "joint": joint, "child": child})
    return links, joints, children


def copy_meshes(out_dir: Path, links: dict[str, ET.Element]) -> None:
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for link in links:
        src = NERO_ROOT / "meshes" / f"{link}.stl"
        if src.is_file():
            shutil.copy2(src, mesh_dir / src.name)


def add_inertial(body: ET.Element, link_node: ET.Element) -> None:
    inertial = link_node.find("inertial")
    if inertial is None:
        return
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        return
    pos = parse_vec(origin.attrib.get("xyz") if origin is not None else None)
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": fmt(pos),
            "mass": mass.attrib["value"],
            "fullinertia": fmt(
                [
                    inertia.attrib["ixx"],
                    inertia.attrib["iyy"],
                    inertia.attrib["izz"],
                    inertia.attrib["ixy"],
                    inertia.attrib["ixz"],
                    inertia.attrib["iyz"],
                ]
            ),
        },
    )


def add_link_geom(body: ET.Element, link: str) -> None:
    if not (NERO_ROOT / "meshes" / f"{link}.stl").is_file():
        return
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{link}_visual",
            "type": "mesh",
            "mesh": mesh_name_from_link(link),
            "rgba": LINK_COLORS.get(link, "0.75 0.75 0.78 1"),
            "contype": "0",
            "conaffinity": "0",
        },
    )


def add_body_recursive(parent_body: ET.Element, link: str, links: dict, children: dict) -> None:
    add_inertial(parent_body, links[link])
    add_link_geom(parent_body, link)
    if link == "link7":
        ET.SubElement(
            parent_body,
            "site",
            {
                "name": "recorded_flange",
                "pos": "0 0 0",
                "size": "0.020",
                "rgba": "0.15 0.95 0.25 1",
            },
        )
        # Tool-centric TCP: along flange→tip (link7 +X), between flange and gripper tip.
        # deliberate change from controller log tcp_offset [0,0,0.13] along flange +Z.
        ET.SubElement(
            parent_body,
            "site",
            {
                "name": "tcp",
                "pos": "0.13 0 0",
                "size": "0.020",
                "rgba": "0.95 0.15 0.85 1",
            },
        )
    if link == "gripper_flange":
        ET.SubElement(
            parent_body,
            "site",
            {
                "name": "jaw_parallel_flange",
                "pos": "0 0 0",
                "size": "0.018",
                "rgba": "1 0.85 0.1 1",
            },
        )
    if link == "gripper_base":
        # Approximate tip mid along gripper +Z (finger start ~0.138 m).
        ET.SubElement(
            parent_body,
            "site",
            {
                "name": "gripper_tip",
                "pos": "0 0 0.175",
                "size": "0.014",
                "rgba": "1 0.55 0.05 1",
            },
        )

    for entry in children.get(link, []):
        joint = entry["joint"]
        child = entry["child"]
        origin = joint.find("origin")
        body = ET.SubElement(
            parent_body,
            "body",
            {
                "name": child,
                "pos": fmt(parse_vec(origin.attrib.get("xyz") if origin is not None else None)),
                "quat": fmt(rpy_to_quat(parse_vec(origin.attrib.get("rpy") if origin is not None else None))),
            },
        )
        if joint.attrib["type"] == "revolute":
            axis = parse_vec(joint.find("axis").attrib.get("xyz", "0 0 1"))
            limit = joint.find("limit")
            attrs = {
                "name": entry["name"],
                "type": "hinge",
                "axis": fmt(axis),
                "limited": "true",
                "range": f"{limit.attrib['lower']} {limit.attrib['upper']}",
                "damping": "1.0",
                "armature": "0.02",
            }
            ET.SubElement(body, "joint", attrs)
        elif joint.attrib["type"] == "prismatic":
            axis = parse_vec(joint.find("axis").attrib.get("xyz", "0 0 1"))
            limit = joint.find("limit")
            attrs = {
                "name": entry["name"],
                "type": "slide",
                "axis": fmt(axis),
                "limited": "true",
                "range": f"{limit.attrib['lower']} {limit.attrib['upper']}",
                "damping": "2.0",
                "armature": "0.002",
            }
            ET.SubElement(body, "joint", attrs)
        add_body_recursive(body, child, links, children)


def add_axis(worldbody: ET.Element) -> None:
    ET.SubElement(worldbody, "geom", {"name": "base_plus_x_axis", "type": "capsule", "fromto": "0 0 0.025 0.45 0 0.025", "size": "0.008", "rgba": "0.9 0.1 0.1 1"})
    ET.SubElement(worldbody, "geom", {"name": "base_minus_x_front_axis", "type": "capsule", "fromto": "0 0 0.025 -0.45 0 0.025", "size": "0.005", "rgba": "0.55 0.05 0.05 1"})
    ET.SubElement(worldbody, "geom", {"name": "base_plus_y_axis", "type": "capsule", "fromto": "0 0 0.035 0 0.45 0.035", "size": "0.008", "rgba": "0.1 0.75 0.15 1"})
    ET.SubElement(worldbody, "geom", {"name": "base_minus_y_left_axis", "type": "capsule", "fromto": "0 0 0.035 0 -0.45 0.035", "size": "0.005", "rgba": "0.05 0.45 0.08 1"})
    ET.SubElement(worldbody, "site", {"name": "robot_left_front_workspace_xneg_yneg", "pos": "-0.35 -0.25 0.025", "size": "0.025", "rgba": "1 0.8 0.05 1"})


def add_camera_setup(worldbody: ET.Element, camera_y: float, camera_height: float) -> None:
    camera_pos = np.array([0.0, camera_y, camera_height], dtype=np.float64)
    target = np.array([-camera_height, camera_y, 0.0], dtype=np.float64)
    x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = np.array([math.sqrt(0.5), 0.0, math.sqrt(0.5)], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "ego_camera_45deg_585mm",
            "pos": fmt(camera_pos),
            "xyaxes": fmt(list(x_axis) + list(y_axis)),
            "fovy": "70",
        },
    )
    ET.SubElement(worldbody, "body", {"name": "camera_body", "pos": fmt(camera_pos)})
    camera_body = worldbody.find("body[@name='camera_body']")
    ET.SubElement(camera_body, "geom", {"name": "camera_box", "type": "box", "size": "0.035 0.022 0.018", "rgba": "0.05 0.08 0.10 1"})
    ET.SubElement(worldbody, "geom", {"name": "camera_optical_axis_projection", "type": "capsule", "fromto": f"{fmt(camera_pos)} {fmt(target)}", "size": "0.004", "rgba": "0.1 0.35 1 0.75"})
    ET.SubElement(worldbody, "site", {"name": "camera_table_intersection", "pos": fmt(target), "size": "0.018", "rgba": "0.1 0.35 1 1"})


def load_real_flange_points(path: Path | None) -> list[list[float]]:
    if path is None:
        return []
    if not path.is_file():
        return []
    pts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "sample":
            pts.append(rec["poses"]["flange_pose"][:3])
    return pts


def load_real_flange_poses(path: Path | None) -> list[tuple[list[float], list[float]]]:
    if path is None:
        return []
    if not path.is_file():
        return []
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "sample":
            pose = rec["poses"]["flange_pose"]
            poses.append((pose[:3], rpy_to_quat(pose[3:6])))
    return poses


def load_humanego_points(path: Path) -> list[list[float]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = []
    for rec in data.get("records", []):
        if rec.get("valid"):
            pts.append(rec["T_ee_in_base"]["translation_m"])
    return pts


def load_humanego_poses(path: Path) -> list[tuple[list[float], list[float]]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = []
    for rec in data.get("records", []):
        if rec.get("valid"):
            T = np.array(rec["T_ee_in_base"]["T"], dtype=np.float64)
            poses.append((T[:3, 3].tolist(), matrix_to_quat(T[:3, :3])))
    return poses


def load_first_humanego_pose(path: Path) -> tuple[list[float], list[float]] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for rec in data.get("records", []):
        if rec.get("valid"):
            T = np.array(rec["T_ee_in_base"]["T"], dtype=np.float64)
            return T[:3, 3].tolist(), matrix_to_quat(T[:3, :3])
    return None


def add_path_points(worldbody: ET.Element, name: str, pts: list[list[float]], rgba: str, every: int) -> None:
    for i, p in enumerate(pts):
        if i % every != 0 and i != len(pts) - 1:
            continue
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": f"{name}_{i:03d}",
                "pos": fmt(p),
                "size": "0.008",
                "rgba": rgba,
            },
        )


def add_endpoint_sites(worldbody: ET.Element, name: str, pts: list[list[float]], start_rgba: str, end_rgba: str) -> None:
    if not pts:
        return
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": f"{name}_start",
            "pos": fmt(pts[0]),
            "size": "0.022",
            "rgba": start_rgba,
        },
    )
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": f"{name}_end",
            "pos": fmt(pts[-1]),
            "size": "0.022",
            "rgba": end_rgba,
        },
    )


def add_pose_frame_body(
    worldbody: ET.Element,
    name: str,
    pos: list[float],
    quat: list[float],
    origin_rgba: str,
    length: float,
    radius: float,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": name,
            "pos": fmt(pos),
            "quat": fmt(quat),
        },
    )
    ET.SubElement(body, "geom", {"name": f"{name}_origin", "type": "sphere", "size": fmt([radius * 2.0]), "rgba": origin_rgba, "contype": "0", "conaffinity": "0"})
    ET.SubElement(body, "geom", {"name": f"{name}_x", "type": "capsule", "fromto": fmt([0, 0, 0, length, 0, 0]), "size": fmt([radius]), "rgba": "1 0.05 0.05 0.9", "contype": "0", "conaffinity": "0"})
    ET.SubElement(body, "geom", {"name": f"{name}_y", "type": "capsule", "fromto": fmt([0, 0, 0, 0, length, 0]), "size": fmt([radius]), "rgba": "0.05 0.8 0.1 0.9", "contype": "0", "conaffinity": "0"})
    ET.SubElement(body, "geom", {"name": f"{name}_z", "type": "capsule", "fromto": fmt([0, 0, 0, 0, 0, length]), "size": fmt([radius]), "rgba": "0.1 0.3 1 0.9", "contype": "0", "conaffinity": "0"})


def add_pose_frames(
    worldbody: ET.Element,
    name: str,
    poses: list[tuple[list[float], list[float]]],
    origin_rgba: str,
    every: int,
    length: float = 0.055,
    radius: float = 0.0035,
) -> None:
    for i, (pos, quat) in enumerate(poses):
        if i % every != 0 and i != len(poses) - 1:
            continue
        add_pose_frame_body(worldbody, f"{name}_{i:03d}", pos, quat, origin_rgba, length, radius)


def add_frame_legend_markers(worldbody: ET.Element) -> None:
    """Mocap bodies for flange / controller TCP / gripper tip (color legend in replay)."""
    specs = [
        ("vis_flange_marker", "0.15 0.95 0.25 1"),
        ("vis_tcp_marker", "0.95 0.15 0.85 1"),
        ("vis_tip_marker", "1 0.55 0.05 1"),
    ]
    for i, (name, rgba) in enumerate(specs):
        marker = ET.SubElement(
            worldbody,
            "body",
            {
                "name": name,
                "mocap": "true",
                "pos": f"0 {0.05 * (i - 1):.2f} 0.20",
                "quat": "1 0 0 0",
            },
        )
        ET.SubElement(
            marker,
            "geom",
            {
                "name": f"{name}_origin",
                "type": "sphere",
                "size": "0.016",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )
        for axis, fromto, axis_rgba in (
            ("x", "0 0 0 0.07 0 0", "1 0.15 0.15 1"),
            ("y", "0 0 0 0 0.07 0", "0.15 0.9 0.15 1"),
            ("z", "0 0 0 0 0 0.07", "0.15 0.35 1 1"),
        ):
            ET.SubElement(
                marker,
                "geom",
                {
                    "name": f"{name}_{axis}",
                    "type": "capsule",
                    "fromto": fromto,
                    "size": "0.0045",
                    "rgba": axis_rgba,
                    "contype": "0",
                    "conaffinity": "0",
                },
            )


def add_humanego_eef_marker(worldbody: ET.Element, humanego_eef_path: Path) -> None:
    pose = load_first_humanego_pose(humanego_eef_path)
    pos, quat = pose if pose is not None else ([0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    marker = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "humanego_eef_marker",
            "mocap": "true",
            "pos": fmt(pos),
            "quat": fmt(quat),
        },
    )
    ET.SubElement(marker, "geom", {"name": "humanego_eef_origin", "type": "sphere", "size": "0.018", "rgba": "0.05 0.75 1 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"name": "humanego_eef_x_axis", "type": "capsule", "fromto": "0 0 0 0.075 0 0", "size": "0.005", "rgba": "1 0.05 0.05 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"name": "humanego_eef_y_axis", "type": "capsule", "fromto": "0 0 0 0 0.075 0", "size": "0.005", "rgba": "0.05 0.8 0.1 1", "contype": "0", "conaffinity": "0"})
    ET.SubElement(marker, "geom", {"name": "humanego_eef_z_axis", "type": "capsule", "fromto": "0 0 0 0 0 0.075", "size": "0.005", "rgba": "0.1 0.3 1 1", "contype": "0", "conaffinity": "0"})
    add_frame_legend_markers(worldbody)


def build_scene(
    out_dir: Path,
    include_trajectories: bool,
    camera_y: float,
    camera_height: float,
    humanego_eef_path: Path,
    realbot_path: Path | None,
    real_path_every: int,
    humanego_path_every: int,
    real_frame_every: int,
    humanego_frame_every: int,
) -> Path:
    links, _joints, children = load_urdf_model()
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_meshes(out_dir, links)

    root = ET.Element("mujoco", {"model": "nero_robot_table_camera_scene"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "meshes", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    ET.SubElement(root, "statistic", {"center": "-0.15 0 0.28", "extent": "1.2"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.65 0.65 0.65", "specular": "0.15 0.15 0.15"})
    ET.SubElement(visual, "rgba", {"haze": "0.85 0.90 0.95 1"})
    ET.SubElement(visual, "global", {"azimuth": "145", "elevation": "-25", "offwidth": "1280", "offheight": "720"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": "0.74 0.74 0.72", "rgb2": "0.86 0.86 0.84", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "table_mat", "texture": "grid", "texrepeat": "8 8", "rgba": "0.78 0.78 0.74 1"})
    ET.SubElement(asset, "material", {"name": "workspace_mat", "rgba": "1 0.82 0.12 0.25"})
    for link in links:
        mesh = out_dir / "meshes" / f"{link}.stl"
        if mesh.is_file():
            ET.SubElement(asset, "mesh", {"name": mesh_name_from_link(link), "file": mesh.name})

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "-0.6 -0.8 1.2", "dir": "0.4 0.5 -1", "diffuse": "0.8 0.8 0.8"})
    ET.SubElement(worldbody, "geom", {"name": "table_top_z0", "type": "box", "pos": "-0.25 0  -0.025", "size": "0.75 0.65 0.025", "material": "table_mat"})
    ET.SubElement(worldbody, "geom", {"name": "left_front_workspace_patch", "type": "box", "pos": "-0.35 -0.25 0.002", "size": "0.30 0.24 0.002", "material": "workspace_mat", "contype": "0", "conaffinity": "0"})
    add_axis(worldbody)
    add_camera_setup(worldbody, camera_y=camera_y, camera_height=camera_height)

    base_body = ET.SubElement(worldbody, "body", {"name": "base_link", "pos": "0 0 0", "quat": "1 0 0 0"})
    add_body_recursive(base_body, "base_link", links, children)
    add_humanego_eef_marker(worldbody, humanego_eef_path)

    if include_trajectories:
        real_points = load_real_flange_points(realbot_path)
        humanego_points = load_humanego_points(humanego_eef_path)
        add_path_points(worldbody, "real_flange_path", real_points, "1 0.65 0.05 0.75", every=real_path_every)
        add_path_points(worldbody, "humanego_eef_path", humanego_points, "0.05 0.75 1 0.75", every=humanego_path_every)
        add_endpoint_sites(worldbody, "real_flange", real_points, "1 0.25 0.05 1", "1 0.95 0.05 1")
        add_endpoint_sites(worldbody, "humanego_eef", humanego_points, "0.0 1.0 1.0 1", "0.25 0.1 1.0 1")
        add_pose_frames(worldbody, "real_flange_frame", load_real_flange_poses(realbot_path), "1 0.65 0.05 1", every=real_frame_every)
        add_pose_frames(worldbody, "humanego_eef_frame", load_humanego_poses(humanego_eef_path), "0.05 0.75 1 1", every=humanego_frame_every)

    actuator = ET.SubElement(root, "actuator")
    for joint in ZERO_QPOS:
        ET.SubElement(actuator, "position", {"name": f"{joint}_pos", "joint": joint, "kp": "80", "ctrlrange": "-3.14 3.14"})
    ET.SubElement(actuator, "position", {"name": "gripper_joint1_pos", "joint": "gripper_joint1", "kp": "40", "ctrlrange": "0 0.05"})
    ET.SubElement(actuator, "position", {"name": "gripper_joint2_pos", "joint": "gripper_joint2", "kp": "40", "ctrlrange": "-0.05 0"})

    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "zero",
            "qpos": fmt(list(ZERO_QPOS.values()) + [0.05, -0.05]),
            "ctrl": fmt(list(ZERO_QPOS.values()) + [0.05, -0.05]),
        },
    )

    indent(root)
    scene_path = out_dir / "scene.xml"
    ET.ElementTree(root).write(scene_path, encoding="utf-8", xml_declaration=True)
    return scene_path


def write_readme(
    out_dir: Path,
    scene_path: Path,
    camera_y: float,
    camera_height: float,
    humanego_eef_path: Path,
    realbot_path: Path | None,
) -> None:
    readme = f"""# Nero MuJoCo Scene

Generated from `/home/ymq/code/agx_arm_urdf/nero`.

Files:
- `scene.xml`: MuJoCo MJCF scene.
- `meshes/*.stl`: local Nero mesh assets copied from the URDF package.

Scene convention:
- Robot base is at table height `z=0`.
- Base `+x` points backward, base `-x` is the front direction.
- Base `+y` points to robot right, base `-y` points to robot left.
- The highlighted yellow patch is the robot-left-front workspace (`x<0, y<0`).
- Camera is at `[0, {camera_y}, {camera_height}]` m, height `{camera_height}` m, pitched down 45 deg.
- Camera optical-axis table intersection is `[-{camera_height}, {camera_y}, 0]`, so its table projection points along base `-x`.

Optional trajectory markers:
- Yellow sites: real robot `flange_pose` samples from `{realbot_path}`.
- Cyan sites: HumanEgo/WiLoR exported EE samples from `{humanego_eef_path}`.
- Small RGB triads: sampled orientation frames for both trajectories. The origin color
  distinguishes the source: yellow/orange for real flange, cyan for HumanEgo EEF.
- Larger endpoint sites mark start/end: real start red-orange, real end yellow;
  HumanEgo start cyan, HumanEgo end violet.

Important frames:
- `recorded_flange`: `link7` origin. This matches JSONL `poses.flange_pose`.
- `tcp`: tool-centric TCP = `flange + 0.13 m` along flange/link7 +X (toward gripper tip).
  This differs from JSONL `poses.tcp_pose` which used `[0,0,0.13]` along flange +Z.
- `gripper_tip`: approximate tip mid on `gripper_base` at `+0.175 Z` in gripper frame.
- `jaw_parallel_flange`: the extra URDF `gripper_flange` fixed adapter after `link7`; it is offset
  from `recorded_flange` and has a different orientation (jaw +Z ≈ flange +X / base left).
- `humanego_eef_marker`: a mocap body used by `patches/replay_humanego_eef_mujoco.py`
  to show the moving HumanEgo/WiLoR EEF frame.
- Full jaw-parallel gripper geometry is attached after `jaw_parallel_flange`.
- `gripper_joint1` and `gripper_joint2` are controlled directly in MuJoCo; they mirror the
  URDF mimic pair with approximately half the recorded gripper width on each finger.

Load with MuJoCo after installing `mujoco` in an environment:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("{scene_path}")
data = mujoco.MjData(model)
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/mujoco_nero_scene")
    parser.add_argument("--camera-y", type=float, default=-0.45)
    parser.add_argument("--camera-height", type=float, default=0.585)
    parser.add_argument(
        "--humanego-eef",
        default="outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json",
    )
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument("--real-path-every", type=int, default=4)
    parser.add_argument("--humanego-path-every", type=int, default=6)
    parser.add_argument("--real-frame-every", type=int, default=8)
    parser.add_argument("--humanego-frame-every", type=int, default=12)
    parser.add_argument("--no-trajectories", action="store_true")
    args = parser.parse_args()

    out_dir = as_abs(args.out)
    humanego_eef_path = as_abs(args.humanego_eef)
    realbot_path = None if args.realbot == "" else as_abs(args.realbot)
    scene_path = build_scene(
        out_dir=out_dir,
        include_trajectories=not args.no_trajectories,
        camera_y=args.camera_y,
        camera_height=args.camera_height,
        humanego_eef_path=humanego_eef_path,
        realbot_path=realbot_path,
        real_path_every=args.real_path_every,
        humanego_path_every=args.humanego_path_every,
        real_frame_every=args.real_frame_every,
        humanego_frame_every=args.humanego_frame_every,
    )
    write_readme(out_dir, scene_path, args.camera_y, args.camera_height, humanego_eef_path, realbot_path)
    print(f"Wrote {scene_path}")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
