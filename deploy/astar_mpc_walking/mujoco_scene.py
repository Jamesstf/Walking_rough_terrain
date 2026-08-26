"""Build a temporary MuJoCo scene with physical obstacles, without editing source XML."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import CircleObstacle, RectangleObstacle


def _merge_section(scene_root: ET.Element, robot_root: ET.Element, tag: str) -> None:
    robot_section = robot_root.find(tag)
    if robot_section is None:
        return
    scene_section = scene_root.find(tag)
    if scene_section is None:
        scene_root.append(copy.deepcopy(robot_section))
        return
    for child in robot_section:
        scene_section.append(copy.deepcopy(child))


def build_augmented_scene_xml(scene_path: str, obstacles) -> str:
    """Inline the robot include, set absolute mesh paths, and add obstacle geoms."""
    scene_file = Path(scene_path).resolve()
    scene_tree = ET.parse(scene_file)
    scene_root = scene_tree.getroot()
    include = scene_root.find("include")
    if include is None:
        raise ValueError(f"Expected one robot <include> in {scene_file}")
    robot_file = (scene_file.parent / include.attrib["file"]).resolve()
    robot_root = ET.parse(robot_file).getroot()
    scene_root.remove(include)

    compiler = robot_root.find("compiler")
    if compiler is not None:
        compiler = copy.deepcopy(compiler)
        mesh_dir = compiler.attrib.get("meshdir", "")
        compiler.attrib["meshdir"] = str((robot_file.parent / mesh_dir).resolve())
        scene_root.insert(0, compiler)

    default = robot_root.find("default")
    if default is not None:
        insertion_index = 1 if compiler is not None else 0
        scene_root.insert(insertion_index, copy.deepcopy(default))

    for section in ("asset", "worldbody", "contact", "equality", "actuator", "sensor"):
        _merge_section(scene_root, robot_root, section)

    worldbody = scene_root.find("worldbody")
    if worldbody is None:
        raise ValueError("Scene does not contain a <worldbody>")
    for index, obstacle in enumerate(obstacles):
        cx, cy = obstacle.center
        if isinstance(obstacle, CircleObstacle):
            attributes = {
                "type": "cylinder",
                "size": f"{obstacle.radius} {0.5 * obstacle.height}",
            }
        elif isinstance(obstacle, RectangleObstacle):
            attributes = {
                "type": "box",
                "size": f"{0.5 * obstacle.size[0]} {0.5 * obstacle.size[1]} {0.5 * obstacle.height}",
            }
        else:
            raise TypeError(f"Unsupported obstacle: {type(obstacle).__name__}")
        attributes.update(
            {
                "name": f"astar_mpc_obstacle_{index}",
                "pos": f"{cx} {cy} {0.5 * obstacle.height}",
                "rgba": "0.75 0.18 0.12 1",
                "friction": "0.8 0.02 0.01",
                "contype": "1",
                "conaffinity": "1",
            }
        )
        ET.SubElement(worldbody, "geom", attributes)

    return ET.tostring(scene_root, encoding="unicode")
