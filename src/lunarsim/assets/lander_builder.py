# SPDX-License-Identifier: BSD-3-Clause
"""Build the lander articulation from a preset.

The lander is one articulation: a root body, and per leg one link (the footpad) on a prismatic
spring-damper joint (the strut). The USD file is written as text, so this part needs no
simulator and is covered by CPU tests.

Collision shapes, all from the preset (see ``configs/landers``):
- descent stage: a cylinder of the stage's width and height (the real stage is an octagon);
- ascent stage: a box of the stage's envelope, on top of the descent stage;
- footpads: a dish (spherical cap, curved side down) of the real diameter and depth, as a
  convex mesh. The real pad swivels on a ball joint; this one is fixed to its leg.
The descent-engine nozzle has no collision shape: the real skirt is crushable. The touchdown
metric reports its clearance above the ground.

Simplifications (also listed in M1A_REPORT.md):
- The strut strokes along the body z axis. The real primary struts are inclined.
- The strut is a linear spring-damper. The real strut crushes honeycomb (no spring-back).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from lunarsim.assets.lander_preset import LanderPreset

LANDER_PRIM_NAME = "Lander"
BODY_LINK_NAME = "body"


def leg_link_name(leg_index: int) -> str:
    """Return the link name of a leg (1-based in the name)."""
    return f"leg_{leg_index + 1}"


def strut_joint_name(leg_index: int) -> str:
    """Return the joint name of a strut (1-based in the name)."""
    return f"strut_{leg_index + 1}"


FOOTPAD_RING_COUNT = 4  # rings of the dish mesh between the lowest point and the rim
FOOTPAD_POINTS_PER_RING = 24


def footpad_dish_mesh(preset: LanderPreset) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Return (points, face vertex counts, face vertex indices) of one dish footpad.

    The dish is a spherical cap with its curved side down. Coordinates are in the leg-link
    frame: the flat top lies at z = 0 and the lowest point at z = -footpad_depth.
    """
    sphere_radius = preset.footpad_sphere_radius
    rim_radius = preset.footpad_radius
    depth = preset.footpad_depth
    points: list[tuple[float, float, float]] = [(0.0, 0.0, -depth)]
    for ring in range(1, FOOTPAD_RING_COUNT + 1):
        ring_radius = rim_radius * ring / FOOTPAD_RING_COUNT
        # Height of the sphere surface below the flat top at this radius.
        height = -depth + (sphere_radius - math.sqrt(sphere_radius**2 - ring_radius**2))
        for index in range(FOOTPAD_POINTS_PER_RING):
            angle = 2.0 * math.pi * index / FOOTPAD_POINTS_PER_RING
            points.append((ring_radius * math.cos(angle), ring_radius * math.sin(angle), height))
    points.append((0.0, 0.0, 0.0))
    counts: list[int] = []
    indices: list[int] = []
    count = FOOTPAD_POINTS_PER_RING

    def ring_point(ring: int, index: int) -> int:
        """Return the point index of point ``index`` on ring ``ring`` (rings start at 1)."""
        return 1 + (ring - 1) * count + index % count

    for index in range(count):  # fan around the lowest point
        counts.append(3)
        indices += [0, ring_point(1, index + 1), ring_point(1, index)]
    for ring in range(1, FOOTPAD_RING_COUNT):
        for index in range(count):
            counts.append(4)
            indices += [ring_point(ring, index), ring_point(ring, index + 1), ring_point(ring + 1, index + 1), ring_point(ring + 1, index)]
    top = len(points) - 1
    for index in range(count):  # flat top
        counts.append(3)
        indices += [top, ring_point(FOOTPAD_RING_COUNT, index), ring_point(FOOTPAD_RING_COUNT, index + 1)]
    return points, counts, indices


def _footpad_lines(preset: LanderPreset) -> list[str]:
    """Return USD lines of the dish footpad collision mesh of one leg."""
    points, counts, indices = footpad_dish_mesh(preset)
    point_text = ", ".join(f"({x:.6f}, {y:.6f}, {z:.6f})" for x, y, z in points)
    return [
        '        def Mesh "footpad" (',
        '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
        "        )",
        "        {",
        '            uniform token physics:approximation = "convexHull"',
        f"            point3f[] points = [{point_text}]",
        f"            int[] faceVertexCounts = [{', '.join(str(value) for value in counts)}]",
        f"            int[] faceVertexIndices = [{', '.join(str(value) for value in indices)}]",
        '            uniform token subdivisionScheme = "none"',
        "            color3f[] primvars:displayColor = [(0.5, 0.5, 0.55)]",
        "        }",
    ]


def _strut_visual_lines(preset: LanderPreset, leg_position: tuple[float, float, float]) -> list[str]:
    """Return USD lines of a thin cylinder from the footpad to the descent stage.

    It is for pictures only: it has no collision shape and no mass, so the physics is the same
    with or without it. It belongs to the leg link, so it slides into the body when the strut
    compresses.
    """
    x, y, z = leg_position
    axis_x, axis_y = preset.vehicle_axis_position
    radial_x, radial_y = x - axis_x, y - axis_y
    radial = math.hypot(radial_x, radial_y)
    stage_radius = 0.5 * preset.descent_stage_width
    attach_height = preset.height_above_center_of_mass(
        preset.descent_stage_bottom_above_footpad_plane + 0.7 * preset.descent_stage_height
    )
    attach = (axis_x + stage_radius * radial_x / radial, axis_y + stage_radius * radial_y / radial, attach_height)
    direction = (attach[0] - x, attach[1] - y, attach[2] - z)
    length = math.sqrt(sum(component * component for component in direction))
    unit = tuple(component / length for component in direction)
    # Shortest rotation that takes the cylinder axis (+z) to the strut direction, as (w, x, y, z).
    quaternion = (1.0 + unit[2], -unit[1], unit[0], 0.0)
    norm = math.sqrt(sum(component * component for component in quaternion))
    w, qx, qy, qz = (component / norm for component in quaternion)
    return [
        "",
        '        def Cylinder "strut_visual"',
        "        {",
        '            uniform token axis = "Z"',
        f"            double height = {length:.6f}",
        "            double radius = 0.08",
        f"            double3 xformOp:translate = ({0.5 * direction[0]:.6f}, {0.5 * direction[1]:.6f}, {0.5 * direction[2]:.6f})",
        f"            quatf xformOp:orient = ({w:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f})",
        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        "            color3f[] primvars:displayColor = [(0.35, 0.35, 0.38)]",
        "        }",
    ]


def build_lander_usda(preset: LanderPreset) -> str:
    """Return the USD (text format) of the lander articulation for ``preset``."""
    body_inertia = preset.body_inertia
    body_centre_of_mass = preset.body_center_of_mass
    descent_x, descent_y, descent_z = preset.descent_stage_center
    ascent_x, ascent_y, ascent_z = preset.ascent_stage_center
    leg_inertia = 0.4 * preset.leg_mass * preset.footpad_radius**2  # solid sphere of the footpad radius
    lines = [
        "#usda 1.0",
        "(",
        f'    defaultPrim = "{LANDER_PRIM_NAME}"',
        "    metersPerUnit = 1",
        "    kilogramsPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "{LANDER_PRIM_NAME}"',
        "{",
        f'    def Xform "{BODY_LINK_NAME}" (',
        '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsArticulationRootAPI"]',
        "    )",
        "    {",
        f"        float physics:mass = {preset.body_mass:.6f}",
        "        point3f physics:centerOfMass = ("
        f"{body_centre_of_mass[0]:.6f}, {body_centre_of_mass[1]:.6f}, {body_centre_of_mass[2]:.6f})",
        f"        float3 physics:diagonalInertia = ({body_inertia[0]:.6f}, {body_inertia[1]:.6f}, {body_inertia[2]:.6f})",
        "        quatf physics:principalAxes = (1, 0, 0, 0)",
        "",
        '        def Cylinder "descent_stage" (',
        '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
        "        )",
        "        {",
        '            uniform token axis = "Z"',
        f"            double height = {preset.descent_stage_height:.6f}",
        f"            double radius = {0.5 * preset.descent_stage_width:.6f}",
        f"            double3 xformOp:translate = ({descent_x:.6f}, {descent_y:.6f}, {descent_z:.6f})",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        "            color3f[] primvars:displayColor = [(0.75, 0.65, 0.3)]",
        "        }",
        "",
        '        def Cube "ascent_stage" (',
        '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
        "        )",
        "        {",
        "            double size = 1",
        f"            double3 xformOp:translate = ({ascent_x:.6f}, {ascent_y:.6f}, {ascent_z:.6f})",
        "            float3 xformOp:scale = ("
        f"{preset.ascent_stage_depth:.6f}, {preset.ascent_stage_width:.6f}, {preset.ascent_stage_height:.6f})",
        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        "            color3f[] primvars:displayColor = [(0.7, 0.7, 0.72)]",
        "        }",
        "    }",
    ]
    for leg_index, (x, y, z) in enumerate(preset.leg_positions):
        lines += [
            "",
            f'    def Xform "{leg_link_name(leg_index)}" (',
            '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
            "    )",
            "    {",
            f"        double3 xformOp:translate = ({x:.6f}, {y:.6f}, {z:.6f})",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            f"        float physics:mass = {preset.leg_mass:.6f}",
            "        point3f physics:centerOfMass = (0, 0, 0)",
            f"        float3 physics:diagonalInertia = ({leg_inertia:.6f}, {leg_inertia:.6f}, {leg_inertia:.6f})",
            "",
            *_footpad_lines(preset),
            *_strut_visual_lines(preset, (x, y, z)),
            "    }",
            "",
            f'    def PhysicsPrismaticJoint "{strut_joint_name(leg_index)}" (',
            '        prepend apiSchemas = ["PhysicsDriveAPI:linear"]',
            "    )",
            "    {",
            f"        rel physics:body0 = </{LANDER_PRIM_NAME}/{BODY_LINK_NAME}>",
            f"        rel physics:body1 = </{LANDER_PRIM_NAME}/{leg_link_name(leg_index)}>",
            '        uniform token physics:axis = "Z"',
            f"        point3f physics:localPos0 = ({x:.6f}, {y:.6f}, {z:.6f})",
            "        point3f physics:localPos1 = (0, 0, 0)",
            "        quatf physics:localRot0 = (1, 0, 0, 0)",
            "        quatf physics:localRot1 = (1, 0, 0, 0)",
            # Joint position = how far the footpad has moved up into the body (strut compression).
            "        float physics:lowerLimit = 0",
            f"        float physics:upperLimit = {preset.strut_stroke:.6f}",
            '        uniform token drive:linear:physics:type = "force"',
            f"        float drive:linear:physics:stiffness = {preset.strut_stiffness:.6f}",
            f"        float drive:linear:physics:damping = {preset.strut_damping:.6f}",
            "        float drive:linear:physics:targetPosition = 0",
            "        float drive:linear:physics:targetVelocity = 0",
            "    }",
        ]
    lines += ["}", ""]
    return "\n".join(lines)


def write_lander_usd(preset: LanderPreset, directory: Path | str) -> Path:
    """Write the lander USD into ``directory`` and return its path.

    The file name contains a hash of the content, so a changed preset never reuses an old file.
    """
    text = build_lander_usda(preset)
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = Path(directory) / f"{preset.name}_{digest}.usda"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(text)
    return path
