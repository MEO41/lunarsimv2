# SPDX-License-Identifier: BSD-3-Clause
"""Build the lander articulation from a preset.

The lander is one articulation: a root body, and per leg one link (the footpad) on a prismatic
spring-damper joint (the strut). The USD file is written as text, so this part needs no
simulator and is covered by CPU tests.

Simplifications (also listed in M1A_REPORT.md):
- The strut strokes along the body z axis. The real primary struts are inclined.
- The strut is a linear spring-damper. The real strut crushes honeycomb (no spring-back).
- Each footpad is a sphere, so every leg touches the ground in one point.
- The body collision shape is one cylinder around the descent and ascent stages.
"""

from __future__ import annotations

import hashlib
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


def build_lander_usda(preset: LanderPreset) -> str:
    """Return the USD (text format) of the lander articulation for ``preset``."""
    body_inertia = preset.body_inertia
    cylinder_height = preset.body_collision_top_above_center_of_mass + preset.body_collision_bottom_below_center_of_mass
    cylinder_centre = 0.5 * (
        preset.body_collision_top_above_center_of_mass - preset.body_collision_bottom_below_center_of_mass
    )
    leg_inertia = 0.4 * preset.leg_mass * preset.footpad_radius**2  # solid sphere
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
        "        point3f physics:centerOfMass = (0, 0, 0)",
        f"        float3 physics:diagonalInertia = ({body_inertia[0]:.6f}, {body_inertia[1]:.6f}, {body_inertia[2]:.6f})",
        "        quatf physics:principalAxes = (1, 0, 0, 0)",
        "",
        '        def Cylinder "collision" (',
        '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
        "        )",
        "        {",
        '            uniform token axis = "Z"',
        f"            double height = {cylinder_height:.6f}",
        f"            double radius = {preset.body_collision_radius:.6f}",
        f"            double3 xformOp:translate = (0, 0, {cylinder_centre:.6f})",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        "            color3f[] primvars:displayColor = [(0.75, 0.65, 0.3)]",
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
            '        def Sphere "footpad" (',
            '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
            "        )",
            "        {",
            f"            double radius = {preset.footpad_radius:.6f}",
            "            color3f[] primvars:displayColor = [(0.5, 0.5, 0.55)]",
            "        }",
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
