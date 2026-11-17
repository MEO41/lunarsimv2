# SPDX-License-Identifier: BSD-3-Clause
"""Render recorded episodes (see ``replay_recording.py``) on a plain stage without physics.

Only the renderer reads this stage, so every lander is placed by setting its transforms directly
from the recording: body pose, strut strokes, and (when contacts were recorded) one arrow per
footpad along the contact force from that leg's sensor. The terrain is rebuilt from the same
scenario, overrides and seed, so it is the same surface that PhysX collided with.

Needs Isaac Sim with the camera renderer; import it only in a process started with cameras on.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy

ARROW_METRES_PER_NEWTON = 1.0 / 3000.0  # 1 m of arrow per 3 kN (about the rest load of one pad)
ARROW_MAXIMUM_LENGTH_M = 3.0
ARROW_SHOWN_ABOVE_NEWTON = 100.0  # the touchdown metric's contact threshold
ARROW_RADIUS_M = 0.05
ARROW_HEAD_LENGTH_M = 0.25
ARROW_HEAD_RADIUS_M = 0.12
ARROW_OUTWARD_SHIFT_M = 0.9  # arrows stand beside their pad, away from the vehicle axis
TOUCHING_COLOR = (0.15, 0.85, 0.25)
BODY_COLOR = (0.95, 0.25, 0.25)


def look_at_matrix(eye, target):
    """Return the USD camera transform that looks from ``eye`` at ``target`` with z up."""
    from pxr import Gf

    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)).GetInverse()


def rotation_from_z_to(direction: numpy.ndarray):
    """Return the USD quaternion that turns the +z axis onto ``direction`` (a unit vector)."""
    from pxr import Gf

    z = numpy.array([0.0, 0.0, 1.0])
    axis = numpy.cross(z, direction)
    sine, cosine = numpy.linalg.norm(axis), float(numpy.dot(z, direction))
    if sine < 1e-9:
        return Gf.Quatf(1.0, 0.0, 0.0, 0.0) if cosine > 0 else Gf.Quatf(0.0, 1.0, 0.0, 0.0)
    axis = axis / sine
    angle = math.atan2(sine, cosine)
    half = 0.5 * angle
    return Gf.Quatf(math.cos(half), *(float(value) * math.sin(half) for value in axis))


class ReplayRenderer:
    """A stage with the terrain of a scenario, landers to place, force arrows and one camera."""

    def __init__(self, simulation_app, scenario_name: str, scenario_overrides: list[str], number_of_landers: int,
                 width: int = 1280, height: int = 720) -> None:
        """Build the stage, the landers (hidden), the arrows, the camera and its image reader."""
        import omni.replicator.core as replicator
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, Vt

        from lunarsim.assets.lander_builder import leg_link_name, write_lander_usd
        from lunarsim.assets.lander_preset import lander_preset_from_dictionary
        from lunarsim.core.binding import BindingContext
        from lunarsim.core.message_bus import MessageBus
        from lunarsim.core.simulation_clock import SimulationClock
        from lunarsim.env.scenario_loader import load_scenario
        from lunarsim.slots.surface.measured_terrain import MeasuredTerrainSurface

        self.simulation_app = simulation_app
        self.width, self.height = width, height
        scenario = load_scenario(scenario_name, [override for override in scenario_overrides if override])
        self.preset = lander_preset_from_dictionary(scenario.lander)
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        self.stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        surface = MeasuredTerrainSurface(
            scenario.slots["surface"].parameters,
            BindingContext(number_of_environments=1, device="cpu", clock=SimulationClock(1, 0.005, 4, device="cpu"), bus=MessageBus()),
        )
        self.terrain = surface.build_terrain(scenario.root_seed)
        self.height_field = self.terrain.height_field()
        UsdGeom.Xform.Define(stage, "/World/ground")
        for name, (vertices, triangles) in self.terrain.meshes().items():
            mesh = UsdGeom.Mesh.Define(stage, f"/World/ground/{name}")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(numpy.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(numpy.full(len(triangles), 3, dtype=numpy.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.astype(numpy.int32).ravel()))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.29, 0.27)])  # regolith grey
        # Sun 15 degrees above the eastern horizon (Apollo 17 surface operations) and a faint sky
        # fill so that the shadows are not fully black. A distant light shines along its -z axis.
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(2500.0)
        sun.CreateAngleAttr(0.53)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
        UsdGeom.Xformable(sun).AddRotateYOp().Set(90.0 - 15.0)
        UsdLux.DomeLight.Define(stage, "/World/Fill").CreateIntensityAttr(25.0)

        lander_file = write_lander_usd(self.preset, Path(tempfile.gettempdir()) / "lunarsim_assets")
        self.landers, self.leg_translate_operations, self.arrows = [], [], []
        self.grain_points = {}  # lander index -> UsdGeom.Points, made by place_grains
        for index in range(number_of_landers):
            path = f"/World/Lander_{index}"
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(str(lander_file))
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, -1000.0))
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            self.landers.append(prim)
            self.leg_translate_operations.append(
                [UsdGeom.Xformable(stage.GetPrimAtPath(f"{path}/{leg_link_name(leg)}")).GetOrderedXformOps()[0] for leg in range(4)]
            )
            self.arrows.append([self._make_arrow(f"/World/Arrows_{index}/leg_{leg + 1}") for leg in range(4)])
        self.leg_bases = self.preset.leg_positions

        camera = UsdGeom.Camera.Define(stage, "/World/RenderCamera")
        camera.CreateFocalLengthAttr(28.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(36.0 * height / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 5000.0))
        self.camera_transform = UsdGeom.Xformable(camera).AddTransformOp()
        render_product = replicator.create.render_product("/World/RenderCamera", (width, height))
        self.annotator = replicator.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([render_product])
        for _ in range(30):  # let the stage load and the renderer start
            simulation_app.update()

    def _make_arrow(self, path: str):
        """Create one hidden arrow (shaft and head) and return its parts."""
        from pxr import Gf, UsdGeom

        root = UsdGeom.Xform.Define(self.stage, path)
        shaft = UsdGeom.Cylinder.Define(self.stage, f"{path}/shaft")
        shaft.CreateRadiusAttr(ARROW_RADIUS_M)
        shaft.CreateHeightAttr(1.0)
        shaft.CreateDisplayColorAttr([Gf.Vec3f(*TOUCHING_COLOR)])
        head = UsdGeom.Cone.Define(self.stage, f"{path}/head")
        head.CreateRadiusAttr(ARROW_HEAD_RADIUS_M)
        head.CreateHeightAttr(ARROW_HEAD_LENGTH_M)
        head.CreateDisplayColorAttr([Gf.Vec3f(*TOUCHING_COLOR)])
        root_operations = (UsdGeom.Xformable(root).AddTranslateOp(), UsdGeom.Xformable(root).AddOrientOp())
        shaft_operations = (UsdGeom.Xformable(shaft).AddTranslateOp(), UsdGeom.Xformable(shaft).AddScaleOp())
        head_operation = UsdGeom.Xformable(head).AddTranslateOp()
        UsdGeom.Imageable(root).MakeInvisible()
        return root, root_operations, shaft_operations, head_operation

    def place(self, index: int, row: numpy.ndarray, columns: dict[str, int]) -> None:
        """Put lander ``index`` on the recorded frame ``row`` and draw its contact arrows."""
        from pxr import Gf, UsdGeom

        translate_operation, orient_operation = UsdGeom.Xformable(self.landers[index]).GetOrderedXformOps()
        translate_operation.Set(Gf.Vec3d(*map(float, row[0:3])))
        orient_operation.Set(Gf.Quatd(float(row[3]), Gf.Vec3d(float(row[4]), float(row[5]), float(row[6]))))
        for leg, (x, y, z) in enumerate(self.leg_bases):
            stroke = float(row[columns[f"strut_{leg + 1}"]])
            self.leg_translate_operations[index][leg].Set(Gf.Vec3d(x, y, z + stroke))
        for leg in range(4):
            root, (root_translate, root_orient), (shaft_translate, shaft_scale), head_translate = self.arrows[index][leg]
            if f"leg_{leg + 1}_force_x" not in columns:
                UsdGeom.Imageable(root).MakeInvisible()
                continue
            force = numpy.array([row[columns[f"leg_{leg + 1}_force_{axis}"]] for axis in "xyz"], dtype=numpy.float64)
            magnitude = float(numpy.linalg.norm(force))
            if magnitude < ARROW_SHOWN_ABOVE_NEWTON:
                UsdGeom.Imageable(root).MakeInvisible()
                continue
            UsdGeom.Imageable(root).MakeVisible()
            leg_position = numpy.array([row[columns[f"leg_{leg + 1}_{axis}"]] for axis in "xyz"], dtype=numpy.float64)
            body_up = self._body_up(row)
            contact_point = leg_position - self.preset.footpad_depth * body_up  # lowest point of the dish
            # Shift the arrow outward, away from the vehicle axis, so the pad does not hide it.
            radial = leg_position[:2] - numpy.asarray(row[0:2], dtype=numpy.float64)
            radial = radial / max(float(numpy.linalg.norm(radial)), 1e-6)
            anchor = contact_point + ARROW_OUTWARD_SHIFT_M * numpy.array([radial[0], radial[1], 0.0])
            length = min(magnitude * ARROW_METRES_PER_NEWTON, ARROW_MAXIMUM_LENGTH_M)
            root_translate.Set(Gf.Vec3d(*anchor))
            root_orient.Set(rotation_from_z_to(force / magnitude))
            shaft_length = max(length - ARROW_HEAD_LENGTH_M, 0.05)
            shaft_translate.Set(Gf.Vec3d(0.0, 0.0, 0.5 * shaft_length))
            shaft_scale.Set(Gf.Vec3f(1.0, 1.0, shaft_length))
            head_translate.Set(Gf.Vec3d(0.0, 0.0, shaft_length + 0.5 * ARROW_HEAD_LENGTH_M))

    @staticmethod
    def _body_up(row: numpy.ndarray) -> numpy.ndarray:
        """Return the body +z axis in the world from the frame's quaternion (w, x, y, z)."""
        w, x, y, z = (float(value) for value in row[3:7])
        return numpy.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)])

    def place_grains(self, index: int, points: numpy.ndarray, colors: numpy.ndarray, widths_m: numpy.ndarray) -> None:
        """Draw the grain parcels of lander ``index`` as dots: world ``points`` (P, 3), RGB ``colors`` (P, 3), ``widths_m`` (P,).

        The dots are far larger than the grains (4-512 microns), so that the cloud can be seen
        (the caller sizes them by their distance to the camera); the point prim is made on first use.
        """
        from pxr import UsdGeom, Vt

        if index not in self.grain_points:
            prim = UsdGeom.Points.Define(self.stage, f"/World/Grains_{index}")
            prim.CreateWidthsAttr()
            prim.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
            prim.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
            self.grain_points[index] = prim
        prim = self.grain_points[index]
        if len(points) == 0:
            UsdGeom.Imageable(prim).MakeInvisible()
            return
        UsdGeom.Imageable(prim).MakeVisible()
        prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(numpy.float32)))
        prim.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(widths_m.astype(numpy.float32)))
        prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray.FromNumpy(colors.astype(numpy.float32)))

    def hide(self, index: int) -> None:
        """Move lander ``index`` far below the ground and hide its arrows and grains."""
        from pxr import Gf, UsdGeom

        UsdGeom.Xformable(self.landers[index]).GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0, 0.0, -1000.0))
        for root, *_ in self.arrows[index]:
            UsdGeom.Imageable(root).MakeInvisible()
        if index in self.grain_points:
            UsdGeom.Imageable(self.grain_points[index]).MakeInvisible()

    def look(self, eye, target) -> None:
        """Point the camera from ``eye`` at ``target``."""
        self.camera_transform.Set(look_at_matrix(eye, target))

    def capture(self, settle_updates: int = 1) -> numpy.ndarray:
        """Render and return the camera image (H, W, 3) as uint8; ``settle_updates`` extra frames first."""
        for _ in range(settle_updates - 1):
            self.simulation_app.update()
        for _ in range(60):
            self.simulation_app.update()
            pixels = numpy.asarray(self.annotator.get_data())
            if pixels.size > 0:
                return pixels[..., :3].astype(numpy.uint8)
        raise RuntimeError("the camera gave no image after 60 updates")


def draw_caption(image, lines: list[str], title_color=(255, 255, 255), size: int = 18):
    """Write ``lines`` on a black band at the top of a PIL image; the first line in ``title_color``."""
    from PIL import ImageDraw, ImageFont

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    line_height = size + 6
    draw.rectangle((0, 0, image.width, 12 + line_height * len(lines)), fill=(0, 0, 0))
    for number, line in enumerate(lines):
        draw.text((12, 6 + line_height * number), line, fill=title_color if number == 0 else (255, 255, 255), font=font)
    return image
