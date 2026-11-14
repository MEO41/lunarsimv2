# SPDX-License-Identifier: BSD-3-Clause
"""``plant: rigid_box`` - one rigid box per environment. The milestone 0 stand-in for the lander."""

from lunarsim.core.binding import SlotName
from lunarsim.core.registry import register_binding
from lunarsim.slots.plant.physx_root_body_plant import PhysxRootBodyPlant


@register_binding(SlotName.PLANT, "rigid_box")
class RigidBoxPlant(PhysxRootBodyPlant):
    """A box with configurable size and mass.

    Parameters: ``size_m`` (3 values), ``mass_kg``.
    """

    def setup_scene(self, environment) -> None:
        """Spawn the box in environment 0; the environment clones it."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg

        size = tuple(float(value) for value in self.parameters.get("size_m", [1.0, 1.0, 1.0]))
        self.reference_height_above_ground_contact = 0.5 * size[2]
        box_configuration = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(self.parameters.get("mass_kg", 100.0))),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5 * size[2])),
        )
        self.asset = RigidObject(box_configuration)
        environment.scene.rigid_objects["box"] = self.asset
