# NOTICE (grader-side; never staged into the agent's home)

The robot of this task is the OpenArm 2.0 by Enactic (https://github.com/enactic/openarm, documentation
https://docs.openarm.dev/hardware/openarm-2.0/). The pair's units (`tests/gt/` and the agent's `given/parts/`,
`given/collision/`) were cut from the project's assembly STEP (`OpenArm_2.0.STEP`, 2026-05-14, from the project's
hardware release; `enactic/openarm_hardware`, CERN-OHL-S-2.0) and the project's MuJoCo model (`enactic/openarm_mujoco`,
commit a3c8a57e, v2 bimanual model, Apache-2.0: the joints, ranges and the motors' friction, damping and armature).
Modifications: the STEP's solids grouped into the 18 assembly units of the kit (per arm: J1 link, J2 motor, J2 link,
upper arm with J3 / J4 motors and J4_D on its bearing, forearm with J5 motor, two J4 cable holders, wrist, gripper),
re-expressed in each unit's frame, meshed for drawing and convex-decomposed (CoACD) for the simulation, the units'
collision pieces carved around their mating partners (`-dev/kit/carve_kit.py`); no change of the design's geometry.
Under CERN-OHL-S-2.0 the modified design is published under the same licence with this source and modification note.

The Franka Panda is the MuJoCo Menagerie `franka_emika_panda` model (Apache-2.0, commit c96a32d). The agent cameras' RGB
is rendered by MuJoCo's Filament module (`arm/assembly_openarm/agent_render.py`, on the session of
`arm/assembly_full/agent_render.py`; the look and the room loader vendored from the chain's `render_filament/` and
`room/`) in a factory hall built from the FactoryWarehouse Interior Modular Pack 1.0.1 (CyberneticWalrus) and the wall
row of the Archviz Laboratory Pack 2.0 (Unity Asset Store; licensed for redistribution), lit by a Blender Cycles bake;
the bench top is Poly Haven's oak_veneer_01 (CC0). Depth, the photo pipeline and the depth-sensor model are the chain's
render layer `shared/render/` (MuJoCo backend). The picture manual (`given/manual/`) was rendered with Blender Cycles
from the same kit (`-dev/manual/build_manual.py`).

The sibling task robotic_arm_full_assembly_so101_panda (the six-servo arm) is the source of the cell's API, the grader's
structure and the information tier's observation side.
