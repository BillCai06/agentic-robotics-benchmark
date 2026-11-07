# NOTICE - source of the ground truth of this task (tests / -dev only; never shown to the agent)

The photographed unit `H_bcad` of this task (its ground truth in `reference/`, the oracle document in
`../solution/reference/`) is a **modified derivative** of the open-hardware **OpenArm 2.0** bimanual arm published by
**Enactic, Inc.**:

- hardware repository: https://github.com/enactic/openarm_hardware, commit `eefc9fa5509562e5e4d630db9a52027afdbc6af1`;
  the OpenArm 2.0 assembly STEP it links (`OpenArm_2.0.STEP`, sha256
  `0f97a5fb308d5b09353aa67bbd32d7c08e55dc5629bcbe20c12f71c82ef13c94`) and its bill of materials;
- licence: CERN Open Hardware Licence Version 2 - Strongly Reciprocal (CERN-OHL-S-2.0, the repository's `LICENSE.txt`).
  This notice is the attribution and the statement of modifications the licence asks for. The derived design (the
  reference parts, the bought-part stand-ins derived from source parts, the oracle document and the photographed unit)
  is distributed under the same licence; its design files are the files of this task (`reference/parts/*.brep`, the
  stand-in STEP files, `../solution/reference/unit.FCStd`).
- modifications (generator `chain/shared/arm/cad_openarm/` of the development workspace: family `openarm_pair_cad_v1`,
  generator `openarm-cad/1.0.0`, build `bimanual-cad-build/3`, the `build` of `reference/gt.json`):
  - the made parts of the torso and both arms were grouped into 29 parts (touching source parts fused into one solid
    each; covers, cable holders and fasteners left out), re-expressed in a base frame at the STEP's design pose;
  - band stretches of links (shoulder, shoulder offset, upper arm, forearm and wrist sections) by different amounts
    on the two arms, a spacer ring under one motor output, and lightening cut-outs (holes, slots) in seven arm parts
    and the base plate;
  - the bought parts (motors, gripper modules, column stand) replaced by stand-ins with fictional part numbers: the
    motors are this task's own parametric models built from the source's simplified motor bodies' dimensions; the
    gripper body and finger pieces and the column stand are the source pieces fused into one solid each, with small
    healing changes (filled nut traps and their holes, removed sliver faces, an enclosed void dropped, two corner
    fillets filled);
  - the unit was recoloured and photographed with the render layer (Blender Cycles backend).
  The task never names the source.
- kinematic reference: the joint ranges that serve as the controllers' stops in the instruction come from the OpenArm
  v2 MuJoCo model, https://github.com/enactic/openarm_mujoco commit `a3c8a57ec15d385e3c1dbd3fd090ee486b7f258c`
  (Apache License 2.0); the descriptions in https://github.com/enactic/openarm_description commit
  `1801c61247593fe2bf30cdbde36279aa559e04aa` (Apache License 2.0) were read for the link structure.
- the photo scenes use procedural, self-made assets (bench, cutting mat, checker targets, rulers, ID-1 card, paper tags;
  `shared/render/scale_refs.py`, `textures.py`) inside the lab room `room_lab_v4` of the project's licensed Unity asset
  pack (Archviz Laboratory Pack); the Cycles world light is the HDRI `schadowplatz_4k.hdr` of the project's licensed
  Unity asset pack. Neither is shipped with the task (only the rendered photos are).
- the public-model digest list (`given/public_blocklist.json`) holds sha256 digests of the OpenArm 2.0 STEP and of the
  mesh files of `openarm_description` and `openarm_mujoco`, plus vertex fingerprints of the source parts as grouped in
  the base frame (those identical to a part of the photographed unit left out); it contains no geometry.
- FreeCAD 1.1.4 (LGPL-2.0-or-later, https://github.com/FreeCAD/FreeCAD) is installed unmodified in the task image from its
  release AppImage, with the FreeCAD MCP server neka-nat/freecad-mcp @ `78bf023` (MIT, https://github.com/neka-nat/freecad-mcp)
  vendored unmodified (`environment/vendor/freecad-mcp/`).
