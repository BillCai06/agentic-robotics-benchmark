# NOTICE (grader-side; never staged into the agent's home)

The two-arm robot of this task is a hidden VARIANT of the open-source OpenArm 2.0 by Enactic
(https://github.com/enactic): the hardware design `enactic/openarm_hardware` (commit eefc9fa5; the OpenArm 2.0 full
assembly STEP of its documentation Drive, 2026-05-14), CERN-OHL-S-2.0, and the MuJoCo model `enactic/openarm_mujoco`
(commit a3c8a57e; the v2 bimanual model), Apache-2.0. The variant is made by this task's ground-truth generator
(`docs/plan/arm/chain/shared/arm/gt_openarm/` in the development workspace). Modification statement (CERN-OHL-S-2.0
section 3.3 / Apache-2.0 section 4(b)): the arm structure is the STEP assembly's machined solids, re-tessellated,
stretched at one cut plane per link (upper arm, forearm, fingers) with extension prisms, re-assembled on a shortened
column with chest adapter plates; the original motors are replaced by parametric cylindrical module housings drawn from
the fictional datasheets; one pocket holds a smaller housing on a machined spacer plate; one machined part per arm is
remade in stainless steel of the same shape; the programmed stops are moved inward; the structure inertials are
recomputed from the solids (densities per material class, calibrated per link on the upstream link masses). Only
images rendered from it (the photo set) and an inertial-only dynamics model (`gt/pair_dynamics.xml`, grader-side) leave
the development workspace; no mesh or STEP file is shipped. Date of the modifications: 2026-10-27 to 2026-10-28 (the
generator's variant family, then the draw of this task's pair). As CERN-OHL-S-2.0 requires, the modified material this
task conveys (the inertial-only model `gt/pair_dynamics.xml` with `gt/pair_model.json`, and the rendered photo set in
the benchmark's asset dataset) is distributed under the same licence; what it takes from `enactic/openarm_mujoco` (the
model's body layout and joint axes; Copyright 2026 Enactic, Inc.) stays under Apache-2.0.

The photos are path traced with NVIDIA OVRTX in the lab room `room_lab_v4`, built from the project's licensed Unity
Asset Store pack (Archviz Laboratory Pack), licensed for redistribution; the segmentation used for the evidence check
comes from MuJoCo (Apache-2.0). The scale references, the tray, the rack, the kitchen scale and the bench props are
generated from numbers by the chain's render layer (`shared/render/`) and this task's bench scene.

The joint modules, bus boards and power supplies, their vendors, part numbers and every datasheet, erratum, lot-report
and price number are fictional (modelled on public classes of BLDC joint modules); `gt/catalogue.json` is the
generator-side truth of the pages and the quote; no agent-visible code carries these numbers.

Per the owner's rule the givens, the instruction and the public card never name the robot, its makers, its motor
vendor or the renderer; this file and the -dev folder carry the attribution.
