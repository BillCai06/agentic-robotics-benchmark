# Shape sorter with a dexterous hand: put every piece through its own hole

A right-handed **Wuji Hand 2** on a floating wrist must post the pieces of a wooden **shape-sorter cube** through
the matching holes. The cube is clamped to a table; **11 pieces** lie on the table next to it. There are **four
scenes**, `inst_a`, `inst_b`, `inst_c` and `inst_d`, each with its own piece layout. For each scene you deliver one
**open-loop plan** (wrist poses and finger targets at 25 Hz). The grader runs your plan in the physics simulator from
that scene's exact start state and counts the pieces that passed **through their own hole** into the cube.

You know each scene's start state only from **RGB-D images** taken by two calibrated cameras. You do not get the
piece poses, nor the exact cube placement, of any scene. The simulator, the hand, the cube geometry, the controller and
the scoring code are exactly the ones the grader uses, and you have them (below), with two exceptions: **each graded
scene has its own hidden physical parameters** (friction, piece mass, finger actuator gain) and **its own hand-eye
calibration error** (the wrist ends up slightly offset from where you command it), both drawn within bounds given
below. The physical parameters are not given. The calibration error is not given either, but each scene comes with
images of a short hand-eye calibration routine from which you can measure it (see below). Your simulator uses the
nominal values and no calibration error, so a plan that works in your simulator may not work in the graded scene.

## What you have

- `~/sorter/`: the simulator and scoring package the grader uses (same files). It needs SuperDex 1.0.0, which is
  installed in the Python environment `/opt/venv` (use `/opt/venv/bin/python3`; run the commands below from your
  home directory). `sorter` selects double precision and finds the assets in `/opt/superdex_assets` and
  `/opt/sorter_data` by itself. Read `world.py`, `plan.py`, `score.py` and `config.py`.
  - `World(start_state)` builds the scene. `w.settle()`, `w.step_waypoint(wp)`, `w.hold()` and `w.result()` are
    exactly the grader's steps. `w.piece_poses()`, `w.wrist_pose()`, `w.hand_q()` and `w.hand_link_poses()` read
    the state. `w.capture()` / `w.restore(snapshot)` save and restore the full state bit-exactly.
  - `/opt/venv/bin/python3 -m sorter.check_plan plan.json` validates a plan without simulating.
  - `/opt/venv/bin/python3 -m sorter.run_plan start_state.json plan.json` runs a plan from a start state you write
    and prints the outcome.
- `~/instances/inst_X/`: the start images of each scene (`top` and `left`: `<cam>_rgb.png` and `<cam>_depth.png`),
  `cameras.json`, and `calib/`: the images of the scene's hand-eye calibration routine with `calibration.json`
  (section "Hand-eye calibration error of each scene, and the calibration images").
- `~/practice/practice_1..3/`: the same for three more scenes drawn the same way (cube placement, physical parameters
  and calibration error included). Their start states and calibration errors are not given, and nothing grades plans
  on them.
- `~/examples/example_plan.json`: a valid plan that only holds the start pose for 1 s.
- `/opt/superdex_assets/`: the public SuperDex asset files used here: `prefabs/shape_box/` (collision SDFs
  `collision/*.mochi.h5` with their surface meshes, textured render meshes `render/*.glb`, CAD-derived geometry) and
  `bots/hands/wuji_hand2_beta1/` (the hand: bot file, collision SDFs and render meshes `right/render/*.glb`).
  `/opt/sorter_data/` holds the cube's collision SDF re-baked at 0.5 mm from the same public mesh.
- Python packages in `/opt/venv`: numpy, scipy, trimesh, h5py, pillow, SuperDex. There is no network access.

## Scene, frames and units

- World frame: z up; the table top is the plane z = 0. Units are metres, radians, seconds, kilograms. The table is
  1.0 m × 0.8 m (centred at the origin) and there is nothing below it: a piece that leaves the table falls away and is
  lost. In the images the table is a 4 mm desk pad (its top is z = 0) lying on the end of a conference table in an
  office; the conference table and the office are scenery only, not part of the simulation.
- **Cube** (public SuperDex `shape_box` prefab): body + lid, both static. Nominally the body frame origin (bottom
  centre) is at world (0.10, 0, 0) with no rotation. In every scene the cube is clamped at a fixed but unknown
  offset from that nominal placement: dx, dy ∈ [−4, 4] mm and a yaw (about +z through the body origin) in
  [−2°, 2°], drawn uniformly and independently per scene (the lid moves with the body). The images show it. The cube is 0.14 m wide, with 10 mm walls and an 8 mm floor. The lid sits
  at z = 0.125–0.130, recessed between rim walls on its −x, +y and −y edges. The lid frame is offset from the body
  frame by (+3 mm, 0, 0.125025 m) (`config.LID_OFFSET`), so the lid and its holes sit 3 mm towards +x;
  `geometry.json` already includes this. The −x wall has no holes.
- **Pieces**: 40 mm long prisms (wood, nominal density 390 kg/m³), named by shape. The public star does not fit its hole,
  so it is not used. Hole of each piece (`~/sorter/geometry.json` gives every hole's plane and outline, the cavity box,
  and each piece's centroid and hull points; box-frame coordinates):

  | face | pieces |
  |---|---|
  | lid (top) | cross, triangle, square |
  | +x wall | parallelogram, octagon |
  | +y wall | rectangle, pentagon, hexagon |
  | −y wall | trapezoid, ellipse, diamond |

  The fits are tight. Measured on the collision SDFs the simulator uses, the smallest gap between a best-aligned
  piece and its hole while it passes through is 0.2–1.7 mm depending on the piece (parallelogram 0.20 mm, trapezoid
  0.26 mm, cross 0.39 mm, all others 0.6 mm or more), except the rectangle, whose best-aligned gap is about zero
  (−0.06 mm): it goes in only when pushed, through the small compliance of the contact model.
- **Start states** (all seven scenes are drawn the same way, from OS entropy): every piece rests on the table with its
  centroid uniformly in x ∈ [−0.42, −0.04], y ∈ [−0.33, 0.33]; centroids are at least 75 mm apart; yaw is uniform.
  With probability 1/2 a piece stands on one of its two end faces, otherwise it lies on a side face. Each layout was
  settled in the simulator. Resting pieces float slightly above the table: in the start states the lowest hull point of
  a piece is 0.6–0.8 mm above z = 0 (median 0.78 mm), because the contact model holds touching bodies a fraction of a
  millimetre apart. A start-state file is `{"pieces": {name: {"pos": [x, y, z], "quat": [x, y, z, w]}},
  "box": {"dx": .., "dy": .., "yaw": ..}}` (the SuperDex actor frame of each piece; the cube offset in metres and
  radians, see `sorter.world.box_pose`; `box` defaults to zero offset). An optional `"physics"` entry sets the
  scene's physical parameters, and an optional `"calibration"` entry its calibration error (next sections); without
  them the nominal values and no error are used.
- **Cameras** `top`, `left`: 640×480 pinhole cameras, the same rig in every scene; `cameras.json` gives
  `fx, fy, cx, cy` and `world_from_camera` (4×4, OpenCV convention: x right, y down, z forward). A camera-frame point
  projects to u = fx·X/Z + cx, v = fy·Y/Z + cy; pixel (column i, row j) samples (u, v) = (i + 0.5, j + 0.5).
  `<cam>_depth.png` is 16-bit depth along the camera z axis in units of 0.1 mm. Every surface in view returns a depth,
  the office's too; 0 (no return) only where nothing is seen within 6.5535 m, the 16-bit range. Both images follow
  this convention: the depth of pixel (i, j) is the depth of the surface point seen through (i + 0.5, j + 0.5)
  (rendered without multisampling), and RGB is antialiased (4 samples per pixel, averaged: rendered at twice the
  resolution, each pixel the mean of its 2 × 2 samples). The RGB images are rendered physically based (MuJoCo's
  Filament renderer) under the office's ceiling lights and daylight. Noise: RGB σ = 2 (8-bit units; values are then
  truncated to integers), depth σ = 2 mm + 0.2 % of the depth, independent per pixel.
  The images show the start state that the grader simulates, after its 0.5 s settle (rendered from a settle with
  the nominal physics; with a scene's hidden physics the settled piece positions differ by at most about 0.02 mm).
  The hand is not in the start images (it is in the calibration images, below).

## Hidden physics of each scene

Each of the seven scenes (the four graded ones and the three practice ones) has its own values of five physical
parameters. Each value was drawn once, independently and uniformly within the bounds below, from OS entropy. The
values are not given for any scene. The grader simulates each graded scene with that scene's values;
`World(start_state)` reads them from `start_state["physics"]` (see `sorter.world.physics_params`,
`config.PHYSICS_NOMINAL`, `config.PHYSICS_BOUNDS`), so you can simulate any values you like.

| key | meaning | nominal | bounds |
|---|---|---|---|
| `mu_piece` | Coulomb friction coefficient of every piece | 0.8 | [0.5, 1.1] |
| `mu_cube` | friction coefficient of the cube body and lid | 0.6 | [0.35, 0.85] |
| `finger_mu_scale` | multiplier on the friction coefficient of every hand link (asset values 0.5 and 1.0) | 1.0 | [0.6, 1.4] |
| `mass_scale` | multiplier on the piece density (mass and inertia) | 1.0 | [0.6, 1.6] |
| `finger_gain` | multiplier on the stiffness and damping of every finger joint's PD controller | 1.0 | [0.6, 1.4] |

Everything else is fixed and known: the table friction (0.6), the wrist controller, the geometry, the start poses
(not given to you, as above), the calibration error (next section; not given to you, but measurable from the
calibration images) and the scoring.

## Hand-eye calibration error of each scene, and the calibration images

As on a real robot, the hand is calibrated against the cameras (the world frame) only up to a small error. In each of
the seven scenes the wrist does not go exactly where a waypoint commands it: when a waypoint commands the wrist pose
(p, q), the controller drives the wrist towards **(p + dp, q_err ⊗ q)**. `dp` is a fixed translation (world axes) and
`q_err` a fixed small rotation about the wrist origin (world axes), the same for the whole scene. Each component of
`dp` was drawn uniformly in **[−1.5, 1.5] mm** and each component of the rotation vector of `q_err` uniformly in
**[−0.5°, 0.5°]**, once per scene, independently, from OS entropy (`config.CALIB_BOUNDS`). A rotation error moves the
fingertips too: 0.5° about the wrist shifts a point 0.15 m away by about 1.3 mm.

The values are not given. Instead, each scene comes with the images of a hand-eye calibration routine that the hand
executed in that scene **with that scene's calibration error**. The error is the same during this routine and during
the graded run of your plan. Estimating `dp` and `q_err` from these images is part of your job.

- Files: `~/instances/inst_X/calib/` and `~/practice/practice_N/calib/` hold `calibration.json` and, for each capture
  k = 0..3 and each camera, `<k>_<cam>_rgb.png` and `<k>_<cam>_depth.png`.
- The routine is the same in every scene (`calibration.json` is identical in all seven). Starting from the start pose
  after the 0.5 s settle, the hand executed the waypoints `calibration.json["routine"]["waypoints"]` (plan format,
  25 Hz) exactly as a plan is executed. It moves to four commanded wrist poses in turn
  (`captures[k]["commanded_wrist_pos"]`, `captures[k]["commanded_wrist_quat"]`) and holds each for 1.2 s; image pair k
  was taken at waypoint `captures[k]["waypoint_index"]`, the last waypoint of that hold. The fingers keep the asset's
  default posture (`sorter.plan.FINGER_DEFAULT`) throughout. At each capture the hand is at rest: the wrist is at
  (p + dp, q_err ⊗ q) for that capture's commanded pose (p, q), and the fingers are at their targets (within 1 µm and
  1e-4 rad in our tests). Nothing touches the hand, so its pose does not depend on the scene's hidden physics. Running
  these waypoints in `World(start_state)` with a `"calibration"` entry reproduces the routine in your simulator for any
  error you want to test (`w.hand_link_poses()` gives every link's pose).
- The images come from the same two cameras (`cameras.json`), with the same conventions (pixel centres, depth units,
  0 = no return) and the same noise model as the start images, with fresh noise. They show the office, the desk pad
  and the hand: the cube and the pieces are left out of these images. The hand is drawn with the public Wuji render
  meshes `/opt/superdex_assets/bots/hands/wuji_hand2_beta1/right/render/*.glb`: every link with a `renderModel` in
  the bot file (`wuji_hand2_beta1_right.superdex_bot`; its `links` are in the same order as `w.hand_link_poses()`) is
  drawn with that mesh at the link's pose. The meshes are glTF (y up), so a mesh vertex (x, y, z) is the link-frame point
  (x, −z, y). The render meshes differ slightly (up to a few millimetres) from the collision shapes that the simulator
  uses.
- The plan check (speed limits, workspace, wrist orientation and finger direction limits) applies to the poses you
  command. The simulator applies the error first and then everything in "Hand and controller" below (orientation
  limits, set-point clamp, wrist speed limit) to the resulting set-point.
- The hand starts at the start pose below; during the grader's 0.5 s settle it moves to the start pose plus the error.
- `World(start_state)` reads the error from `start_state["calibration"] = {"dp": [dx, dy, dz], "q": [qx, qy, qz, qw]}`
  (see `sorter.world.calibration_error`; without the entry there is no error). The finger joints have no calibration
  error.

## Hand and controller

- Hand: public SuperDex `wuji_hand2_beta1` (right), 20 finger joints. Its root link `link_00_mount` (the "wrist") is
  free-floating and gravity-compensated. An implicit PD controller in the simulator drives the wrist position
  (5000 N/m, 100 N·s/m), the wrist orientation (500 N·m/rad, 10 N·m·s/rad) and every finger joint
  (30 N·m/rad, 1.2 N·m·s/rad nominal, scaled by the scene's `finger_gain`) toward their set-points.
- Joint ranges: positive flexion bends a finger toward the palm. The flexion joints (thumb mcp and ip; mcp_flex, pip
  and dip of the other four fingers) range from 0 (straight) to their full flexion; the fingers cannot bend backward
  (no hyperextension). This is stricter than the asset file, which allows about −1.05 rad. The abduction joints and
  the thumb's cmc joints keep the asset's ranges. The simulator enforces these ranges with stiff joint-limit springs
  (under contact a joint can yield a few degrees past its limit, at most about 4° in our tests), and the plan check
  below rejects any target outside them. All ranges are in `sorter.plan.FINGER_LO` / `FINGER_HI`.
- Start pose: wrist at (−0.05, 0, 0.30) with orientation quaternion (−0.085497, 0.648759, −0.07424, 0.752523)
  (palm down, fingers toward +x); fingers at the asset's default pose (`sorter.plan.FINGER_DEFAULT`).
- Wrist orientation limit: the hand stands in for a hand on a robot arm at a table, so it cannot turn palm-up. The
  palm normal (the unit vector out of the palm; at the start pose it points straight down, (0, 0, −1)) must stay
  within **105°** of straight down: 0° is palm down, 90° is palm sideways, and the palm may face at most 15° above the
  horizontal (`config.WRIST_MAX_PALM_DEG`). `sorter.plan.palm_angle_deg(quat)` gives the palm angle of a wrist
  orientation. The plan check rejects any waypoint beyond the limit (the error names the waypoint and its angle), and
  the simulator also projects every wrist orientation set-point, including the ones interpolated between waypoints,
  back onto the limit (`sorter.plan.wrist_limit_project`).
- Finger direction limit: for the same reason the hand cannot point its fingers up with the wrist underneath. The
  finger axis (the unit vector from the wrist towards the fingertips; at the start pose it is horizontal, pointing
  towards +x) may point at most **35°** above the horizontal (`config.FINGER_MAX_PITCH_DEG`); pointing down is not
  limited. `sorter.plan.finger_pitch_deg(quat)` gives this finger pitch for a wrist orientation. The plan check
  rejects any waypoint beyond the limit (the error names the waypoint and its pitch), and the simulator projects every
  wrist orientation set-point, including the interpolated ones and after the calibration error, back inside both
  orientation limits (`sorter.plan.orientation_limit_project`; it does nothing inside them).
- Each waypoint lasts 0.04 s (20 physics steps). Over those steps the set-points move linearly from the previous
  waypoint to the new one (orientation by normalised quaternion interpolation). The wrist set-point is clamped to stay
  within 25 mm and 0.4 rad of the wrist's actual pose, so a blocked hand pushes with a bounded force.
- Wrist speed limit (the hand stands in for an arm-mounted hand, which cannot be flung): in every 2 ms physics step the
  internal wrist set-point moves at most 0.6 m/s × 2 ms in position and 4.8 rad/s × 2 ms in orientation away from its
  previous value (`config.SP_MAX_LIN_SPEED`, `SP_MAX_ANG_SPEED`; 1.2 × the plan limits, so a plan that keeps its
  limits is never slowed down). In addition, the clamp above shrinks as the wrist moves towards the set-point, like a
  motor's force–speed curve: the allowed lead is 25 mm × max(0, 1 − v / 0.8 m/s), and for the orientation
  sin(lead angle / 2) ≤ sin(0.2) × max(0, 1 − ω / 6.4 rad/s), where v and ω are the wrist's actual linear and angular
  speed towards the set-point over the last physics step (`config.WRIST_V0_LIN`, `WRIST_V0_ANG`; see
  `World._set_hand_target`). A blocked hand still pushes with the full bounded force, but when it comes free it does
  not jump to its target: it moves on at about the plan speeds (in our tests the actual wrist speed stayed below about
  0.8 m/s and the rotation speed below about 4.9 rad/s) and catches up with the plan. Free motion within the plan
  limits is not affected. Because v and ω are measured over a single physics step, the wrist can chatter while this
  clamp is active (blocked, pushing or catching up): its speed may alternate between two values from one step to the
  next, a deterministic wiggle of about 0.1 mm at 250 Hz. It is part of the simulator the grader uses.

## Plan format (the deliverable)

Write `~/submission/plans/<scene>.json` for each scene, for example `~/submission/plans/inst_a.json`:

```json
{"format": "shape_sorter_plan/v1",
 "waypoints": [[x, y, z, qx, qy, qz, qw, f0, f1, ..., f19], ...]}
```

- One row per 0.04 s, at most 4500 rows (180 s). Each row = wrist target position (world frame), wrist target
  orientation (unit quaternion xyzw) and the 20 finger joint targets in SuperDex DOF order (thumb cmc_flex, cmc_abd,
  mcp, ip; then index, middle, ring, pinky with mcp_flex, mcp_abd, pip, dip each).
- Limits, checked between consecutive rows (the first row against the start pose): wrist speed ≤ 0.5 m/s, wrist
  rotation ≤ 4 rad/s, each finger ≤ 6 rad/s; wrist position inside x ∈ [−0.5, 0.5], y ∈ [−0.4, 0.4],
  z ∈ [0, 0.6]; finger targets inside the joint ranges above (`sorter.plan.FINGER_LO` / `FINGER_HI`); palm angle of
  every row ≤ 105° (wrist orientation limit above); finger pitch of every row ≤ 35° (finger direction limit above);
  quaternion norm within 1e-3 of 1; finite numbers; file ≤ 8 MB. A plan that breaks any rule is invalid, and that
  scene scores 0.
- After the last row the controller holds the last targets for 2 s, then the scene is scored.

## Scoring

The grader rebuilds each scene from its exact start state, settles 0.5 s, executes your plan, holds 2 s and applies
`sorter/score.py`. Throughout the run it tracks each piece's centroid at every physics step:

- a piece **enters** through a hole when its centroid crosses the face's mid-wall plane inward inside that hole's
  outline (3 mm tolerance); a crossing anywhere else is an entry "through the wall" (the simulator does not allow it,
  but it would not count); moving back out clears the record;
- at the end, a piece counts as **passed** if its last entry was through **its own hole** and all of its hull points
  are inside the cube's cavity (1 mm tolerance). A piece pushed in through a different hole does not count and stays
  in the cube.

Scene score = passed pieces / 11. The reported raw metric is the mean of the four scene scores; the task reward is
that value relative to a reference solution. If the simulation of a scene ever produces a non-finite state, the run
stops and that scene scores 0.

## Simulator facts that matter for planning

- SuperDex 1.0.0, double precision, single-threaded (`initialize(0)`), Backward Euler, 2 ms steps, 20 Newton
  iterations. A run is deterministic: the same start state and plan give a bit-identical result, in any process
  and on any machine. The image pins the numerics for that: `OPENBLAS_CORETYPE=Haswell` and
  `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` are set in its environment, because without them
  OpenBLAS picks its kernels by CPU and the same plan can pass different pieces on different machines. Keep these
  two settings when you simulate, or your local results may differ from the grader's. `python3 -m sorter.numerics`,
  run from your home directory, compares the machine with the reference. The grader runs the same check first and
  refuses to grade where it fails. Different start states diverge: a sub-millimetre change in where a piece starts
  can change the outcome. Even a last-bit change of a waypoint number can: the grader feeds your numbers exactly as
  written in the JSON (as Python floats) to `World.step_waypoint`, which normalises the quaternion itself, in a fresh
  process with one World.
- Speed on one core: about 1 s of wall time per simulated second when nothing touches anything, and about 4–10 s per
  simulated second while the fingers hold or push a piece. A full 180 s plan takes roughly 10–25 minutes on one core.
  The grader runs the four scenes in parallel on four cores.
- Contact: pieces, cube and lid use the prefab's contact settings with nominal friction 0.8 (pieces) and 0.6 (cube,
  table; see the hidden physics above),
  friction regularisation velocity 1e-3 m/s; the hand uses its asset contact settings except a friction
  regularisation velocity of 1e-4 m/s (`config.HAND_FRICTION_FALLOFF`), so a held piece does not creep in the grasp.

## Budget

You have **2 hours** of wall time, **4 CPU cores** and 16 GB of memory. Plans are only read at the end, from
`~/submission/plans/`.
