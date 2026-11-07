# The assembly cell: the customer's room, two Unitree G1 humanoids, the rules

This is the integrator's description of the robot cell your program drives. Everything stated here as exact is exact;
everything else (where each part is, which way it faces, the robots' start poses, sensor noise) must be measured in
each episode. The cell never reports part poses, part identities, joint states of the furniture or any fastening.

## 1. The room (exact survey)

Room frame: origin at the south-west floor corner, **x east, y north, z up**, metres, radians, quaternions `w x y z`.

| item | geometry |
|---|---|
| floor | z = 0, oak planks |
| walls | inner faces at x = 0, x = 6.0, y = 0, y = 5.0; 2.6 m high, plaster, the ceiling at z = 2.6; door opening in the south wall for 4.80 <= x <= 5.70, 2.05 m high, onto a hallway; window in the east wall for 1.60 <= y <= 3.00, 0.90 <= z <= 2.10: a white frame and clear glass set into the wall (6.03 <= x <= 6.10), the street outside |
| drawn only (no obstacles: nothing collides with them or rests on them) | white skirting along the walls, 8 cm high and 1.2 cm deep; a window board under the window, 5.94 <= x <= 6.0, 1.55 <= y <= 3.05, 0.875 <= z <= 0.90, with a small potted plant on it at about (5.98, 2.75) that reaches to x = 5.84 and z = 1.21; framed pictures on the north wall (two, 0.73 x 0.57 m, centred at x = 1.25 and x = 2.15, z = 1.25), on the west wall (centred at y = 4.25, z = 1.05) and on the south wall (centred at x = 2.55, z = 1.10), and a wall clock on the north wall (centred at x = 4.95, z = 1.75), all within 5 cm of their walls; two ceiling light panels, 1 m square, centred at (1.8, 2.0) and (4.0, 2.2) |
| base cabinet (the customer's, fixed) | a box, centre of its footprint (3.40, 3.30), size 0.80 (x) x 0.36 (y) x 0.45 (z): it spans 3.00 <= x <= 3.80, 3.12 <= y <= 3.48, 0 <= z <= 0.45 |
| panel cart (fixed) | three grooves running east-west, centred at y = 1.15, 2.15 and 3.15, each on its own bed block 0.90 (x) x 0.16 (y) x 0.45 (z) centred at x = 1.05; each groove is two rails 0.90 x 0.03 x 0.05 m on the bed, the gap between them = the board's thickness + 4 mm |
| kitting table (fixed) | top centre (4.20, 1.60), 2.50 (x) x 0.50 (y), top surface at z = 0.72; two battens along x on the top, 0.04 m wide and 0.05 m high, centred at y = 1.565 and y = 1.635 |
| couch (the customer's, fixed) | a leather two-seater against the east wall, seat towards the west, within 5.09 <= x <= 5.95, 3.36 <= y <= 4.94 (the drawn couch to within 2 mm); its collision envelope is four boxes on four legs: seat 5.09 <= x <= 5.75, 3.49 <= y <= 4.81, 0.16 <= z <= 0.48; back 5.75 <= x <= 5.95, 3.49 <= y <= 4.81, 0.16 <= z <= 0.94; arms 5.09 <= x <= 5.95, 0.16 <= z <= 0.72, one for 3.36 <= y <= 3.49, one for 4.81 <= y <= 4.94; legs 0.035 x 0.035 x 0.16 m centred at (5.12, 3.39), (5.12, 4.91), (5.93, 3.39) and (5.93, 4.91) |
| plant (the customer's, fixed) | a potted plant with broad leaves in the north-west corner; its collision envelope is two upright cylinders: the pot, radius 0.13, axis at (0.51, 4.51), 0 <= z <= 0.25; the foliage, radius 0.40, axis at (0.45, 4.55), 0.25 <= z <= 1.62 (the drawn leaves reach up to 8 cm beyond it, the pot's rim 2 cm) |
| rug (the customer's) | a grey rug with rounded corners, flat on the floor in front of the couch, 1.27 (x) x 1.52 (y) x 0.01 m centred at (4.85, 4.15): 4.215 <= x <= 5.485, 3.39 <= y <= 4.91; drawn only, it is no obstacle and supports nothing (whatever is put on it rests on the floor) |

The drawings' floor plan shows the bookcase's place.

## 2. The flat-pack as delivered

The bookcase's parts are in the room, unpacked, and apart from the fixed items of section 1 nothing else is: the
**tall boards** stand upright in the panel cart's grooves, one per groove, each on its bottom edge, its long horizontal
side along the groove; which board stands in which groove and which of its faces looks north vary per episode, and a
board may lean slightly in its groove.
The **other boards** lie flat on the kitting table's battens, spread along the table, a front edge towards the north or
the south, positions and headings varying by a few centimetres / degrees per episode.

Materials: particleboard (18 mm boards) 650 kg/m^3 with white melamine faces, raw (light brown) edges except the
white edge banding the drawings mark; hardboard (6 mm) 900 kg/m^3, one white face, one raw (brown) face. Holes show as
dark dots, pre-installed dowels as wooden pegs.

## 3. The robots

Two Unitree G1 humanoids (29 DoF with Dex3-1 hands), named **`a`** and **`b`**, each starting near the south wall
(roughly x 1.9 and 3.2, y 0.55, facing north; the exact start varies per episode). The model is MuJoCo Menagerie's
`unitree_g1/g1_with_hands.xml` (copy in `~/given/robot/unitree_g1/`, BSD-3-Clause) with the cell's modifications:

* **Locomotion controller.** The pelvis rides four base joints: `base_x`, `base_y` (room x / y), `base_yaw` and
  `base_z`, driven by force-limited servos (x / y 300 N, yaw 150 N m) that the robot's own locomotion controller
  commands; the legs are posed from the stand height and do not touch the floor (no balance falls are simulated). You
  command the base by velocity or by a goal pose (section 5). A robot obstructed by furniture stops; its controller
  does not wind up beyond 5 cm / 0.1 rad.
* **Legs in the camera images.** The cameras draw each robot's legs stepping while its base travels or turns: a
  walking animation that follows the base (the feet are set down and stay put while the pelvis passes over them,
  rolling heel to toe; a foot the leg cannot reach flat rolls over its heel or toe). Small base moves (a few
  centimetres, a turn of up to about 20 degrees) are taken up by the planted legs without a step, and a robot that
  stops stands on the feet where they last stepped, on the floor (or on the rug). The animation is drawn only, in the
  colour and the depth images alike; nothing above the hips changes in them (the pelvis, the torso, the arms, the head
  and the head camera are where the physics has them). The physics keeps the posed legs: their colliders stay as posed
  and the drawn legs touch nothing (while a robot walks, a drawn foot can be up to about 35 cm from its collider), and
  nothing the program reads changes.
* **Stand height** (pelvis height above the floor) 0.48 - 0.80 m, nominal 0.793 m; lower means a squat (the knees go
  forward).
* Every robot body is gravity-compensated; payloads are not: a held board loads the arms, the waist and the base.
* Joints: waist yaw / roll / pitch, 7 per arm (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw), 7 per hand;
  ranges as in the model (`describe()` lists them). Torque limits as in the model (shoulders / elbow / wrist roll
  25 N m, wrist pitch / yaw 5 N m, waist roll / pitch 50 N m).
* **Collision:** body links are capsules (Menagerie's `g1_mjlab.xml` colliders); robots collide with the furniture, the
  parts and each other, never with themselves or the floor. **The hands (the `wrist_yaw_link` and the finger links) carry
  no contact**: grasping is the cell's grip rule (section 4), not finger contact.
* **Grasp site** of each hand, `<side>_grasp`: in the `<side>_wrist_yaw_link` frame at (0.150, -0.046, 0) for the left
  hand and (0.150, +0.046, 0) for the right hand, with the link's axes: **x = approach** (along the fingers), **y = pinch
  axis**, z = x cross y. The left hand's fingers lie on its +y side, the right hand's on its -y side; a board edge held in
  a grip sits at the grasp site with the board's thickness along the pinch axis.
* **Head camera** `head`: an RGB-D camera on `torso_link` at (0.0576, 0.0175, 0.4299) with the optical axis along the
  link's +x pitched down by 0.8308 rad (the D435 mount of the Unitree URDF); it moves with the waist.

## 4. Grip rule (hands)

Closing a hand (`"close"`) engages a grip on a board when, at that instant, the hand's grasp site straddles an edge of
that board: the board's thickness direction within 35 deg of the pinch axis, the grasp site within the board's slab
(at most 15 mm outside its faces), and the site inside the board's outline at most 45 mm from one of its edges (and
not within 5 mm of a corner along that edge). The board is then held rigidly at the relative pose of that instant
until the hand opens or the grip slips. A closed hand that straddled nothing holds nothing (open it and close it
again to retry). Grip limits: a board held by **one** hand slips when the hand's load exceeds **30 N** or its torque
about the grasp site exceeds **5 N m**; a board held by **k >= 2** hands (of one or both robots) slips from all of them
when its net load per hand (the weight plus whatever pushes on it, shared by k) exceeds **30 N**; an excess must last
50 ms. A slipped board falls. The hand's tactile reading (`get_state()`: `tactile_N`) is its share of the held board's
load (0 when it holds nothing).

## 5. Control (each 10 ms tick)

* Base: `base_velocity` [vx, vy, wz] in the robot's heading frame (x forward, y left; limits 0.5, 0.3 m/s, 0.8 rad/s;
  accelerations 0.8, 0.8 m/s^2, 1.5 rad/s^2), held until replaced; or `base_goal` [x, y, yaw] in the room: a
  straight-line approach (proportional, gain 1.2 /s, within the same limits, no obstacle avoidance) that stops within
  5 mm / 0.01 rad. `stand_height` moves at 0.15 m/s.
* Waist: joint targets [yaw, roll, pitch], at most 0.8 rad/s.
* Arms: either joint targets (`q`, 7 values) or a grasp-site pose target (`pose` [x, y, z, qw, qx, qy, qz]) in the
  robot's **base frame** (origin on the floor under the pelvis, x along the heading, z up; the default) or in the
  **room frame**. A pose target is solved by the cell's damped-least-squares IK over the arm's 7 joints (base, waist and
  legs as they are), with a seeded global solve when the target is new. `path: "line"` (default) moves the grasp site
  on a straight line with a minimum-jerk timing (about 0.25 m/s and 1 rad/s on average, peak accelerations at most
  1 m/s^2 and 4 rad/s^2; the two arms of one `command` call share one timing, so a board held by both is carried
  consistently); `path: "joint"` moves in joint space to the target's
  IK solution. Joint targets move at most 1.5 rad/s with 6 rad/s^2. An unreachable target leaves the arm where the IK
  gets closest (`get_state()`: `ik_residual` [position m, rotation rad]).
* Hands: `"open"` or `"close"` (the Dex3-1 open / pinch synergies; closing applies the grip rule once).

## 6. Sensors

* `get_state()`: per robot the localised base pose [x, y, yaw] (LiDAR localisation against the surveyed room:
  independent noise of about 3 mm and 0.2 deg per reading), the stand height, whether the base is moving, all waist,
  arm and hand joint angles (exact encoders), per hand `closed`, `tactile_N`, the grasp site's pose in the robot's base
  frame (exact kinematics) and the IK residual of a pose target.
* Cameras (`get_camera`): `ceiling_west` and `ceiling_east`, fixed RGB-D cameras 960 x 720 (vertical field of view 72
  deg) whose exact pose and intrinsics `describe()` gives; `a/head` and `b/head`, 640 x 480 (vertical field of view 58
  deg), whose pose in the room at capture time comes with each frame (composed from the localised base pose and the
  exact kinematics). Pinhole, no distortion: fx = fy = (H / 2) / tan(fovy / 2), principal point at the image centre.
  Depth is along the optical axis, registered to the colour pixels, 1 mm quantisation, noise sigma 1.5 mm +
  2.5 mm/m^2 z^2, invalid (0) below 0.28 m (head) / 0.40 m (ceiling), beyond 6 m, on depth discontinuities and on 2 %
  of the pixels at random. Colour is rendered physically based (MuJoCo's Filament renderer) in the room as section 1
  describes it, the same light in every episode: the morning sun shines in through the east window (sharp shadows,
  bright patches on the kitting table and the floor west of the window, which throw a warm light back up into the
  room), the two ceiling panels light the room from above (soft shadows) and the light the room's surfaces reflect
  fills in. Colour carries a per-episode white balance / brightness draw and pixel noise. Colour and depth draw the
  robots' legs as section 3 says (stepping while a base moves).
  **At most 900 frames per episode** over all cameras; rendering is the cell's time, not your program's (section 9).

## 7. How the bookcase holds together (the cell's fastening rules; never reported)

* **Dowel joint.** When every dowel of one end of a board lies within 5 mm of the axis of a distinct hole of a group of
  holes on another board, with axes within 4 deg, and the dowel tips between 3 mm outside the holes and fully in, the
  joint **engages**: the holes pull the board onto the seated pose they define (dowels fully in, faces in contact), the
  way chamfered dowels and a cam fitting would. A joint engaged and seated within 1.5 mm and 1.5 deg, at rest (relative
  speed below 5 mm/s) for 0.3 s, **locks** rigidly. An engaged board pulled 12 mm / 8 deg away is released again. Any
  geometrically matching dowel-hole combination engages, including a wrong one.
* **Back panel.** A back panel lying in a side panel's rabbet within 4 mm across, 8 mm up or down, 3 mm in depth and 3
  deg engages the same way and locks the same way.
* A board may still be held while it engages and locks. A locked board stays where it is when released.

## 8. Safety (a violation makes the episode score 0; the episode goes on)

* a robot link other than the hands pressing on the furniture, the walls or a part with more than 150 N for 0.2 s;
* the two robots touching each other with more than 80 N;
* a part moving faster than 3 m/s (a dropped or thrown board);
* a part pressed on another part or the room with more than 400 N for 0.2 s.

## 9. Budgets per episode

1500 s of simulated time; 3000 s of program time (the episode's wall clock minus the cell's own time inside `step`,
simulating, and inside `get_camera`, rendering: your program's computation and all its other requests count); 120 s
from a reply to the next request (silence) ends the episode; 900 camera frames; at most 500 ticks per `step` call.
