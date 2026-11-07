# Customer brief: 7 replicas of the lab's bimanual pair (upgraded duty)

Clause ids in square brackets are the trace ids of your requirements sheet (`source_id`).

[B-01] Context. The customer orders 7 units that replicate the lab's reference pair (the two-arm unit on its column
shown in the survey photos PH-01..PH-41 and described in the build log BL): the same kinematics, the same machined
parts and the same programmed stops (two arms of seven revolute joints, each with a two-finger gripper, on one pedestal
column), fitted with NEW joint modules, new bus boards and a new power supply for a heavier, faster duty. The reference
pair itself has been shipped away and cannot be measured again.

[B-02] Precedence. This brief > the build log > the lot reports > the datasheets. An erratum sheet overrides the datasheet
it corrects. Lot data describe the lot they were measured on; the pilot modules come from the lot named in the quote.

[B-03] Units. Deliverables use SI units (m, rad, kg, s, N m, rad/s, A, V); the unit table of the schemas lists the accepted
equivalents (mm, deg, g, kgf.cm, rpm, ...).

[B-04] Working area. The pair carries the customer's tray by its two handle bars. The tray reference point (Appendix D)
stays inside the taped rectangle on the bench, i.e. inside the tape's INNER edges (PH-37).

[B-05] Heights and grasp. Tray reference heights run from the tray resting on the bench to the tray on the rack's top
shelf plus 15 mm (PH-37, PH-38). Both grippers hold the tray by its handle bars with the approach pitch
-45 deg (Appendix D: the gripper points forward and down) and the fingers closing across the bar (along y); the
elbow swivel angle is 54 deg on both arms (Appendix D: outward).

[B-06] Payload. The customer's loaded tray stands in PH-38. The top of the payload band is its mass, which the
customer read off the kitchen scale for photo PH-40 (read the display); the bottom of the band is 0.94 kg. Its
centre of mass lies midway between the handles and up to 79 mm below the handle grip points; each arm carries
half of it.

[B-07] Motion. Every move between two targets takes 1.16 s: the tray reference point travels on the straight line
between them (a trapezoid in path length with equal thirds: accelerate, cruise, decelerate) and the tray keeps its
orientation; the pair holds 0.9 s at every target; the duty is continuous (an endless sequence of one move and one
hold, no rest).

[B-08] Proof test. 1.25 x the top of the payload band is held at every target with the pair stationary.

[B-09] Grip. Pad friction coefficient 0.55, safety factor 2.0 against slip of a handle bar at the largest
vertical acceleration of the moves; the grippers hold the tray for the whole transport (size the gripper modules on their
RATED torque); a gripper opens its full stroke (closed to the open stop) within 0.7 s at no load. The grippers open
at least 10 mm wider than the handle bars (PH-38).

[B-10] Modules. From the vendor pages DS-* only. The module's housing must fit the replicated pocket at that joint (the
pockets are the reference pair's, see PH-31, PH-32, PH-33, PH-34, PH-35 and PH-36): one module per joint and per gripper, no gearing changes. A vendor
without an incoming-inspection report is counted at no more than 70 % of its typical (datasheet) peak torque,
rated torque, no-load speed and speed at peak torque, on the datasheet's straight-line envelope.

[B-11] Power. One bus for the pair at either 24 V or 48 V; one power supply (PSU) per pair; the harness voltage drop of
BL-03 applies at every module; no current limiting during any acceptance move.

[B-12] Cost. At most 4178 USD per pair for modules + bus boards + PSU at the 10-99 pcs list price (price_list.md).

[B-13] Engineering change. Any BOM line changed after the pilot acceptance costs 120 USD per line per pair (rework
and re-qualification); a module changed on both arms is two lines.

[B-14] Acceptance procedure (the customer runs it on a pilot pair; public, verbatim):
  a) Targets: the tray reference point at x in 6 equal steps over the taped rectangle's extent along x, y in 5 equal
     steps over its extent along y, and z in {lowest, midpoint, highest} of B-05: 90 targets in the grid order
     (x, y, z). The left TCP is at the tray reference point + (0, w/2, 0) and the right TCP at - (0, w/2, 0), with w the
     distance between the two handle grip points (PH-38); both TCPs at the approach pitch and finger direction of
     B-05.
  b) Joint angles by inverse kinematics, per arm: the wrist centre lies wrist_to_tcp behind the TCP along the gripper's
     pointing direction; J4 >= 0 from the shoulder-to-wrist distance; the elbow point on its circle about the
     shoulder-to-wrist line at the elbow swivel angle of B-05; J2 on its principal branch (|J2| <= 90 deg), J6 on its
     principal branch (|J6| <= 90 deg); an angle outside the stops is taken +-360 deg where that puts it inside the
     stops. Every target of both arms lies inside the stops.
  c) Moves: for each joint J1..J7, the 8 target pairs not yet selected with the largest joint-angle change of that joint
     (the larger of the two arms' changes; ties: the pair with the lowest indices in the grid order), then the corner
     pairs of the box x in {min, max} x y in {min, max} x z in {lowest, highest} (its 4 space diagonals, its 4
     edges along x and its 4 edges along y), each pair once, each run in both directions with the B-07 motion and the
     payload band top. Both arms move together. Joint angles at every 5 ms of a move by b); joint rates and accelerations
     from the path: qd = q'(s) ds/dt, qdd = q''(s) (ds/dt)^2 + q'(s) d2s/dt2, with s in [0, 1] the trapezoid's path
     fraction.
  d) Loads: per arm, rigid-body inverse dynamics of the arm (the machined structure + the BOM's modules as point masses
     with the inertia of their housing class at the pockets + half the payload as a point mass, rigid on the gripper,
     below the TCP by the CoM drop of B-06); tau_j = the load torque at joint j (it excludes the joint's own output
     inertia J_out of its datasheet, which the pass rule adds); qd_j, qdd_j from c).
  e) Capability of a SKU, for the weakest expected unit of the lot the pilot modules come from. The module sees
     V_s = V_PSU - (the harness drop of BL-03). Datasheet no-load speed and speed at peak torque at V_s on the straight
     line through the datasheet's two voltage points (interpolated between them, extrapolated outside them); peak and
     rated torque do not depend on the voltage. A
     unit's ratio to the datasheet (measured / datasheet) is f_peak x k_SKU x (1 + s z1) for the peak torque,
     f_speed x (1 + s z2) for both speeds and f_rated x k_SKU x (1 + s z3) for the rated (continuous) torque, z1..z3
     independent standard normal per unit. The lot factors f_peak, f_speed, f_rated, the spread s and the droop d belong
     to one lot; the SKU factor k_SKU belongs to the SKU's design and is the same in every lot of that vendor. The
     weakest expected unit is the lot's 2.5th percentile, z = -1.96 on every factor: tau_s = peak x f_peak x k_SKU x
     (1 - 1.96 s), w_k = speed_at_peak(V_s) x f_speed x (1 - 1.96 s), w0 = no_load(V_s) x f_speed x (1 - 1.96 s),
     tau_c = rated x f_rated x k_SKU x (1 - 1.96 s). Torque available at speed |w|: tau_s up to w_k; between w_k and w0
     the tau at which w = w_k + (w0 - w_k) (1 - tau / tau_s) (1 - d tau / tau_s) (d >= 0; d = 0 is the datasheet's
     straight line); none above w0. Bus current of a module: I = I_idle + (|tau_m w| + P_cu x (tau_m / tau_c)^2) / V_s,
     with tau_m = tau_j + J_out qdd_j, w = qd_j, P_cu the datasheet's copper loss at rated torque and tau_c the unit's
     continuous torque above, no credit for regeneration. A vendor without inspection data: B-10.
  f) Joint j of an arm passes iff at every move sample |tau_j| + J_out |qdd_j| <= the torque available at |qd_j|; every
     hold and every proof hold has |tau_j| <= tau_s; the duty RMS sqrt(s_move x max over moves of mean((tau_j +
     J_out qdd_j)^2) + s_hold x max over targets of tau_hold^2) <= tau_c, with s_move = T / (T + H) and s_hold = 1 - s_move;
     the module's voltage range contains the bus voltage; and its housing fits the pocket.
  g) A gripper passes iff its module's tau_c >= tau_grip = 2.0 x (m_top / 2) x (g + a_z,max) x r_J / 0.55 (a_z,max:
     the largest vertical TCP acceleration over the moves, r_J: the finger lever of Appendix D; the module drives both
     fingers through their 1:1 gear pair), its w0 >= the finger open angle / 0.7 s, its voltage range contains the bus
     voltage and its housing fits.
  h) PSU passes iff its voltage is the bus voltage, every module's voltage range contains it, and its rated current >= the
     peak over time of the summed bus current of all 16 modules of e) during any move (both arms at the same instant;
     the grippers at their holding current of g)).
  i) Bus / boards pass iff, for each arm and each protocol among that arm's modules, the BOM's boards provide one channel
     of that protocol with the bus voltage inside the board's input range (a channel serves at most the number of
     modules on its datasheet).
  j) Budget passes iff the BOM list price (modules + boards + PSU) <= the B-12 ceiling.
  The pair is accepted iff all 19 items pass (14 joint modules, 2 grippers, PSU, bus / boards, budget). A failed item is
  fixed by an engineering change (B-13).

[B-15] Conventions: Appendix D below (verbatim part of the brief).

[B-16] Deliverables: `requirements.yaml`, `bom.json`, `kinematic_sketch.json` (schemas in `schemas/`). The requirements
sheet reports, with `source_id: derived` and a `derivation` note, the derived items of the acceptance duty for YOUR BOM
installed on the replica, each as the value the acceptance procedure produces: for each arm a (left, right) and joint
j1..j7 `joint_peak_torque:<a>_<j>` (max |tau_j| over the move samples and the holds), `joint_rms_torque:<a>_<j>` (the duty
RMS of the LOAD torque tau_j, without J_out), `joint_peak_speed:<a>_<j>` (max |qd_j| over the moves); for each arm
`grip_torque:<a>` (tau_grip of B-14 g) and `moving_mass:<a>` (everything that moves with that arm's J1: the machined
structure plus your modules of j2..j7 and the gripper; no payload); and `supply_peak_current` (the peak summed current of
B-14 h with your BOM and the lot's weakest expected unit).

## Appendix D. Geometric definitions

Base frame: origin on the column's axis in the mounting plane (the base plate's bottom face, which is the bench top), z up,
+x along the arrow sticker on the bench (the pair's front), +y = z x x (towards the left arm).

Points of each arm: the shoulder centre (the common point of its J1, J2 and J3 axes); the elbow point (where its J4 axis
meets its J3 axis); the wrist centre (the common point of its J5, J6 and J7 axes); the fingertips (the closed fingers'
lowest point along the gripper axis, the line through the wrist centre along the gripper's pointing direction); the TCP
(on the gripper axis, 15.0 mm back from the fingertips towards the wrist; rigid on the gripper); a finger's pad centre
(the centre of its rubber pad's contact face). The handle grip point: the point on a handle bar's axis where a holding
TCP sits; the tray reference point: the midpoint of the two handle grip points.

Zero pose: each arm hangs straight down (shoulder centre, elbow point, wrist centre and TCP on one vertical line), its J4
and J6 axes parallel to y, its J7 axis parallel to x, its fingers closing along y. Joint angles: the right-hand rotation
about the joint's axis direction at the zero pose (the axis moves with the parent part):

| joint | left arm | right arm |
|---|---|---|
| J1 | +y | +y |
| J2 | -x | +x |
| J3 | -z | +z |
| J4 | -y | -y |
| J5 | -z | +z |
| J6 | -y | -y |
| J7 | +x | -x |

The right arm's directions are the mirror image of the left arm's in the x-z plane, so mirrored poses have equal angles.
The gripper angle (j8 in the sketch): the finger opening angle, 0 at the closed stop (fingertips touching), positive
opening; both fingers turn by it in opposite senses.

| quantity | definition |
|---|---|
| `shoulder_height` | height of the shoulder centres above the mounting plane (both arms are mounted level) |
| `shoulder_half_span` | distance of each shoulder centre from the x-z plane |
| `shoulder_forward_offset` | x of the shoulder centres |
| `upper_arm` | distance from the shoulder centre to the elbow point |
| `forearm` | distance from the elbow point to the wrist centre |
| `wrist_to_tcp` | distance from the wrist centre to the TCP |
| `finger_lever` | distance from a finger's hinge axis to its pad centre |
| `grip_stroke` | distance between the two pad centres at the open stop |
| approach pitch | elevation angle of the gripper's pointing direction above the horizontal (negative = pointing below it) |
| elbow swivel angle | rotation of the elbow point about the shoulder-to-wrist line, from the direction "down" (the projection of -z onto the plane normal to that line), positive outward (towards +y for the left arm, towards -y for the right arm) |
| stops | `[lower, upper]` of each joint in these conventions (the programmed limits of the reference pair, PH-13..PH-26); the gripper's `[0, open angle]` (PH-27, PH-28, PH-29 and PH-30) |
