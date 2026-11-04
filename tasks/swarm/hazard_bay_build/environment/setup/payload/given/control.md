# The bay mission: your fleet's program

The second half of the submission is the program that runs the fleet through the plant's whole job (`bay_brief.md`):
close the leaking line's valve, put the fire out with the bucket, clear the assembly station into the crate. You write
the control for every robot and the planning that sequences them; how you do it is yours. After the scene checks, the
grader runs your program in **the real bay**, twice, with different seeds: its own stage of the bay as it really is,
with your four robots in it exactly as your scene builds them (The real bay, below). It measures only what happens:
whether the steam is released, the fire out, the parts in the crate, how long it took, how steadily the robots moved,
what was spilt, lost or hit, and how alike the two runs were.

Your program runs as a separate process, launched as `bash run.sh` in your submission folder (`/opt/venv/bin/python3`
with NumPy and `pxr` is there; `run.sh` may start anything that speaks the wire; your submission folder, scene
included, is its working directory), and talks over its stdin / stdout: one JSON object per line.

## The mission

At the start the flying robots stand on their pads, the ropes rigged (slack, lying on the floor), the bucket full beside
the tank, the fire burning, the steam leaking, the vehicle in its bay, the six parts somewhere on the two tables. The
jobs:

- **The steam**: turn the valve's lever, from wherever it was left, to its closed stop. The line's pressure then falls
  to nothing over 3 s and the leak stops (the real valve's behaviour). The vehicle's arm works the lever by pushing or
  pulling it; nothing turns it for you. The valve's packing holds the lever wherever it is: its own weight does not
  turn it, a push does.
- **The fire**: lift the bucket on its three ropes, carry it to the pan, tip it over the pan. Once 1 litre has been
  poured on the pan the fire is out.
- **The station**: every part off the tables and into the crate.

Time limit 600 s of simulation. The mission ends when all three jobs are done and everything is still (the flying
robots slower than 0.3 m/s, the bucket hanging or standing within 8° of upright, the vehicle slower than 0.1 m/s) for
3 s, or at the time limit, or when every robot is lost.

**The bay's rules, as the grader keeps them**: a flying robot inside the steam plume while the pressure is up for 1 s
is lost; the vehicle within 2 m (horizontally) of the pan's centre while the fire burns for 1 s is lost. A lost robot is
out for the rest of the run: its thrust is cut, its wheels and arm stop. Your scene's hazard logic does not come into
the real bay (the scene checks test it).

## The wire

```
READY                                       <- your first line, within 40 s of launch
{"type": "reset"}                           -> {"ok": true}
{"type": "act", "t": 12, "obs": {...}}      -> {"action": {"thrust": {"UAV_1": [fx, fy, fz], ...}, "ugv": {...}}}
{"type": "close"}                           -> {"ok": true}, then exit
```

- Control steps come at 60 Hz of simulation time (two physics steps of 1/120 s each). `t` counts them from 0;
  `obs.time` is the simulation clock.
- Every reply within 10 s of wall time. A missed deadline, a crash, a malformed reply or a reply without an action
  aborts the run: every check of that run scores 0. (The grader restarts `run.sh` at most once in a run; your process
  must be able to start twice, and it is started afresh for the second run, after everything the first one left
  running has been stopped.)
- A run also has a wall-clock budget: one that has not ended 30 minutes after its first observation is stopped and
  counts as aborted. The simulation itself takes about 20 ms of wall time per control step, so a run that lasts its full
  600 s (36,000 steps) leaves your program about 20 ms per reply on average.
- **A step may be asked more than once with the same `t`.** Repeated queries do not advance the mission; answer each
  from the observation it carries, and count steps by `t`, not by requests.
- Nothing you print besides the wire's replies reaches the grader; stderr is discarded.

## What you command

```
{"thrust": {"UAV_1": [fx, fy, fz], "UAV_2": [...], "UAV_3": [...]},
 "ugv": {"vel": {"<joint>": rad_per_s, ...}, "pos": {"<joint>": rad, ...}, "grip": true | false}}
```

- `thrust`: one force vector per flying robot in newtons, world axes, applied at the airframe. The grader clamps it to
  the airframe's `thrust max` (catalog), to F_z ≥ 0, and to a 40° tilt of the thrust vector from vertical (the
  attitude limit). The flying robots' attitude is held level by a perfect inner loop (their heading stays as on the
  pad); their position is pure dynamics: your thrust, gravity, the rope, the wind. A robot without a thrust entry
  gets zero.
- `ugv.vel`, `ugv.pos`: velocity and position targets for the vehicle's joints, by the joint names of your scene (the
  `mission` lists them), applied to the joints' drives as your scene authored them (a drive with stiffness follows a
  position target, one with damping only a velocity target). Velocity targets are clamped to ±40 rad/s, position
  targets to the joint's limits. A target keeps its last value until you send another.
- `ugv.grip`: the vehicle's suction tool at the origin of the arm's last link (the last rigid body, in the vehicle's
  link order, under the prim that declares `arb:arm_model`; the `mission` names it). While it is on, it takes the first
  part whose bounding box comes within 4 cm of that point and holds it rigidly; switched off, it lets go. It holds
  parts only (the lever is worked by contact).

## What you observe (per control step)

```
{"t": 12, "time": 0.2, "mission": {...},
 "uav": {"UAV_1": {"pos": [x, y, z], "vel": [vx, vy, vz], "load": [fx, fy, fz],
                   "neighbors": [[dx, dy, dz], ...], "bucket": [dx, dy, dz] | null, "bucket_tilt": deg | null,
                   "pan": [dx, dy, dz] | null, "fire": true | false | null, "disabled": false}, "UAV_2": {...}, "UAV_3": {...}},
 "ugv": {"pose": [x, y, heading_deg], "vel": [vx, vy, vz], "yaw_rate": deg_per_s,
         "joints": {"<joint>": [position, velocity], ...}, "grip_point": [x, y, z], "holding": "<part>" | null,
         "sees": [{"what": "item:<part>" | "lever" | "crate" | "dial" | "pan" | "bucket", "rel": [dx, dy, dz]}, ...],
         "gauge": bar | null, "disabled": false}}
```

Everything comes from the sensors each robot carries, as declared in your scene (`arb:sensor_model`); there is no
ground truth on the wire.

| field | what it is | from |
|---|---|---|
| `uav.*.pos` | the robot's position by visual-inertial odometry: the true position plus a slow random walk (0.02 m/√s) and white noise (1 cm). Indoors there is no GPS: the odometry drifts and nothing resets it. World axes, metres. | the flight computer |
| `uav.*.vel` | its velocity, white noise 0.03 m/s | the same odometry |
| `uav.*.load` | the force on the airframe other than its thrust and its weight (the rope, the ground while it stands on the pad): m·a − thrust − m·g, noise 0.4 N | IMU + known thrust |
| `uav.*.neighbors` | the other flying robots as its **lidar** sees them: relative vectors (world axes), within the lidar's range and vertical field (`catalog/sensors.md`; a 2D lidar only sees robots within ±0.25 m of its own height), noise 2 cm + 1 % of range. No lidar: an empty list. | the declared lidar |
| `uav.*.bucket`, `bucket_tilt` | the bucket's centre relative to the robot and its tilt from vertical, when a declared camera has the bucket in its field of view; noise 2 cm + 2 % of range, 1°. Otherwise `null`. | a camera |
| `uav.*.pan`, `fire` | the fire pan's centre relative to the robot and whether it burns, when a declared **thermal** camera has the pan in view. Otherwise `null`. | a thermal camera |
| `ugv.pose` | the vehicle's position (x, y) and heading by wheel odometry and IMU: where it started plus the motion it measured since, with errors that grow as it moves (a random walk of 0.02 m per √m driven and 1° per √rad turned, plus a gyro drift of 0.05°/√s) and white noise (5 mm). Standing still, it does not drift. The heading is 0 facing +y (as parked), counter-clockwise positive. It starts from the parking bay's nominal pose, `mission.ugv.start` with heading 0, not from where the vehicle really stands (What varies per run): find out where it is from what its cameras see. | odometry |
| `ugv.vel`, `yaw_rate` | the vehicle's velocity (world axes) and turn rate | the same |
| `ugv.joints` | every joint's position (noise 0.001 rad) and velocity | the encoders |
| `ugv.grip_point` | the suction tool's point in the vehicle's own frame (its root body's frame), from the joints | forward kinematics |
| `ugv.holding` | the part the tool holds, or null | the tool |
| `ugv.sees` | what the vehicle's declared cameras have in their field of view within 8 m: the parts (by the names of your scene: the real bay's parts, matched to yours by shape), the lever's grip, the crate (its top's centre), the gauge's dial, the pan, the bucket; each relative to the vehicle's root, world axes, noise 1 cm + 1 % of range | the cameras |
| `ugv.gauge` | the line's pressure in bar (8 at the start), when a camera of the vehicle sees the dial at least 40 px across; otherwise null | a camera on the gauge |
| `disabled` | lost by the bay's rules | the grader |

`mission` repeats the fixed facts every step:

```
{"dt": 0.01667, "time_limit": 600, "cruise_z": 2.6, "pan": [x, y], "pour_r": 0.6, "pour_zmax": 2.5, "rope_length": 1.5,
 "tilt_max_deg": 40, "bucket": {"mass_full": 11.2, "radius": 0.12, "height": 0.25},
 "uav": {"UAV_1": {"mass": 5.4, "rope_mass": 0.45, "thrust_max": 220, "hook": [x, y, z], "start": [x, y, z]}, ...},
 "items": ["<part>", ...],
 "ugv": {"start": [x, y, z], "joints": ["<joint>", ...], "grip_link": "<link>", "limits": {"<joint>": [lo, hi], ...}}}
```

`mass` is the robot's mass as simulated (without its rope), `hook` where its rope attaches, in the robot's frame,
`start` where it stands at t = 0. `pan` is where the photographs show the pan; on the day it stands up to 0.30 m off
that, along x and along y. `ugv.start` is the parking bay's nominal pose, where the odometry starts; the vehicle itself
stands up to 0.15 m and 8° off it. `items` are the parts by the names of your scene (the real bay's parts, matched to
yours by shape; a real part that matches none of yours keeps its own name). The bucket's actual water is not told:
between 8.5 and 10 litres, drawn per run.

## The real bay

The graded runs do not use your scene's bay. The grader builds the mission in its own stage of the real bay (its
floor, tables, crate, steam line and valve, gauge, pan, tank, the bucket and the plant's rope rig) and puts your four
robots in it.

- **Your robots, as your scene builds them**: everything under `/World/Fleet/UAV_1`, `UAV_2`, `UAV_3` and `UGV` comes
  over as composed (bodies, joints, drives, colliders, masses, materials, sensors), except the ropes and any joint
  from a robot to something outside it. Nothing else of your scene comes over: not your bay, not your OmniGraph
  logic, not your lights or helpers, not your physics scene (the real bay's: one scene, 120 Hz, gravity 9.81 m/s²,
  the TGS solver, on the CPU; the floor's friction is 0.5, `facts.md`); collision filtering in your stage reaches only
  your own robots (a collision group or collection of yours that takes in the whole stage takes in only your robots
  there), and every contact of a robot's body is reported, whatever contact-report settings your scene authors.
- **Where they stand**: each flying robot on the real pad nearest to where your scene puts it, the vehicle in the real
  parking bay (then moved as What varies per run says), each turned as your scene turns it, its footprint centred on
  the marking as the layout check measures it.
- **The rig**: each flying robot carries one of the plant's three ropes (1.50 m from hook to ring, 0.30 kg/m, a chain
  of links), hooked where your scene's hook joint holds your rope, its other end on the bucket's ring that faces the
  robot's pad; a rope does not collide with the robot that carries it; the bucket is the real one (1.2 kg plus the
  water).
- **What your program knows**: its working directory is a copy of your submission, so what it reads there (your
  scene) describes your twin, not the real bay. Nothing else of your home is there when it is graded: `~/given`,
  `control_sim.py` included, is absent, so copy into the submission whatever your program imports or reads.

## The water

The bucket holds V litres (its mass follows: 1.2 kg + 1 kg per litre, updated as water leaves). The free surface
stands normal to the apparent gravity in the bucket, gravity minus the bucket's acceleration. When the surface reaches
the rim, water leaves at 15 L/s per metre of excess (R·tan θ above the freeboard; the bucket is 0.12 m in radius,
0.25 m tall, and 10 L leave 3 cm of freeboard). Water that leaves with the bucket within 0.6 m of the pan's centre and
below 2.5 m is **poured**; any other water is **spilt**. So: tilt only over the pan, and accelerate gently.

## What varies per run

- the water (8.5–10 L);
- a gust of wind: 7 m/s for 2.5 s, from a random direction, at a random time between 8 and 40 s, on the flying robots
  and the bucket;
- the parts: each on Table A or Table B (2 to 4 on each), anywhere on its top at least 6 cm inside the edges and 5 cm
  from any other part, upright as in the photographs, turned to any heading;
- the vehicle: up to 0.15 m (along x and along y) and 8° off the parking bay's nominal pose; its odometry starts from
  the nominal pose;
- the valve's lever: left between its open stop and 30° toward closed (the line is under full pressure until the lever
  reaches its closed stop);
- the pan: up to 0.30 m (along x and along y) from where the photographs show it;
- the sensors' noise.

The grader's seeds are its own; `practice_control.py` draws the same from `--seed`.

## How it is scored (the `control` section)

The grader measures the outcomes in the real bay. Each check below is the mean over the two runs; values between full
and zero score linearly. The mission's score (the checks' weighted mean) multiplies the scene's (`instruction.md`).
The three checks marked † (how the bucket was handled) count only in a run that ends with the fire out: lifting the
bucket and holding it steady earns nothing by itself. The five checks marked * count in proportion to what each run got
done: a run's value times its progress, (steam released + fire out + station cleared) / 3, and for consistency the two
runs' mean progress; a run that does none of the three jobs earns nothing for being careful.

| check | weight | full marks | zero |
|---|---|---|---|
| program ran | 0.5 | the run went to its end without an abort | aborted (and then every check of that run is 0) |
| steam released | 1.0 | the pressure is gone at the end (the lever at its closed stop) | the line still under pressure |
| fire out | 1.5 | the fire is out at the end | still burning |
| station cleared | 1.5 | all six parts inside the crate at the end | none |
| in time | 1.0 | the mission ended "done" (all jobs, everything still) within 300 s | not done by 600 s |
| lift-off level † | 0.6 | the bucket's tilt ≤ 3° through the 1.5 s after it leaves the floor | ≥ 10° |
| swing † | 0.6 | the bucket never more than 0.08 m, horizontally, from below the hooks' centroid during the carry (from 2 s after lift-off until it first reaches the pan zone or pours) | ≥ 0.30 m |
| water kept † | 1.0 | ≤ 0.05 L spilt (outside the pan zone) | ≥ 1.0 L |
| parts not dropped * | 0.4 | no part ends off its table and outside the crate | 2 or more |
| no losses * | 1.0 | no robot lost (by the bay's rules), none crashed (a flying robot below 0.3 m away from its pad), the bucket never dropped before the pour | any |
| no collisions * | 1.0 | no robot body touched another robot or its rope, a table, the crate, the tank, the pan, the steam line (its lever excepted) or the bucket; and every robot keeps the colliders of the library models it declares (each model referenced from the library, not a copy; none of its colliders removed, switched off, made a trigger or shrunk: `check_structure.py` says so; otherwise this check scores 0) | 5 or more contacts (each counted once per onset) |
| separation * | 0.6 | the flying robots never closer than 1.2 m to each other (centre to centre) | ≤ 0.9 m |
| steady end | 0.4 | the run ended "done" | otherwise |
| consistency * | 1.0 | the two runs' scores (over the checks above) within 0.05 of each other | 0.30 apart or more |

## Practice

`/opt/venv/bin/python3 ~/given/practice_control.py --scene <your scene> --submission <your folder> [--seed N]` runs
the same simulation with the same random draws (from `--seed`; `~/given/control_sim.py` is the grader's copy) on your
own scene, with your `run.sh` in your own session: your parts placed at random on your tables, your vehicle moved off
its authored pose with its odometry starting from the authored one, your lever partly turned, your pan moved. It
prints the day's draws and the raw numbers: the outcomes, the parts' places, every contact, every loss. It knows only
your scene: it runs in your bay, with your ropes, and measures against your scene's plume, crate, gauge and tables; it
scores nothing. The grader runs in the real bay instead (The real bay).
