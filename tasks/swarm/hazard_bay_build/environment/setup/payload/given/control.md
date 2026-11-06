# The bay mission: your fleet's program

The second half of the submission is the program that runs the fleet through the plant's whole job (`bay_brief.md`):
close the leaking line's valve, put the fire out with the bucket, clear the assembly station into the crate. You write
the control for every robot and the planning that sequences them; how you do it is yours. After the scene checks, the
grader runs your program in **the real bay**, twice, on two days of its own (What varies per run): its own stage of the
bay as it really is, with your four robots in it exactly as your scene builds them (The real bay, below). It measures
only what happens: whether the steam is released, the fire out, the parts in the crate, how long it took, how steadily
the robots moved, what was spilt, lost or hit, and how alike the two runs were.

Your program runs as a separate process, launched as `bash run.sh` in a copy of your submission folder, and talks over
its stdin / stdout: one JSON object per line (The wire). Nothing in `~/given` runs the mission: you rehearse it in your
own twin, with a harness of your own (Rehearsing, at the end). This page says what the grader simulates and measures.

## The mission

At the start the flying robots stand on their pads, the ropes rigged (slack, lying on the floor), the bucket full beside
the tank, the fire burning, the steam leaking, the vehicle in its bay, the six parts somewhere on the two tables. The
jobs:

- **The steam**: turn the valve's lever, from wherever it was left, to its closed stop. The line's pressure then falls
  to nothing over 3 s and the leak stops (the real valve's behaviour). The vehicle's arm works the lever by pushing or
  pulling it; nothing turns it for you. The valve's packing holds the lever wherever it is: its own weight does not
  turn it, a push does.
- **The fire**: lift the bucket on its three ropes, carry it to the pan, tip it over the pan. Once 1 litre has been
  poured on the pan the fire is out (The water).
- **The station**: every part off the tables and into the crate.

**The valve, as the grader keeps it.** The lever's angle is counted from its open stop, where it points straight up;
it closes by turning toward +x (to the right, seen from the bay), a quarter turn to its closed stop. It counts as closed
from 85° on, and once closed it counts as open again only below 75°. It turns only while a body of a robot touches it
(a part the tool holds does not count) and for 0.5 s after the last touch, its own weight cancelled; the rest of the
time it stays exactly where it is. The line's pressure is 8 bar while the lever counts as open; from the moment it
counts as closed the pressure falls linearly to 0 over 3 s, and the steam is on as long as the pressure is above 0;
opened again, the line is back at 8 bar at once.

Time limit 600 s of simulation. The mission ends "done" when all three jobs are done (no pressure, the fire out, all six
parts in the crate) and everything has stayed still for 3 s: every flying robot slower than 0.3 m/s, the bucket hanging
or standing within 8° of upright, the vehicle slower than 0.1 m/s. The grader checks this every 0.5 s on the true
state; the done time is when the still period began. The mission also ends at the time limit, or when every robot is
lost.

**The bay's rules, as the grader keeps them**: a flying robot whose position (What you observe) stays inside the steam
plume's box for 1 s in a row while the steam is on is lost; the vehicle whose position stays within 2 m (horizontally)
of the pan's centre, where the pan stands that day, for 1 s in a row while the fire burns is lost. A lost robot is out
for the rest of the run: a flying robot's thrust is cut (the wind still pushes it); the vehicle's velocity targets go to
0, its position targets stay as they were and its grip lets go. Your scene's hazard logic does not come into the real
bay (the scene checks test it).

**The start.** Before your program starts, the grader sets the day up (What varies per run): the pan moved, the vehicle
moved and turned, the lever turned, each part set down 1 mm above its table's top. Then it runs 1 s of physics with
nothing commanded (no thrust, no wind, the vehicle's drives on your scene's own targets) while the parts and the ropes
settle, and only then starts the clock: t = 0, every contact so far forgotten, `mission.uav.*.start` and the vehicle's
odometry taken there. Only then is your program started.

## The wire

```
READY                                       <- within 40 s of launch
{"type": "reset"}                           -> {"ok": true}
{"type": "act", "t": 12, "obs": {...}}      -> {"action": {"thrust": {"UAV_1": [fx, fy, fz], ...}, "ugv": {...}}}
{"type": "close"}                           -> {"ok": true}, then exit
```

- **Starting.** The grader starts your program once the run is set up: `bash run.sh`, as your account, in a session of
  its own, with exactly this environment and nothing else: `PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin`,
  `HOME=/home/user`, `LANG=C.UTF-8`. Call `/opt/venv/bin/python3` explicitly (NumPy and `pxr` are there); `run.sh` may
  start anything that speaks the wire. Its stdin and stdout are the wire; its stderr goes nowhere.
- **The working directory** is a fresh copy of your submission folder (your scene included) that belongs to root: your
  program can read everything in it and write nothing, and no file in it keeps an executable bit, so start everything
  through an interpreter (`exec /opt/venv/bin/python3 controller.py`, `bash helper.sh`; not `./helper`; with `-I` Python
  leaves the script's own folder off its import path). Write logs
  and scratch files elsewhere, under `/tmp`. The submission may hold only regular files and directories: a symbolic
  link or any other kind of file anywhere in it, or more than 2 GiB in all, and the whole submission scores 0.
- **No network.** When it is graded your program has no network: it cannot open an internet socket, not even one to
  localhost; only Unix-domain sockets and pipes work.
- **`READY`, then `reset`.** The first line that is exactly `READY` must come within 40 s of launch (up to 199 other
  lines before it are skipped). The grader then sends `{"type": "reset"}` at once; any JSON object answers it.
- **stdout carries replies only.** Each request has one deadline, 10 s of wall time from when it is written, and after
  `READY` every line your program prints is read as the reply to the request in hand (blank lines are skipped). A line
  that is not a JSON object, or an `act` reply without an `action` (`"action": null` included), aborts the run; a
  second line for one request is taken as the next request's reply. Print your logs to stderr or to a file.
- **Aborts.** A missed deadline, a crash, a malformed reply or a reply without an action aborts the run: every check of
  that run scores 0. Each run starts `run.sh` afresh, after every process of your account has been stopped, and a run
  that fails on the grader's side before its first step is run once more from its set-up, `run.sh` included. So your
  program must be able to start more than once (what it left under `/tmp` is still there).
- **Your program's time.** Besides the 10 s per request, a run gives your program 30 minutes of its own time: the time
  from each request to its reply, summed over the run (the reset included). A run whose replies take longer than that
  in all is stopped and counts as aborted. The simulator's own time does not count against your program: a run that
  lasts its full 600 s (36,000 steps, and about one step in twenty asked twice) leaves your program about 47 ms per
  reply on average. (A run is also stopped, as aborted, 90 minutes of wall clock after it started, whatever the time
  went into; the simulator takes 12 to 33 minutes for a full run.)
- **Steps.** Control steps come at 60 Hz of simulation time (two physics steps of 1/120 s each). `t` counts them from
  0; `obs.time` is the simulation clock. A step's observation is taken before its action is applied (Rehearsing).
- **A step may be asked more than once with the same `t`.** About one step in twenty is asked twice, in an order
  nobody tells you: once with its observation and once with a copy in which every flying robot's `pos` is shifted by up
  to 0.4 m along each axis and the vehicle's `pose` by up to 0.4 m along x and along y (each robot and axis by its own
  amount; the heading and everything else as in the observation). Only the answer to the true observation is applied,
  and the shift does not carry into later steps. Answer each request from the observation it carries and from your
  state as it was before that `t`; count steps by `t`, not by requests, and move your state on once per `t` (the next
  step's observation continues the true one).
- **The end.** No message announces the mission's end: when the run ends (done, at the time limit, every robot lost, or
  aborted), the grader sends `{"type": "close"}`, closes your stdin, waits up to 10 s, then stops your process group and
  every process of your account. Exit on `close` or when stdin ends.

## What you command

```
{"thrust": {"UAV_1": [fx, fy, fz], "UAV_2": [...], "UAV_3": [...]},
 "ugv": {"vel": {"<joint>": rad_per_s, ...}, "pos": {"<joint>": rad, ...}, "grip": true | false}}
```

- `thrust`: one force vector per flying robot in newtons, world axes, applied to its root body (What you observe) in
  both physics steps of the control step, together with the wind. The grader clamps it in this order: F_z to ≥ 0; the
  horizontal part to F_z · tan 40° (a 40° tilt of the thrust vector from vertical, the attitude limit); then the whole
  vector to the airframe's `thrust max` (`catalog.json`, `mission.uav.*.thrust_max`; 0 when the robot's
  `arb:airframe_model` names no catalog airframe). The flying robots' attitude is held by a perfect inner loop: after
  every physics step the robot's orientation is set back to the one it had at t = 0 (as it stood on its pad: its
  heading never changes) and its angular velocity to zero; its position is pure dynamics: your thrust, gravity, the
  rope, contacts, the wind. A robot without a thrust entry gets zero. The flying robots' arms are not on the wire:
  their drives keep the targets your scene authors.
- `ugv.vel`, `ugv.pos`: velocity and position targets for the vehicle's joints, by the joint names of your scene (the
  `mission` lists them; names must be unique within the vehicle: a target goes to the first joint of that name),
  applied to the joints' drives as your scene authored them (a drive with stiffness follows a position target, one with
  damping only a velocity target). Velocity targets are clamped to ±40 rad/s, position targets to the joint's limits;
  an unknown name, or a value that is not a finite number, is ignored. A target keeps its last value until you send
  another. At t = 0 every position target is the joint's position when the mission was set up (before the 1 s settle)
  and every velocity target is 0; from the first step on the grader writes every joint's targets at each step (your
  scene's own targets apply only during the settle).
- `ugv.grip`: the vehicle's suction tool at the origin of the arm's last link (the last rigid body, in the vehicle's
  link order, of those under the prim that declares `arb:arm_model`; the `mission` names it). The grader applies it
  once per control step, before the physics. While it is on and holds nothing, it takes the nearest part whose bounding
  box (axis-aligned, in world axes) comes within 4 cm of that point and holds it rigidly: the part then follows the
  link exactly. Switched off, it lets go, and the part leaves with the link's velocity (without spin). An action without
  `grip` leaves it as it was. It holds parts only (the lever is worked by contact).
- **Shapes.** Send exactly these shapes. An `action` that is not an object (null aside: The wire) counts as an empty
  one, and so does every key an action leaves out: no thrust, the vehicle's targets and grip as they were. `thrust`,
  `ugv`, `ugv.vel` and `ugv.pos` must be objects where they are present, and each thrust entry a list of three finite
  numbers: anything else there counts as zero or aborts the run.

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

Most of it comes from the sensors each robot carries, as declared in your scene (`arb:sensor_model`), with the noise
below (Gaussian, standard deviations, along each axis); what the table calls true is exact.

**Positions.** A robot's position is the origin of its articulation's root body: the prim under the robot that carries
`ArticulationRootAPI`, or, when that prim is not a rigid body, the first rigid body below it. Every vector on the wire
is in world axes, except `hook` and `grip_point`, which are in the robot's root body's frame; lengths are in metres.

**Cameras.** A camera counts only as a `UsdGeom.Camera` with `arb:sensor_model` naming a catalog camera and that
camera's focal length and apertures (within 2 %). The grader takes it at its pose relative to the robot's root body as
your scene authors it, and keeps it there: a camera on an arm link does not follow the arm. Its field of view is the
rectangle hfov × vfov (from the focal length and the apertures) about its −Z axis, +Y up. A target is a single point,
in view when it lies inside that rectangle in front of the camera; nothing hides it. A flying robot's cameras see at
any range, the vehicle's within 8 m of the camera. Each target is reported once, by the first camera, in your scene's
prim order, that has it in view. On a flying robot any camera reports the bucket, and only a thermal one the pan and
the fire; the vehicle's `sees` reports every target, the pan included, from any of its cameras. Declarations
and sensors inside an instanceable prim are not read (`interface.md`).

| field | what it is | from |
|---|---|---|
| `uav.*.pos` | the robot's position by visual-inertial odometry: the true position plus a slow random walk (0.02 m/√s along each axis, height included) and white noise (1 cm). Indoors there is no GPS: the odometry drifts and nothing resets it. | the flight computer |
| `uav.*.vel` | its velocity, white noise 0.03 m/s | the same odometry |
| `uav.*.load` | the force on the airframe other than its thrust and its weight (the rope, the wind, the ground while it stands on the pad): m·a − T − m·g, with m = `mission.uav.*.mass`, a its true acceleration over the last control step, T the thrust as the grader clamped it for that step (counted even after the robot is lost and its thrust cut), g = (0, 0, −9.81) m/s²; noise 0.4 N. Standing on its pad it reads about (0, 0, +m · 9.81). | IMU + known thrust |
| `uav.*.neighbors` | the other flying robots as its **lidar** sees them: relative vectors from its position to theirs, true plus noise 2 cm + 1 % of range, unlabelled, in the order UAV_1, UAV_2, UAV_3 with itself left out; a robot beyond the lidar's range or outside its vertical field (`catalog/sensors.md`: a 2D lidar sees only robots within ±0.25 m of its own height, a 3D lidar those within ±45° of its horizontal) is left out; lost robots are still seen. Only the first lidar the robot declares counts. No lidar: an empty list. | the declared lidar |
| `uav.*.bucket`, `bucket_tilt` | the bucket's bottom centre (the centre of its base, on its axis, 0.25 m below the rim: the bucket body's origin) relative to the robot, noise 2 cm + 2 % of range, and the bucket's tilt from vertical (the angle of its axis, without a direction), noise 1° (so it can read a little below 0), when one of the robot's cameras has that point in view. Otherwise `null`. | a camera |
| `uav.*.pan`, `fire` | the fire pan's centre, 5 cm above the floor, where the pan stands that day, relative to the robot, noise 2 cm + 2 % of range, and whether it burns (true), when one of the robot's **thermal** cameras has that point in view. Otherwise `null`. | a thermal camera |
| `ugv.pose` | the vehicle's position (x, y) and heading by wheel odometry and IMU. At t = 0 it reads the parking bay's nominal pose, `mission.ugv.start` with heading 0, not where the vehicle really stands (What varies per run): find out where it is from what its cameras see. From then on x, y = `mission.ugv.start` + the vehicle's true displacement since t = 0 (in world axes: an error in the heading does not turn the track) + a random walk of 0.02 m per √m driven, along x and along y, + white noise 5 mm; the heading = its true turn since t = 0 + a random walk of 1° per √rad turned + a gyro drift of 0.05°/√s, which runs on while the vehicle stands still; no white noise on the heading. The heading is 0 facing +y (as parked), counter-clockwise positive, in degrees from −180 to 180. | odometry |
| `ugv.vel`, `yaw_rate` | the vehicle's velocity (world axes) and turn rate (counter-clockwise positive), true | the same |
| `ugv.joints` | every joint of the vehicle (its wheels included), in PhysX's order: position (rad, or m for a sliding joint; noise 0.001) and velocity (true) | the encoders |
| `ugv.grip_point` | the suction tool's point in the vehicle's own frame (its root body's frame), from the joints, true | forward kinematics |
| `ugv.holding` | the part the tool holds, or null | the tool |
| `ugv.sees` | what the vehicle's declared cameras have in view within 8 m: each target's point relative to the vehicle's position, true plus noise 1 cm + 1 % of range. The targets, in this order: the parts (`item:<name>`, by the names of `mission.items`: the real bay's parts, matched to yours by shape; the centre of each part's bounding box; the part the tool holds is not listed), the lever (a point on the handle 0.1955 m out from the valve's stem, which turns with the lever; the grip's centre, `requirements.md` R7, is 0.22 m out), the crate (the centre of its top), the gauge's dial (its centre), the pan (its centre where it stands that day, 5 cm above the floor), the bucket (its bottom centre) | the cameras |
| `ugv.gauge` | the line's pressure in bar (8 at the start) plus noise 0.1 bar, clipped at 0, when the first of the vehicle's cameras that has the dial in view sees it at least 40 px across (px = the camera's horizontal resolution × the dial's angular size, from its diameter, ÷ the camera's horizontal field of view); otherwise null. A closed line reads exactly 0 only about half the time: do not wait for 0. | a camera on the gauge |
| `disabled` | lost by the bay's rules (true) | the grader |

`mission` repeats the fixed facts every step:

```
{"dt": 0.016666666666666666, "time_limit": 600.0, "cruise_z": 2.6, "pan": [x, y], "pour_r": 0.6, "pour_zmax": 2.5,
 "rope_length": 1.5, "tilt_max_deg": 40.0, "bucket": {"mass_full": 11.2, "radius": 0.12, "height": 0.25},
 "uav": {"UAV_1": {"mass": 5.4, "rope_mass": 0.45, "thrust_max": 220.0, "hook": [x, y, z], "start": [x, y, z]}, ...},
 "items": ["<part>", ...],
 "ugv": {"start": [x, y, z], "joints": ["<joint>", ...], "grip_link": "<link>", "limits": {"<joint>": [lo, hi], ...}}}
```

`mass` is the robot's mass as simulated (the bodies of its articulation, without the rope), `rope_mass` that of the
plant's rope it carries, `thrust_max` its airframe's, `hook` where its rope attaches (the hook joint's point on the
robot), in its root body's frame, and `start` its position at t = 0 (true); the hook in the world is `pos` + R0 ·
`hook`, R0 the robot's orientation (held as at t = 0). `cruise_z` is the height at which the plant's rig carries its
hooks (`facts.md`); no rule uses it. `pan` is where the photographs show the pan; on the day it stands up to
0.45 m off that, along x and along y. `ugv.start` is the parking bay's nominal pose (the vehicle's position there),
where the odometry starts; the vehicle itself stands up to 0.30 m (along x and along y) and 25° off it.
`ugv.joints` are the vehicle's joints in PhysX's order, `ugv.grip_link` the suction tool's link, `ugv.limits` the
joints' limits as PhysX has them (rad, or m; an unlimited joint shows ±3.4e38). `items` are the parts by the names of
your scene (the real bay's parts, matched to yours by shape); a real part that matches none of yours keeps its own name,
`cracker_box`, `mustard_bottle`, `sugar_box`, `soup_can`, `power_drill` or `mug` (with `_2` added if one of your names
is already that). The bucket's actual water is not told: between 8.5 and 10 litres.

One `act` request as it comes over the wire, on one line. The format, the order of the keys and the rounding are the
grader's; the numbers are made up (another bay, another fleet), so read nothing about this bay or a fleet into them:

```
{"type": "act", "t": 3720, "obs": {"t": 3720, "time": 62.0, "mission": {"dt": 0.016666666666666666, "time_limit": 600.0, "cruise_z": 2.6, "pan": [0.0, 2.5], "pour_r": 0.6, "pour_zmax": 2.5, "rope_length": 1.5, "tilt_max_deg": 40.0, "bucket": {"mass_full": 11.2, "radius": 0.12, "height": 0.25}, "uav": {"UAV_1": {"mass": 8.4127, "rope_mass": 0.45, "thrust_max": 360.0, "hook": [0.0, 0.0, -0.162], "start": [2.45, -0.45, 0.2843]}, "UAV_2": {"mass": 8.4127, "rope_mass": 0.45, "thrust_max": 360.0, "hook": [0.0, 0.0, -0.162], "start": [4.1, -1.7, 0.2843]}, "UAV_3": {"mass": 8.4127, "rope_mass": 0.45, "thrust_max": 360.0, "hook": [0.0, 0.0, -0.162], "start": [4.15, 0.6, 0.2843]}}, "items": ["cracker", "mustard", "sugar", "soup", "drill", "mug"], "ugv": {"start": [-1.15, -3.5, 0.0731], "joints": ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr", "arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6"], "grip_link": "tool_link", "limits": {"wheel_fl": [-3.4028234663852886e+38, 3.4028234663852886e+38], "wheel_fr": [-3.4028234663852886e+38, 3.4028234663852886e+38], "wheel_rl": [-3.4028234663852886e+38, 3.4028234663852886e+38], "wheel_rr": [-3.4028234663852886e+38, 3.4028234663852886e+38], "arm_joint_1": [-6.2831854820251465, 6.2831854820251465], "arm_joint_2": [-6.2831854820251465, 6.2831854820251465], "arm_joint_3": [-3.1415927410125732, 3.1415927410125732], "arm_joint_4": [-6.2831854820251465, 6.2831854820251465], "arm_joint_5": [-6.2831854820251465, 6.2831854820251465], "arm_joint_6": [-6.2831854820251465, 6.2831854820251465]}}}, "uav": {"UAV_1": {"pos": [-0.5069, 1.7184, 2.7731], "vel": [-0.1478, 0.2409, 0.0217], "load": [16.2371, 0.4127, -38.6904], "neighbors": [[1.1592, -0.6838, 0.0141], [1.1874, 0.6702, -0.0087]], "bucket": [0.7718, -0.0612, -1.8109], "bucket_tilt": 2.87, "pan": [0.1849, 0.6157, -2.7533], "fire": true, "disabled": false}, "UAV_2": {"pos": [0.6147, 1.0012, 2.7346], "vel": [-0.1951, 0.1823, -0.0315], "load": [-8.3017, 13.8562, -38.9915], "neighbors": [[-1.1803, 0.6811, -0.0112], [-0.0069, 1.3462, 0.0158]], "bucket": [-0.3326, 0.6981, -1.7367], "bucket_tilt": 1.64, "pan": [-0.9864, 1.2208, -2.6611], "fire": true, "disabled": false}, "UAV_3": {"pos": [0.6572, 2.3897, 2.7615], "vel": [-0.1702, 0.2167, 0.0044], "load": [-7.9128, -14.2209, -38.4472], "neighbors": [[-1.1652, -0.6693, 0.0054], [0.0121, -1.3577, -0.0093]], "bucket": [-0.4115, -0.7392, -1.7488], "bucket_tilt": 2.33, "pan": null, "fire": null, "disabled": false}}, "ugv": {"pose": [-0.0986, -1.0183, 1.27], "vel": [0.0011, 0.0004, 0.0002], "yaw_rate": 0.06, "joints": {"wheel_fl": [17.20416, 0.0021], "wheel_fr": [-3.88257, -0.0017], "wheel_rl": [16.95533, 0.0019], "wheel_rr": [-4.10872, -0.0024], "arm_joint_1": [0.03126, 0.0012], "arm_joint_2": [-1.10593, 0.2417], "arm_joint_3": [1.92874, -0.3108], "arm_joint_4": [-2.39042, 0.0695], "arm_joint_5": [-1.57164, 0.0003], "arm_joint_6": [0.01893, -0.0001]}, "grip_point": [0.6125, -0.0418, 0.5873], "holding": "sugar", "sees": [{"what": "item:soup", "rel": [0.9586, 0.4431, 0.3874]}, {"what": "crate", "rel": [-0.0517, 0.6342, 0.2386]}, {"what": "bucket", "rel": [0.3829, 2.7023, 0.9145]}], "gauge": null, "disabled": false}}}
```

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
- **Where they stand**: each flying robot on the real pad nearest to where your scene puts it (the three assigned so
  that their distances add up to the least), the vehicle in the real parking bay (then moved as What varies per run
  says), each by one horizontal shift of the whole robot: heights and headings as your scene has them, its footprint
  (the box round everything under the robot but its `Rope`) centred on the marking as the layout check measures it.
- **The rig**: each flying robot carries one of the plant's three ropes (1.50 m from hook to ring, 0.30 kg/m, a chain
  of links), hooked where your scene's hook joint holds your rope (the same body, the same point), its other end on the
  bucket's ring that faces the robot's pad; it starts slack, lying on the floor. A rope never collides with the bucket;
  its contacts with the robot that carries it never count (and its links that start next to that robot's colliders do
  not collide with it at all). The bucket is the real one (1.2 kg plus the water; its rings at the rim, 0.14 m from its
  axis).
- **A fleet that does not fit** aborts the run before your program starts (every check of that run scores 0): a robot
  with no articulation root outside its rope, a flying robot with no hook joint between one of its bodies and its rope,
  or a hook that, with its robot on the real pad, stands more than 1.45 m from its ring.
- **What your program knows**: its working directory is a copy of your submission, so what it reads there (your
  scene) describes your twin, not the real bay. Nothing else of your home is there when it is graded: `~/given` is
  absent, so copy into the submission whatever your program imports or reads.

## The water

The bucket holds V litres (its mass follows: 1.2 kg + 1 kg per litre, updated as water leaves). The free surface
stands normal to the apparent gravity in the bucket, gravity minus the bucket's acceleration. When the surface reaches
the rim, water leaves at 15 L/s per metre of excess (R·tan θ above the freeboard; the bucket is 0.12 m in radius,
0.25 m tall, and 10 L leave 3 cm of freeboard). Water that leaves while the bucket's bottom centre is within 0.6 m
(horizontally) of the pan's centre, where the pan stands that day, and no higher than 2.5 m is **poured**; any other
water is **spilt**. So: tilt only over the pan, and accelerate gently.

The grader's model, once per control step, from the bucket body's true motion: the acceleration is the velocity's
change over the step, smoothed (a ← 0.7 a + 0.3 a_step); θ is the angle between the bucket's axis and the apparent
gravity's upward direction; the freeboard is 0.25 m − V / (π · 0.12²) (V in m³); water leaves at 15 L/s per metre of
max(0, 0.12 m · tan θ − freeboard), and at 20 L/s once θ ≥ 89°, for that step's 1/60 s.

## What varies per run

The two runs are the grader's own days, drawn once from these ranges and kept hidden: every submission meets the same
two. On each day:

- the water: between 8.5 and 10 L;
- the wind: 2 or 3 gusts on the flying robots and the bucket (not on the ropes or the vehicle): one that starts 2 to
  12 s after the bucket first leaves the floor (lift-off, as How it is scored defines it), and 1 or 2 more that start
  at times between 8 and 240 s; each 4 to 10 m/s for 1.5 to 5 s, steady while it blows, from its own horizontal
  direction. A gust pushes along its direction with 0.5 · 1.2 kg/m³ · A · v², A = 0.12 m² on each flying robot,
  whatever its airframe, and 0.06 m² on the bucket (7.2 N and 3.6 N at 10 m/s); gusts that overlap add up;
- the parts: each on Table A or Table B (1 to 5 on each), anywhere on its top with its footprint at least 4 cm inside
  the edges and 5 cm from any other part's, turned to any heading; the cracker box, the sugar box, the mustard bottle
  and the power drill each standing as in the photographs or lying on a broad side (a quarter turn, either way, about
  the long side of its footprint), the soup can and the mug always standing;
- the vehicle: up to 0.30 m (along x and along y) and 25° off the parking bay's nominal pose, turned about its position,
  so it may stand partly over the bay's marking; its odometry starts from the nominal pose;
- the valve's lever: left between its open stop and 60° toward closed (the line is under full pressure until the lever
  counts as closed);
- the pan: up to 0.45 m (along x and along y) from where the photographs show it;
- the sensors' noise.

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
| in time | 1.0 | the mission ended "done" (all jobs, everything still) with its still period begun by 300 s | not done by 600 s |
| lift-off level † | 0.6 | the bucket's tilt ≤ 3° through the 1.5 s after lift-off | ≥ 10° |
| swing † | 0.6 | the bucket's bottom centre never more than 0.08 m, horizontally, from the hooks' centroid during the carry (from 2 s after lift-off until the pour begins, whenever the bucket is outside the pan zone) | ≥ 0.30 m |
| water kept † | 1.0 | ≤ 0.05 L spilt (The water) | ≥ 1.0 L |
| parts not dropped * | 0.4 | no part dropped | 2 or more |
| no losses * | 1.0 | no robot lost (by the bay's rules), none crashed, the bucket never dropped | any |
| no collisions * | 1.0 | no contact (below); and every robot keeps the colliders of the library models it declares (each model referenced from the library, not a copy; none of its colliders removed, switched off, made a trigger or shrunk: `check_structure.py` says so; otherwise this check scores 0) | 5 or more contacts |
| separation * | 0.6 | the flying robots never closer than 1.2 m to each other (position to position, from t = 0 on) | ≤ 0.9 m |
| steady end | 0.4 | the run ended "done" | otherwise |
| consistency * | 1.0 | the two runs' scores (over the checks above) within 0.05 of each other | 0.30 apart or more |

What the grader measures, on the true state:

- **Lift-off**: the first moment the bucket's bottom centre is more than 0.08 m above the floor while all three hooks
  are within 1.7 m of the centre of its rim.
- **The pan zone**: within 0.6 m (horizontally) of the pan's centre, where the pan stands that day. **The pour** begins
  when water first leaves over the pan (The water).
- **The bucket dropped**: its bottom centre below 0.03 m more than 1 s after lift-off and before the pour has begun;
  setting the bucket down before the pour counts as dropping it.
- **A crash**: a flying robot whose position comes below 0.30 m more than 0.6 m (horizontally) from where it stood at
  t = 0; landing anywhere but on its own pad counts.
- **Where a part is**: in the crate when the centre of its bounding box lies inside the crate's box seen from above and
  no more than 0.10 m above the crate's top; on a table when that centre lies inside the table's box seen from above
  and no lower than 2 cm below the table's top; anywhere else the part is dropped. A part still held at the end is
  neither in the crate nor dropped.
- **Contacts**: a body of a robot (its rope aside) touching another robot or another robot's rope, a table, the crate,
  the tank, the pan, the steam line (its valve body and gauge included, its lever not) or the bucket. The floor, the
  parts, the lever, the floor markings and the calibration mat are free. Each robot and each thing it touches count
  once per onset: a contact that has been gone for 0.5 s starts a new one.

## Rehearsing

Nothing in `~/given` runs the mission: rehearse it in your own twin, with a harness of your own that keeps the
grader's model as this page gives it (to rehearse a day, move your twin's pan, vehicle, lever and parts as What varies
per run says). The grader runs Isaac Sim 6.2 (`SimulationContext`, physics at 1/120 s on the CPU, nothing rendered) on
the real bay's physics scene, and drives and reads the bodies through PhysX's tensor views (articulation and rigid-body
views; the forces applied at every physics step). One control step:

1. the observation of the step: the noise drawn, the odometry's random walks moved on;
2. your action: the thrusts clamped (What you command), the vehicle's targets written, the grip applied;
3. two physics steps of 1/120 s. Before each: the thrust and the wind on each flying robot's root body (no thrust on a
   lost one), the wind on the bucket, the lever kept where it is unless a robot has touched it within 0.5 s, the held
   part put where the grip link carries it. After each: each flying robot's orientation set back to its orientation at
   t = 0 and its angular velocity to zero, the contacts read;
4. the water (The water); the bay's rules (the valve, the pressure, the fire, the hazards); the measurements (lift-off,
   the bucket's tilt, swing and height, the separation, crashes); every 30 steps, where the parts are and whether the
   mission is done.
