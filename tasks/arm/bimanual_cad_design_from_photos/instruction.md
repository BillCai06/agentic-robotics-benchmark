# Re-model a modified dual-arm unit in FreeCAD from a photo set

You are the mechanical designer of a lab that builds dual-arm research units: an aluminium column on a base plate
carrying two seven-joint arms, each ending in a two-finger gripper module. One unit was modified by a student (some
links lengthened or shortened, a spacer ring added under an actuator output, lightening cut-outs milled; the two arms
were not changed alike) and its CAD is lost. Before the unit was shipped to a partner lab, a colleague took it apart and
photographed everything on the workbench with a phone: the made parts laid out on a cutting mat, each part turned over
once, close-ups, then the re-assembled unit in three poses and its grippers closed and open. The photos are all there
is: nobody can measure the unit again. Re-create the unit in FreeCAD so it can be re-made and simulated: the made parts
as solids around the bought actuators, the bought parts placed, assembled with the unit's joints.

Your work is graded against the photographed unit's ground truth, which only the grader has: how well your mechanism,
your parts' features and shapes, their masses and centres of mass match the unit's, and whether the result works
(actuators free of part material and held, no collision over the unit's range of motion, within an arm and between the
arms). Each arm is graded on its own.

## How you work

- Do the work in FreeCAD through the FreeCAD MCP server `freecad`: it drives the FreeCAD 1.1 session running on this
  machine (documents, objects, views, and `execute_code` / `execute_code_async`, which run Python inside that session,
  with FreeCAD's modules: Part, PartDesign, Sketcher, Assembly, Import, Qt; its state persists between calls). Build the
  parts, place the bought parts, set up the assembly and save the document there, as a designer working in FreeCAD
  would.
- Your other tools (reading files and images, a shell) are there for everything around it: looking at the photos and the
  bought parts' documents, cropping a detail, notes and checks. There is no internet access. In the shell, the image's
  Python with its scientific packages (numpy, scipy, Pillow, trimesh and others) is `/opt/venv/bin/python3`; the
  `python3` on your `PATH` may be a bare interpreter.
- `get_view` and the screenshots the object tools return show FreeCAD's 3D view of the session. `Read` shows a photo
  scaled down to fit; to look at a detail at full resolution, crop it to a file first and read the crop. Keep scratch
  files under `/home/user/work/`.

## Given inputs (in `/home/user/`)

**`photos/photo_001.jpg` ... `photo_153.jpg`**: the fixed photo set of the unit. A paper tag printed with the part's id
lies next to every part in the teardown photos: `T01` is the base plate, `L01` ... `L14` are the parts of the left arm,
`R01` ... `R14` those of the right arm. Left and right are the unit's own: seen from behind the unit, with the elbows
bending away from you, the left arm is on the left.

| photos | what the photographer recorded |
|---|---|
| 001-006 / 007-012 | teardown: T01 on the mat, first side up, six directions / turned over onto another face |
| 013-018 / 019-024 | L01, L02, L03, L04: first side / turned over |
| 025-030 / 031-036 | L05, L06, L07, L08: first side / turned over |
| 037-042 / 043-048 | L09 ... L14: first side / turned over |
| 049-054 / 055-060 | R01, R02, R03, R04: first side / turned over |
| 061-066 / 067-072 | R05, R06, R07, R08: first side / turned over |
| 073-078 / 079-084 | R09 ... R14: first side / turned over |
| 085-127 | close-ups (actuator mounting and output faces, pin ends, cut-outs, the spacer ring), the parts laid out as: 085 T01 (first side); 086 T01 (turned over); 087-090 L01-L04 (first side); 091-093 L01-L04 (turned over); 094-096 L05-L08 (first side); 097-099 L05-L08 (turned over); 100-104 L09-L14 (first side); 105-106 L09-L14 (turned over); 107-111 R01-R04 (first side); 112-115 R01-R04 (turned over); 116-118 R05-R08 (first side); 119-121 R05-R08 (turned over); 122-124 R09-R14 (first side); 125-127 R09-R14 (turned over) |
| 128-133 | the re-assembled unit on the bench top (the surface it stands on), configuration **Z**: every actuator at its zero position, the grippers closed |
| 134-139 | configuration A (joint values not recorded; the fingers closed) |
| 140-145 | configuration B (joint values not recorded; the fingers open) |
| 146-147 / 148-149 | configuration Z, the left / right gripper close up, fingers closed |
| 150-151 / 152-153 | the same, the fingers opened as far as they go |

All photos come from one phone camera (wide lens, about 20-30 mm 35 mm-equivalent; not calibrated; no camera pose was
recorded; exposure and white balance vary a little). The actuators and the gripper modules are installed in the
assembled photos. Scale references, physical objects in the scene, with their nominal sizes:

- cutting mat: green, A3 (420 x 297 mm), printed grid every 10 mm with every fifth line bold, numbered in cm; 2 mm thick,
  the parts lie on its top face;
- checker card: 10 x 7 squares of 12 mm (9 x 6 inner corners), 6 mm white margin, on a 132 x 96 x 1 mm card lying flat on
  the mat in every teardown and close-up photo;
- A4 checker board: 10 x 7 squares of 25 mm (9 x 6 inner corners), 10 mm white margin, on a 270 x 195 x 3 mm board standing
  beside the assembled unit;
- two steel rules (300 mm scale with 1 mm ticks, 316 x 25 x 0.9 mm), one on the mat, one on the bench;
- an ID-1 card (bank-card size, 85.6 x 53.98 mm, 0.76 mm thick) on the bench.

**`bought/`**: one folder per bought part, each with a datasheet page, a dimensioned drawing, the vendor STEP and a
keep-out STEP (the envelope the grader intersects with your parts). The unit's purchase list:

| part | quantity | folder |
|---|---|---|
| joint actuator HXA-9862 (98 mm frame) | 4 | `bought/HXA-9862/` |
| joint actuator HXA-5756 (57 mm frame, large output flange) | 2 | `bought/HXA-5756/` |
| joint actuator HXA-5753 (57 mm frame) | 2 | `bought/HXA-5753/` |
| joint actuator HXA-5746 (57 mm frame, short) | 6 | `bought/HXA-5746/` |
| gripper module OGM-25-L (left-hand): body OGM-25-B-L, fingers OGM-25-FA-L and OGM-25-FB-L | 1 | `bought/OGM-25-L/` |
| gripper module OGM-25-R (right-hand): body OGM-25-B-R, fingers OGM-25-FA-R and OGM-25-FB-R | 1 | `bought/OGM-25-R/` |
| column stand LPS-CS1-A (60 x 60 x 750 mm profile, two corner brackets, a diagonal brace) | 1 | `bought/LPS-CS1-A/` |

- **Actuators.** In each actuator's own frame, +z is the output axis, from the housing towards the output face, with
  the origin on that axis at the housing's rear face; a positive command turns the output about +z relative to the
  housing (right-hand rule). The output flange turns; the housing bolts to the structure through its threaded mounting
  faces (datasheet tables H1-H4).

  | actuator | housing | output face (z, dia.) | output bolts | mass |
  |---|---|---|---|---|
  | HXA-9862 | dia. 98, 61.7 mm long | z = 61.7, dia. 40 | 6 x dia. 4.2 on PCD 32 | 896 g |
  | HXA-5756 | dia. 57, 56.5 mm long | z = 56.5, dia. 37.5 | 6 x dia. 2.5 on PCD 27 | 362 g |
  | HXA-5753 | dia. 57, 53.25 mm long | z = 53.25, dia. 35 | 6 x dia. 2.5 on PCD 27 | 362 g |
  | HXA-5746 | dia. 57, 46 mm long | z = 46, dia. 35 | 6 x dia. 2.5 on PCD 27 | 300 g |

- **Gripper modules.** The module frame has its origin on the drive axis at the mounting face (the face that bolts to an
  actuator's output: 6 x dia. 3.4 on PCD 27, seated on the output face), +z along that axis away from the actuator, +x
  from finger hinge B to finger hinge A. The body and the two finger pieces of a module share that frame at the closed
  stop (fingers at zero). Both hinges are parallel to z: hinge A through (18, -68) and B through (-18, -68) in the
  left-hand module, (18, 68) and (-18, 68) in the right-hand one. The fingers are geared 1:1 and open up to 45 deg:
  left-hand module A about +z, B about -z; right-hand module A about -z, B about +z. Tool point (the module's datasheet):
  (0.09, -159.78, -34.14) mm in the left-hand module's frame, (0.06, 159.78, -34.17) in the right-hand one's. Masses:
  body 412 g, finger A 120 g, finger B 110 g.
- **Column stand.** Its frame: origin on the column's axis at its bottom end (the face that stands on the base plate),
  +z up, +x towards the brace side. Mass 5.83 kg.
- **The actuators' controllers** stop each joint at these angles from configuration Z, in the joint's positive sense
  (below), counting the actuators of an arm from the column outwards:

  | actuator | left arm (deg) | right arm (deg) |
  |---|---|---|
  | 1st | -200 ... +80 | -80 ... +200 |
  | 2nd | -10 ... +190 | -190 ... +10 |
  | 3rd | -90 ... +90 | -90 ... +90 |
  | 4th | -140 ... 0 | -140 ... 0 |
  | 5th | -90 ... +90 | -90 ... +90 |
  | 6th | -45 ... +45 | -45 ... +45 |
  | 7th | -90 ... +90 | -90 ... +90 |

## Deliverable: `/home/user/submission/unit.FCStd`

One FreeCAD document, saved there, holding:

1. **Twenty-nine PartDesign Bodies**, one per made part, each labelled with the id on its paper tag (`T01`, `L01` ...
   `L14`, `R01` ... `R14`), each one valid solid, in millimetres. Every made part is aluminium alloy (2.70 g/cm3).
2. **The bought parts**, each imported from its STEP file and placed where it is mounted: the fourteen actuators, both
   gripper modules (the body and the two finger pieces of each, the left-hand module on the left arm), the column stand.
   Move an imported part with its Placement only; do not transform its geometry (the grader recognises each bought part
   by its shape in its own frame).
3. **One Assembly** (FreeCAD's Assembly workbench: an Assembly object with its joints) holding every part: a Grounded
   joint on the base plate; Fixed joints for everything rigidly attached (parts bolted together, each actuator housing
   to the part it is bolted to, the column stand to the base plate, each gripper body to its actuator's output); one
   Revolute joint for every hinge of the unit, that is every place where two parts turn relative to each other about
   an axis, between the two parts it connects. Enable both angle limits (Min / Max angle) on every joint whose angle is
   independent (those driven by an actuator, directly or through a linkage, and each finger) and set them at the range
   the joint moves through from the saved pose, every other joint at its zero, until a collision or its controller's
   stop; for the fingers, both turning together (they are geared) from the closed stop until a collision or the
   module's open stop. A joint whose angle follows from others in a closed loop needs no limits.
4. **Saved in configuration Z**, the pose of photos 128-133.

The grader reads the document as saved; anything not saved in `unit.FCStd` is not seen.

## How it is graded

The grader reads your document with FreeCAD and works out your mechanism from it: parts by their labels, bought parts
by their shape, links = parts and bought parts held together by Fixed joints, the Revolute joints between links matched
to the unit's by the two links they connect (in either order; Gears and other coupling joints are ignored). It places
your grounded torso (the base plate, the column stand and the parts fixed to them) onto the unit's and compares
everything with the unit at configuration Z. A joint's positive sense is its driving actuator's positive rotation about
the actuator's own +z (a joint driven through a linkage takes its actuator's sense); a finger opens in the positive
sense; every zero is the saved pose. The tool points and fingertips are the gripper modules' own (their datasheets),
carried by your module placements.

Each arm is graded on its own; the base plate counts in both. An arm whose mechanism differs from the unit's (a link
not connected, a missing or extra joint between its links) scores 0 on its items; the other arm is still scored. Each
of the unit's links is matched to the rigid link of yours that holds most of its parts' volume; a part fixed to another
link than the unit's is graded where you put it (it moves with that link: masses, centres of mass and collisions
follow). An actuator whose housing is on the other side of its joint than the unit's, or of another type than the
unit's, still drives its joint (its sense follows the side it is mounted on); its mounting is scored against the
unit's in `component_fit`.

**Gates** (each 0 or 1, multiplied into every score):

- `cad_valid` (the unit): the document opens; 29 bodies labelled T01, L01 ... L14, R01 ... R14, T01 one valid
  watertight solid (an arm part that is not one valid watertight solid scores its arm 0); every Fixed and Revolute joint
  readable and joining two of the Assembly's parts; the bought parts of the purchase list (the actuators counted
  together); a grounded torso holding the base plate and the column stand; the document was not written after the
  session ended.
- `public_model_reuse` (the unit): no part is a published model.
- `cad_json_consistent` (per arm, multiplied into that arm's score): each actuator-driven joint's axis lies on its
  actuator's output axis, each finger's axis on its module's hinge, within 1 deg and 0.5 mm; the linkage's pins are
  parallel to its crank; the two frames of every joint coincide at the saved pose.

**Criteria** (weights within an arm), every band absolute against the unit: band(e; full, zero) = 1 at |e| <= full, 0 at
|e| >= zero, linear between.

- `kinematic_fidelity` (0.30): each joint axis (0.3 / 3 mm off the unit's axis, 0.5 / 5 deg), the arm's key lengths
  between its joint axes, the output faces of its 1st, 5th and 7th actuators and its tool point (0.3 / 3 mm), the tool
  point and both fingertips over 64 random poses inside the unit's ranges (median 0.5 / 8 mm, 90th percentile 1 / 15 mm),
  and how well your joint limits match the unit's ranges (overlap over union, per joint).
- `feature_fidelity` (0.25): cut-outs (position 0.3 / 3 mm, area 10 / 50 %), the spacer ring's thickness (0.3 / 3 mm),
  each part's extents (0.3 / 3 mm).
- `shape_fidelity` (0.25): each part against the unit's after a rigid alignment: 90th-percentile surface distance
  (0.3 / 2.5 mm) and volume IoU (1 - IoU 0.05 / 0.40).
- `mass_properties` (0.10): each link's mass (solid volume x 2700 kg/m3 + the bought parts' masses; 2 / 15 %) and centre
  of mass at configuration Z (0.5 / 6 mm).
- `component_fit` (0.05): each actuator free of part material (its keep-out), held on its housing's mounting face and
  on its output face as in the unit; an actuator of another type than the unit's at a joint scores 0.
- `interference_free` (0.05): no collision between the links of an arm and the torso over random poses inside the
  unit's ranges where the unit itself is collision-free (weight 0.75), and between the two arms (0.25).

Floors per arm: the mean over the arm's parts of clamp((IoU - 0.20) / 0.30, 0, 1) multiplies `kinematic_fidelity` and
`feature_fidelity`; that times band(median tool-point error; 15, 60 mm) multiplies `mass_properties`, `component_fit` and
`interference_free`. The arm score is the weighted mean; `design_score` is the mean of the two arm scores. The
photographed unit itself scores 1.0 on everything.

## Practical notes

- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that limit
  is not graded at all, so a late finish scores nothing. Note the time when you begin (for example print
  `datetime.datetime.now()` through `execute_code`), save a complete document early and keep saving it, and end the
  session yourself (stop working and say that you are done) at least 15 minutes before the limit.
- Environment fact (harness limit): a response that reaches the 64k output-token limit is lost and, after at most a few
  retries, ends the session with whatever is in `/home/user/submission/` at that moment. Keep every single response
  well below that limit.
