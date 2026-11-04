# Specify seven upgraded replicas of a two-arm robot that is no longer in the lab

A lab built one two-arm robot last year (the "reference pair"): two seven-joint arms, each with a two-finger gripper,
on one pedestal column. A customer now orders seven units with the same kinematics, the same machined parts and the
same programmed stops, fitted with new joint modules, new bus boards and a new power supply for a heavier, faster
duty: the two arms carry a loaded tray together. The reference pair has been shipped to a partner lab and cannot be
measured again. What is left: a colleague's photo set of it on the workbench, the build log, the module vendors'
datasheet pages, a distributor quote, two incoming-inspection reports of module lots and the customer's brief.

You are the requirements engineer. Recover from this evidence what the replicas need (the pair's key dimensions,
stops and module pockets, the working area, the payload, the loads the customer's acceptance duty puts on every joint
of both arms, the real capability of the modules you would buy), choose the modules, the bus and the supply, and write
it down traced to the evidence. You are scored on the OUTCOME in the real world of that pair: how well your sketch
replicates the real reference pair, how close your derived loads are to the real ones, whether your BOM passes the
customer's acceptance procedure on the real pair with the real module lot, and what it costs.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `brief.md` | the customer brief, clauses `[B-01]..[B-16]`, including the ACCEPTANCE PROCEDURE (B-14, public and exact, including the model of a module lot's capability), the deliverables and Appendix D (the frames, points, angle conventions and geometric definitions of every graded quantity) |
| `build_log.md` | the colleague's notes `[BL-..]` on the build and on the photo session |
| `photos/PH-01.jpg .. PH-41.jpg`, `photos/photos.json` | the survey photo set as the colleague left it. `photos.json` lists per photo its public label and the focal-length class of the phone camera (main or 2x tele). Camera poses and intrinsics are NOT known and the lenses are not calibrated |
| `datasheets/DS-*.png` | vendor pages (images) of the joint modules, bus boards and PSUs, plus the module vendor's erratum sheet; nominal (typical) values in each vendor's own units |
| `price_list.md` | the distributor quote: every orderable part id, role, housing class, protocol, list price, and the lot the modules ship from |
| `lot_reports/LR-*.png` | two incoming-inspection reports (scanned pages, images only) of module lots |
| `schemas/` | JSON schemas of the three deliverables (`requirements`, `bom`, `kinematic_sketch`) and their unit table, and `catalogue.schema.json` (the format in which the public grading modules take a transcription of the vendor pages) |
| `grader_public/` | the grader's own public modules (acceptance procedure, lot model, optimum search (the cheapest passing BOM), repair, every criterion). They contain no vendor, price or lot number: they take a catalogue in the format of `catalogue.schema.json` and a pair model (an MJCF plus a JSON, formats in `grader_public/procedure.py`) as inputs, and carry nothing of the real pair |
| `schema_check.py` | the structure gate G0 on your submission (no ground truth involved) |

There is no checker of the real pair and no way to query it: the real pair, bench and module lots exist only in the
grader, which uses the vendors' own numbers.

## What you deliver (`/home/user/submission/`)

- `requirements.yaml`: the traced requirements sheet. One item per requirement the documents or the photos fix
  (`id` REQ-..., `quantity` from the schema's vocabulary, `comparator`, `bound`, `unit`, `source_id` = the clause
  (`B-06`, `BL-03`), the photo (`PH-41`), the page (`DS-<part id>`), the lot report (`LR-yyyy-nnnn` or `LR`) or
  `derived`) plus the 47 derived items of brief B-16 for YOUR BOM (source `derived`, with a `derivation` note).
- `bom.json`: `bus_voltage_v` (24 or 48) and one line per part: one `module` line with `count` 1, its `arm` (`left`,
  `right`) and its `joint` (`j1`..`j7`, `j8` = the gripper) for each of the 16 joints; the `bus_board` line(s); exactly
  one `psu` line with `count` 1. Every `part_id` from `price_list.md`.
- `kinematic_sketch.json` (SI: m, rad): `geometry` = the Appendix-D quantities, each `{nominal, tol}`: the shared
  `shoulder_height`, `shoulder_half_span`, `shoulder_forward_offset`, and per arm (`left`, `right`) `upper_arm`,
  `forearm`, `wrist_to_tcp`, `finger_lever`, `grip_stroke`; `joint_ranges` = per arm the programmed stops
  `[lower, upper]` of `j1`..`j7` in the Appendix-D conventions and the gripper's `j8` = `[0, open angle]`; `joints` = per
  arm and joint `{module_id (= the BOM line), body_class}` (`body_class` = the class of the POCKET at that joint:
  `large`, `medium` or `small`).
- `notes/` (optional): your method, code, intermediate estimates, your catalogue transcription. Recorded, never scored.

## How it is scored

Gates (multiply every criterion): **G0 structure** (the files parse and pass `schema_check.py`) and **G1 submission
integrity** (nothing in `/home/user/submission` may be written after your session ends, and the folder itself must stay
the one your session wrote: a symbolic link, or a folder replaced or renamed into place after the session, scores 0;
any process you leave running is stopped when grading starts). Then five criteria, each in [0, 1], weighted:

| criterion | weight | what is measured | band (linear between full and zero credit) |
|---|---|---|---|
| `kinematic_replication` | 0.30 | your sketch vs the real reference pair (Appendix D): lengths group (0.55 of the criterion) = the 3 shared and 2 x 3 per-arm lengths, stops group (0.30) = the 30 stops (2 arms x 7 joints x lower / upper, and the 2 grippers' open stops), gripper group (0.10) = the 2 x 2 finger levers and strokes, body classes (0.05) = the 16 pocket classes, exact | lengths: full <= 1 mm, zero >= 6 mm; stops: full <= 1 deg, zero >= 6 deg; gripper: full <= 0.5 mm, zero >= 3 mm |
| `requirements_sheet` | 0.10 | facet F1 between your non-derived items and a hidden reference sheet valued in the real world (the brief's customer requirements incl. the payload band, and the items the photos fix: `tray_x_min`, `tray_x_max`, `tray_y_min`, `tray_y_max`, `tray_height_min`, `tray_height_max`, `handle_spacing`, `gripper_opening_min`); an item matches a reference item only with the same `quantity` AND the same `comparator` (`<` and `<=` count as the same comparator, a reversed one does not match; the comparator convention is in the description of `schemas/requirements.schema.json`); an item traced to a source other than the reference item's clause(s) keeps half its weight; invented or wrong-valued items lower the precision | brief items (incl. the payload band): relative error full <= 2 %, zero >= 4 %; photo lengths: error full <= the larger of 1 % and 1.5 mm, zero >= the larger of 6 % and 6 mm; angles: full <= 1.5 deg, zero >= 6 deg |
| `derived_requirements` | 0.20 | the 47 derived items of B-16 (per arm and joint: peak torque, duty-RMS load torque, peak speed; per arm the grip torque and the moving mass; the peak supply current) vs their values in the real world with YOUR BOM installed on the real pair and the real delivered lot (inverse dynamics of the real pair over the real acceptance duty); a missing item scores 0 | relative error, normalised by max(true value, floor) with floors 0.05 N m, 0.2 rad/s, 0.5 A, 0.02 kg: full <= 3 %, zero >= 15 % |
| `bom_acceptance` | 0.10 | the acceptance procedure of B-14 run on the real pair with the weakest expected unit of the real delivered lot (B-14 e): the pair is accepted (1) iff all 19 items pass (14 joint modules, 2 grippers, PSU, bus / boards, budget), else 0 | pass / fail |
| `bom_economy` | 0.30 | `max(0, 1 - (J_eff - J*) / (2 x the engineering-change cost)) x taper`: `J*` = the cheapest BOM that passes the acceptance procedure on the real pair with the real lot (recomputed live by the grader; an untested vendor counted as B-10 says, here and in `bom_acceptance`); `J_eff` = your BOM after the grader's deterministic repair: while an item fails, the failing line is replaced by the cheapest line that makes it pass (module lines: the left arm's j8 down to j1, then the right arm's; then the boards; then the PSU; the bus voltage is never changed) and the brief's engineering-change cost is added per replaced line; 2 x the engineering-change cost of B-13 = one joint changed on both arms: a BOM that much dearer than `J*` earns nothing; taper = 1 if `J_eff` <= the B-12 ceiling, else `max(0, 1 - (J_eff - ceiling) / (0.1 x ceiling))` | full at `J_eff` = `J*`, zero at `J*` + 2 engineering changes |

`overall` = the weighted mean. The acceptance procedure and every formula above are public (brief B-14,
`grader_public/`); their inputs (the pair's geometry, pockets and masses, the working area, the payload, the vendors'
numbers, the lots' real capability) are not: they are behind the photos, the build log, the datasheet and erratum
pages, the quote and the lot reports.

## Notes

- Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, MuJoCo 3.14,
  matplotlib, Pillow, PyYAML, jsonschema, trimesh. No network, no OCR engine. The photos and pages are large: process the
  photos programmatically and view only the images you need; read the document pages as images.
- Units: SI in the files (the schemas' unit table lists accepted equivalents such as mm, deg, g, kgf.cm, rpm).
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it, and run `schema_check.py` after each part.
