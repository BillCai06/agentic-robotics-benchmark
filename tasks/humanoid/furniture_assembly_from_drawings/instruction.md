# Assemble a flat-pack bookcase from its drawings with two Unitree G1 humanoids

You are the robot integrator of a furniture-assembly service. A customer has bought the **B930 bookcase**; it has
been delivered and unpacked in their room, and two Unitree G1 humanoids (29 DoF, Dex3-1 hands) are on site. Your job:
write the program that makes the two robots build the bookcase **exactly as its drawings show** and stand it **where
the drawings' floor plan says**, using only what the robots can sense: their head cameras, two ceiling cameras of the
room, their joint encoders, their localisation and their hands' tactile load. Several steps are marked for two
persons; one humanoid alone cannot hold a tall board upright and push another board into it at the same time.

Nothing gives the answer directly: no part list with poses, no part identities, no assembly graph, no "seated" signal.
Which board is which, which way round it goes, in what order, and where each one ends up are in the drawings and in
the camera images. The parts lie in different places and orientations in every episode.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once
per hidden episode (**3 episodes**) in its own physics cell; the program talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`). Per episode: 1500 s of simulated time; 3000 s of program time, which is the episode's wall
clock minus the time the cell spends simulating (inside `step`) and rendering (inside `get_camera`), so it measures your
program's own computation and its other requests; at most 120 s from a reply to your next request; 900 camera frames
in total.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = a copy of `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble`
(or `bash run.sh`). Resolve given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write
only under the run's HOME / TMPDIR. A program that crashes ends that episode with what it achieved; the other episodes
still run. Your submission must be final when your session ends: the submission folder is then copied out of your
sandbox and graded in a fresh sandbox that holds nothing else of yours (no other file, no process you started).

## What you are given

| path | content |
|---|---|
| `~/given/drawings/sheet_00_cover.png` .. `sheet_09_floor_plan.png` | the manufacturer's drawing set of the B930: cover with the overall dimensions and the finishes, parts list, dimensioned part drawings, five wordless assembly steps, and the floor plan with the bookcase's place |
| `~/given/cell/cell_manual.md` | the cell: the room survey (exact), how the flat-pack was delivered, the robots and their controllers, the grip rule, the sensors and cameras, the fastening rules, safety rules, budgets |
| `~/given/cell/cell_api.md` | the cell's programming interface and how to run a program against the practice cell |
| `~/given/robot/unitree_g1/` | the robot's MuJoCo model (Menagerie, BSD-3-Clause); the cell's modifications are in the manual |
| `~/given/lib/furniture_client/` | the cell's client (`Cell`, `connect_practice`, `decode_frame`, the launcher) |

**Practice cell.** A practice service runs the real cell for you (one fresh practice episode per connection, drawn
like the graded ones, never the same; at most 60 practice episodes in this session): see `cell_api.md`. It answers the
same public calls and nothing else. There is no answer key and no checker: you judge your program's work from the
cameras, like on site.

Tools: Python 3.12 at **`/opt/venv/bin/python3`** (use this path; the plain `python3` on your PATH may lack the
packages) with numpy, scipy, OpenCV (`cv2`, headless), Pillow, matplotlib and MuJoCo 3.16. No network in your session.

## How you are graded (3 hidden episodes)

Gates (multiply the score): `contract_ok` = `assemble.py` is a regular file that parses and defines `solve`, imports
under the run regime above, makes at least one `step` call over the episodes, causes no network activity, and your own
`.py` files import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`,
`subprocess`, ...), load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a
non-literal) and start no other program; and the submission holds at most 60 MB in all, no weight file (`.pt`, `.pth`,
`.onnx`, `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`) and no `.py` or `.sh` file whose text contains
`torch.optim`, `.backward(` or `jax.grad` anywhere (comments and strings included). Per episode `safety`
(cell_manual.md section 8): a violation zeroes that episode.

`assembly_score` = the mean over the episodes of the bookcase's credited weight / 9:
* each of the six dowel joints of the carcass (the top, the fixed shelf and the bottom in each side panel): 1.0 when the
  right board is locked in the right group of holes the right way round, as the drawings show;
* the back panel locked in the rabbets of both side panels the right way round: 1.5;
* the bookcase in its place, 1.5: all six dowel joints credited and the carcass standing where the floor plan puts it
  (within 30 mm and 3 deg, upright within 3 deg).

The cell never tells your program whether anything fastened; it reads the furniture's state when the episode ends.

## Notes

- Images (drawings, camera frames) are meant to be processed by your program; look at the ones you need.
- The room survey, the camera calibrations, the robots' kinematics and joint angles are exact; the localisation is
  slightly noisy; everything about the parts must be measured in each episode.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is
  lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment.
  Write large files in several parts, keep every single response well below that limit, write a first complete
  submission early and refine it.
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that limit
  is not graded at all, so a late finish scores nothing. Note the start time with `date` when you begin, keep a complete
  submission in place from early on, and end the session yourself (stop working and say that you are done) at least
  15 minutes before the limit.
