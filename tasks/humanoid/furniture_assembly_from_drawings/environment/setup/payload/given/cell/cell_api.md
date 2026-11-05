# Cell API (`cell`), version furniture-cell/1.0

Your program `/home/user/submission/assemble.py` exposes `solve(cell) -> None`. `cell` is a proxy
(`furniture_client.client.Cell`, in `~/given/lib/furniture_client/`): every method is one JSON-lines request
`{"id": n, "method": ..., "params": {...}}` answered by `{"id": n, "result": {...}}` or `{"id": n, "error": "..."}`
(the proxy raises `CellError` on an error). Nothing moves between requests: only `step` advances time.

| call | returns | notes |
|---|---|---|
| `describe()` | the static facts: robots, joint names and ranges, control limits, grasp-site definition, grip rule and limits, cameras (intrinsics; the ceiling cameras' exact poses; the head cameras' mount), camera budget, room survey, budgets, safety thresholds | static for the episode; no part poses |
| `get_state()` | `sim_time`, `done`, `robots: {a: {...}, b: {...}}` with `base_pose` [x, y, yaw] (localised, noisy), `stand_height`, `base_moving`, `q` {joint name: angle} for the waist, arms and hands, `hands: {left/right: {closed, tactile_N, grasp_pose_in_robot [x, y, z, qw, qx, qy, qz], ik_residual}}` | see `cell_manual.md` section 6 |
| `command(robot, **targets)` | `{"ok": true, "notes": [...]}` | `robot` is `"a"` or `"b"`; any subset of the targets below; each target is held until replaced |
| `step(n_ticks)` | `sim_time`, `done`, `events` | advances `n_ticks` x 10 ms (1 <= n_ticks <= 500); `events`: only `safety:<what>` and `episode_over` |
| `get_camera(name, depth=True)` | `camera`, `frame_index`, `width`, `height`, `format` (`png`), `rgb_png_b64`, `intrinsics`, `pose_in_room` {`position`, `rotation` (3 x 3, columns = the image's x right, y down, z forward axes in the room)}, `sim_time`, and with depth `sensor_depth_u16_mm_b64`, `depth_unit_m` (0.001), `depth_invalid_value` (0) | `name`: `ceiling_west`, `ceiling_east`, `a/head`, `b/head`; at most 900 frames per episode, then `{"error": "camera_budget"}` |
| `finish()` | `{"ok": true}` | ends the episode; the grader then reads the furniture's state |

`command` targets (cell_manual.md section 5):

| key | value |
|---|---|
| `base_velocity` | [vx, vy, wz] in the robot's heading frame (cancels a base goal) |
| `base_goal` | [x, y, yaw] in the room, or `null` to cancel |
| `stand_height` | metres, 0.48 - 0.80 |
| `waist` | [yaw, roll, pitch] joint targets |
| `left_arm`, `right_arm` | `{"q": [7 joint targets]}` or `{"pose": [x, y, z, qw, qx, qy, qz], "frame": "base" or "room", "path": "line" or "joint"}` (the grasp site's target pose; defaults `"base"`, `"line"`) |
| `left_hand`, `right_hand` | `"open"` or `"close"` |

`furniture_client.client.decode_frame(reply)` returns `(rgb uint8 H x W x 3, depth float32 H x W in metres with NaN where
the sensor reported nothing)`.

Malformed parameters (a wrong type or shape, an unknown key, a number that is not finite or exceeds 1000 in magnitude,
`base_velocity` and `base_goal` in the same `command`) answer `{"error": "bad_params", "detail": "..."}` and change
nothing. Any other method answers `{"error": "forbidden"}`, does nothing and is counted as an illegal call.

## Running a program

* Graded episodes: the grader launches your program itself (see the instruction): the client's `main()` imports
  `assemble` from the current directory and calls `solve(cell)` on a cell connected to the program's stdin / stdout,
  then calls `finish()`. `print()` goes to stderr.
* Practice: `furniture_client.client.connect_practice()` connects to the practice cell, a separate service
  (`/run/furniture-cell/practice.sock`) that runs one fresh practice episode per connection with exactly the graded
  rules and budgets (episodes drawn like the graded ones, never the same), at most 60 practice episodes in your session
  (`cell._call("status")` tells how many are used). Example:

```python
import sys; sys.path.insert(0, "/home/user/given/lib")
from furniture_client.client import connect_practice
sys.path.insert(0, "/home/user/submission")
import assemble
cell = connect_practice()
try:
    assemble.solve(cell)
finally:
    cell.finish()
```

The practice cell answers only the public calls above (and `status`); it never tells you about part poses or
fastenings either. Close the episode with `finish()` (or by closing the connection) before opening the next one.
