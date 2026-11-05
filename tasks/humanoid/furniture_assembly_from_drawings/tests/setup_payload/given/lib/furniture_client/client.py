"""Program-side proxy of the furniture cell, and the launcher that runs a submission against it.

    from furniture_client import decode_frame

    def solve(cell):                                    # in /home/user/submission/assemble.py
        info = cell.describe()                          # robots, joints, control limits, cameras, room, budgets
        f = cell.get_camera("ceiling_west")             # one RGB-D frame (counts against the frame budget)
        rgb, depth_m = decode_frame(f)                  # uint8 H x W x 3, float32 H x W metres (NaN = no reading)
        st = cell.get_state()                           # both robots' own state (localised base pose, joints, hands)
        cell.command("a", base_goal=[1.9, 0.9, 1.57])   # robot "a" or "b"; see the cell API document
        cell.command("b", left_arm={"pose": [0.35, 0.2, 0.9, 1, 0, 0, 0], "frame": "base"}, left_hand="open")
        r = cell.step(50)                               # 50 x 10 ms of physics; r["sim_time"], r["done"], r["events"]
        cell.finish()                                   # ends the episode (the launcher also calls it)

Every method is one JSON-lines request ({"id": n, "method": ..., "params": {...}}) answered by one reply line
({"id": n, "result": ...} or {"id": n, "error": code, "detail": text}). A reply carrying an error raises CellError
(its `code` is e.g. "bad_params", "forbidden", "camera_budget"; a "forbidden" call is counted against the program).

Two transports, one protocol:
  * connect_stdio(): the graded run. The launcher (`main`) starts the program with the cell on its own stdin / stdout;
    it moves this process's stdout (Python's and the C-level file descriptor 1) to stderr first, so prints never
    corrupt the wire.
  * connect_practice(path): the practice lab's cell over a Unix stream socket ($ARB_PRACTICE_SOCKET, default
    /run/furniture-cell/practice.sock), for developing the program against the same physics.

numpy-typed arguments are accepted (sent as plain numbers / lists).
"""

import base64
import io
import json
import os
import sys
import traceback

__all__ = ["Cell", "CellError", "decode_frame", "connect_stdio", "connect_practice", "given_dir", "main", "BOOTSTRAP",
           "PUBLIC_CAMERAS", "PRACTICE_SOCKET"]

# the launcher the graded run starts: `python -B -E -s -c BOOTSTRAP <library dir> assemble`
BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from furniture_client.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PUBLIC_CAMERAS = ("ceiling_west", "ceiling_east", "a/head", "b/head")
PRACTICE_SOCKET = "/run/furniture-cell/practice.sock"


class CellError(RuntimeError):
    """An error reply of the cell (or a broken connection). `code`: the error code; `detail`: the cell's explanation."""

    def __init__(self, code, detail=None):
        self.code = str(code)
        self.detail = detail
        RuntimeError.__init__(self, self.code if not detail else "%s: %s" % (self.code, detail))


def _jsonable(obj):
    """json.dumps fallback for numpy scalars and arrays."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError("%r is not JSON serialisable" % (obj,))


class Cell(object):
    """The proxy. Units: metres, radians, seconds; quaternions w, x, y, z; the room frame of the cell API document."""

    def __init__(self, send, recv, close=None):
        self._send = send
        self._recv = recv
        self._close = close
        self._id = 0

    def _call(self, method, **params):
        """One request; returns the reply's result or raises CellError."""
        self._id += 1
        req = {"id": self._id, "method": str(method), "params": params}
        try:
            self._send(json.dumps(req, default=_jsonable) + "\n")
            line = self._recv()
        except OSError as e:                     # e.g. BrokenPipeError: the cell ended the episode and hung up
            raise CellError("closed", "the connection to the cell broke (%s)" % e.__class__.__name__)
        if not line:
            raise CellError("closed", "the cell closed the connection")
        try:
            reply = json.loads(line)
        except ValueError:
            raise CellError("bad_reply", line[:120])
        if not isinstance(reply, dict) or reply.get("id") != self._id:
            raise CellError("bad_reply", "reply id mismatch")
        if "error" in reply:
            raise CellError(reply["error"], reply.get("detail"))
        return reply.get("result")

    # ---- information
    def describe(self):
        """The cell's published facts: robots, joint names and ranges, control limits, cameras (intrinsics and poses),
        the room survey, budgets and safety limits."""
        return self._call("describe")

    def get_state(self):
        """sim_time, done, and per robot: base_pose [x, y, yaw] (localised, noisy), stand_height, base_moving, joint
        positions q, hands {closed, tactile_N, grasp_pose_in_robot, ik_residual}."""
        return self._call("get_state")

    def get_camera(self, name, depth=True):
        """One frame of `name` (ceiling_west, ceiling_east, a/head, b/head): PNG rgb + 16-bit PNG depth in mm (0 = no
        reading), intrinsics, the camera's pose in the room at capture time. Use decode_frame() on the reply."""
        return self._call("get_camera", name=str(name), depth=bool(depth))

    # ---- motion
    def command(self, robot, **targets):
        """Set targets of robot "a" or "b" (held until changed): base_velocity [vx, vy, wz] or base_goal [x, y, yaw]
        (None stops a goal), stand_height, waist [yaw, roll, pitch], left_arm / right_arm {"q": [7]} or
        {"pose": [x, y, z, qw, qx, qy, qz], "frame": "base" | "room", "path": "line" | "joint"}, left_hand /
        right_hand "open" | "close". Returns {"ok": True, "notes": [...]} (notes name clamped values)."""
        return self._call("command", robot=str(robot), **targets)

    def step(self, n_ticks=1):
        """Advance the physics n_ticks x 10 ms (1 .. max_ticks_per_step); returns {sim_time, done, events}."""
        return self._call("step", n_ticks=int(n_ticks))

    def finish(self):
        """End the episode (the work is judged as it stands)."""
        return self._call("finish")

    def close(self):
        """Close the connection (practice lab); the graded run's launcher handles its own."""
        if self._close is not None:
            try:
                self._close()
            finally:
                self._close = None


def decode_frame(reply):
    """(rgb uint8 H x W x 3, depth float32 H x W in metres with NaN where the sensor gave no reading) of a get_camera
    reply; depth is None when the frame was taken without depth."""
    import numpy as np
    from PIL import Image
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["rgb_png_b64"]))).convert("RGB"))
    depth = None
    if reply.get("sensor_depth_u16_mm_b64"):
        d = np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["sensor_depth_u16_mm_b64"]))), dtype=np.float32)
        depth = d * float(reply.get("depth_unit_m", 0.001))
        depth[d == float(reply.get("depth_invalid_value", 0))] = np.nan
    return rgb, depth


def given_dir():
    """The folder of the task's given material: $ARB_GIVEN_DIR in the graded run (a read-only copy), else ~/given."""
    return os.environ.get("ARB_GIVEN_DIR") or os.path.join(os.path.expanduser("~"), "given")


def connect_stdio():
    """The cell on this process's original stdin / stdout. From then on file descriptor 1 (and Python's sys.stdout)
    write to stderr and descriptor 0 reads nothing, so prints, C extensions and child processes cannot corrupt the
    wire."""
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    inp = os.fdopen(os.dup(0), "r", encoding="utf-8")
    os.dup2(2, 1)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    sys.stdout = sys.stderr

    def send(text):
        out.write(text)
        out.flush()

    def recv():
        return inp.readline()

    return Cell(send, recv)


def connect_practice(path=None, timeout=None):
    """The practice lab's cell over a Unix stream socket: `path` (default $ARB_PRACTICE_SOCKET or
    /run/furniture-cell/practice.sock). `timeout` (s, optional) bounds every socket operation. Same protocol and
    methods as the graded cell; close() the returned Cell when done."""
    import socket
    if path is None:
        path = os.environ.get("ARB_PRACTICE_SOCKET", PRACTICE_SOCKET)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        if timeout is not None:
            s.settimeout(float(timeout))
        s.connect(path)
    except OSError as e:
        s.close()
        raise CellError("unreachable", "cannot reach the practice cell at %s: %s" % (path, e))
    rf = s.makefile("r", encoding="utf-8", newline="\n")
    wf = s.makefile("w", encoding="utf-8", newline="\n")

    def send(text):
        wf.write(text)
        wf.flush()

    def recv():
        return rf.readline()

    def close():
        for f in (wf, rf):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
        s.close()

    return Cell(send, recv, close=close)


def main(argv=None):
    """Launcher of the graded run: import `assemble` (or argv[0]) from the current directory, call its solve(cell)
    with the cell on the wire, and always end the episode (finish) when solve returns or raises. Returns the exit
    code: 0 ok, 1 an exception in the program, 2 a cell error."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else "assemble"
    cell = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        solve = getattr(mod, "solve")
        solve(cell)
    except CellError as e:
        print("[assemble] cell error: %s" % e, file=sys.stderr)
        code = 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        try:
            cell.finish()
        except Exception:  # noqa: BLE001
            pass
        try:
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
