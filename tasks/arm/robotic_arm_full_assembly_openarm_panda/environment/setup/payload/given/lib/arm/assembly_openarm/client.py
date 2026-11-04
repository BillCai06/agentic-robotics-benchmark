"""Agent-side proxy of the OpenArm pair's assembly cell + its launcher (PUBLIC CODE; the wire of `arm.assembly_cell`).

    def solve(cell):                                  # in /home/user/submission/assemble.py
        info = cell.describe()                        # robots "left" / "right", cameras, budgets, safety
        f = cell.get_camera("top")                    # also "wrist_left", "wrist_right"
        rgb, depth_m = decode_frame(f)                # uint8 H x W x 3, float32 H x W metres (NaN where invalid)
        st = cell.get_state()                         # st["robots"]["left"]: q, dq, cmd_q, gripper_width, tcp_pos, tcp_quat, wrench, ...
        cell.set_control("left", "cartesian", 0.35)
        cell.command("left", tcp_pos=[...], tcp_quat=[w, x, y, z], width=0.08)
        r = cell.step(20)                             # both Pandas move; r["events"]: only "safety:..." and "episode_over"
        cell.finish()
"""
import os
import sys
import traceback

from ..assembly_cell.client import Cell, CellError, connect_stdio as _connect_stdio
from ..assembly_info.client import decode_frame

__all__ = ["PairCell", "CellError", "decode_frame", "connect_stdio", "main", "BOOTSTRAP", "launcher_argv"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from arm.assembly_openarm.client import main; "
             "sys.exit(main(sys.argv[2:]))")


class PairCell(Cell):
    """the cell with two Pandas: every control call names its robot, "left" or "right"."""

    def set_control(self, robot, mode="joint", stiffness_scale=1.0):
        return self._call("set_control", robot=str(robot), mode=str(mode), stiffness_scale=float(stiffness_scale))

    def command(self, robot, q=None, tcp_pos=None, tcp_quat=None, width=None):
        params = {"robot": str(robot)}
        if q is not None:
            params["q"] = [float(v) for v in q]
        if tcp_pos is not None:
            params["tcp_pos"] = [float(v) for v in tcp_pos]
        if tcp_quat is not None:
            params["tcp_quat"] = [float(v) for v in tcp_quat]
        if width is not None:
            params["width"] = float(width)
        return self._call("command", **params)

    def get_camera(self, name="top", depth=True):
        return self._call("get_camera", name=str(name), depth=bool(depth))


def connect_stdio():
    c = _connect_stdio()
    return PairCell(c._send, c._recv)


def launcher_argv(python, lib_dir, program_dir, module="assemble"):
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    from ..assembly_cell.runner import PYTHON_FLAGS
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import assemble.py from the current directory and run solve(cell)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else "assemble"
    cell = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(cell)
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
    return code


if __name__ == "__main__":
    sys.exit(main())
