#!/usr/bin/env python3
"""The practice lab of furniture_assembly_from_drawings: practice episodes of the furniture cell on a Unix socket.

environment/setup/setup.sh installs this folder at /opt/arb-lab (owner: the lab account `arbcell`, mode 0700, so the
agent can read neither this code, nor the lab's copy of the cell, nor the lab's state) and starts this server as that
account:

    python3 -s -B lab_server.py --socket /run/furniture-cell/practice.sock --allow-user user --log <file>

The server
  * listens on a Unix stream socket (mode 0666; connections from accounts other than root and the --allow-user ones are
    refused, so the program the grader runs as another account cannot reach the lab);
  * speaks the grader's wire, one JSON object per line: {"id": n, "method": m, "params": {...}} ->
    {"id": n, "result": {...}} or {"id": n, "error": "<code>"[, "detail": "<text>"]};
  * serves ONE episode per connection and one episode at a time: the first cell call of a connection starts its
    episode (a fresh CellServer, never privileged, never on the nominal layout, on a 64-bit seed from os.urandom);
    a connection that asks while another episode runs waits for its turn, first come first served;
  * answers "status" at any time with {"episodes_used", "episodes_max"}; status never starts an episode, nor does a
    `finish` sent before one (answered {"ok": true}, then the connection is closed);
  * ends an episode on `finish`, on disconnect, at the simulated-time cap (as the grader's wire does, a few more calls
    are answered first), at the program-time cap (the wall clock minus the cell's simulation and rendering, as the
    grader's wire counts it), after `silence_s` without a request or on a malformed request, and then closes the
    connection;
  * counts the episodes in a persisted counter (<state>/episodes_used.json) and refuses new ones once
    `practice_episodes_max` are used: {"error": "practice_budget"};
  * runs every episode in a forked child: a crash or a hang of an episode costs that episode, never the server. The
    parent imports everything (numpy, mujoco, the cell) once at start-up, so the lab keeps working whatever happens
    to files later, and never renders (no GL context crosses a fork).

What reaches the client is the cell's public replies and short error codes: never a traceback, the seed, a score or
any other hidden state. The server's and the cell's log lines go to the server's log file (the lab account's), stray
output to the daemon's stderr. Episode budgets (sim_cap_s, camera_budget_frames, max_ticks_per_step, settle_s,
program_time_cap_s, silence_s) come from lab_config.json next to this file and equal the grading run's. Stdlib + numpy (+ the cell).
"""

import argparse
import collections
import importlib
import json
import os
import pkgutil
import pwd
import selectors
import signal
import socket
import stat
import struct
import sys
import time
import traceback
import warnings

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)            # furniture_cell: the lab's own copy, next to this file

MAX_LINE = 4 * 1024 * 1024              # the grader's wire limit on one request line
PRE_LINE = 64 * 1024                    # a request before the episode starts (status, the first cell call)
MAX_CONNECTIONS = 16                    # open connections (idle + waiting); more are refused with "busy"
SIM_CAP_GRACE_STEPS = 50                # the grader's wire: step calls answered after the sim cap before the end
REQUEST_CAP_S = 300.0                   # one request (or the cell's build) longer than this is a hang: SIGALRM
BACKSTOP_S = 2 * REQUEST_CAP_S          # the parent kills an episode this long after its longest legitimate run
# the cell's own work (not charged to the program) is bounded by the sim cap and the frame budget; the parent's backstop
# allows this much wall clock for it per simulated second and per frame (several times what a slow machine needs)
WORK_S_PER_SIM_S, WORK_S_PER_FRAME = 1.5, 5.0
SELFTEST_TIMEOUT_S = 50.0
EXIT_OK, EXIT_ERROR, EXIT_BUILD_FAILED = 0, 1, 3
# the grader-only calls of the cell (with the server module's PRIVILEGED_METHODS, added at start-up), refused here
# whatever the CellServer would answer; so is every name starting with "_"
FORBIDDEN = {"truth", "reference_assemble", "score", "finalize"}
ERROR_KEYS = frozenset(("error", "detail"))     # the grader's wire: an answer with only these keys is an error reply

STOP = [False]


class Log(object):
    """One line per event, flushed at once (a forked child inherits no buffered text)."""

    def __init__(self, path=None):
        self.f = sys.stderr
        if path:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self.f = os.fdopen(fd, "a", buffering=1, encoding="utf-8", errors="replace")

    def __call__(self, msg):
        try:
            self.f.write("%s [%d] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), os.getpid(), msg))
            self.f.flush()
        except Exception:  # noqa: BLE001 - logging never takes the lab down
            pass


LOG = Log()


# ------------------------------------------------------------------------------------------------ start-up
def load_config(path):
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("%s is not a JSON object" % path)
    cfg = {k: v for k, v in raw.items() if not k.startswith("_") and k != "note"}
    for key in ("sim_cap_s", "camera_budget_frames", "max_ticks_per_step", "settle_s", "program_time_cap_s", "silence_s",
                "practice_episodes_max"):
        if key not in cfg:
            raise ValueError("%s has no %r" % (path, key))
    return cfg


def import_cell():
    """Import numpy, mujoco and the whole cell package now (every submodule: a lazy import inside the cell then never
    touches the disk again). The server and the cell are required; another submodule that fails is logged."""
    import numpy  # noqa: F401
    import mujoco
    import furniture_cell
    from furniture_cell import cell as _cell  # noqa: F401
    from furniture_cell import server as _server
    from furniture_cell.server import CellServer
    FORBIDDEN.update(str(m) for m in getattr(_server, "PRIVILEGED_METHODS", ()))
    for mod in pkgutil.iter_modules(furniture_cell.__path__):
        name = "furniture_cell." + mod.name
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            LOG("start-up: importing %s failed (only a lazy use of it would need it)\n%s" % (name, traceback.format_exc()))
    LOG("start-up: mujoco %s, cell modules %s" % (mujoco.__version__, sorted(
        m.split(".", 1)[1] for m in sys.modules if m.startswith("furniture_cell."))))
    return CellServer


def new_seed():
    return int.from_bytes(os.urandom(8), "little")


def make_cell(CellServer, seed, cfg):
    return CellServer(seed, privileged=False, nominal=False, config=dict(cfg), log=LOG)


def is_error(result):
    return isinstance(result, dict) and "error" in result and set(result) <= ERROR_KEYS


def finalize(srv):
    fin = getattr(srv, "finalize", None)
    if not callable(fin):
        return None
    try:
        return fin()
    except TypeError:
        return fin(None)


def selftest(CellServer, cfg):
    """Build one cell in a forked child (as an episode would be) and ask it to describe itself and its state."""
    t0 = time.monotonic()
    pid = os.fork()
    if pid == 0:
        code = EXIT_ERROR
        try:
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.alarm(int(SELFTEST_TIMEOUT_S) + 5)
            srv = make_cell(CellServer, new_seed(), cfg)
            for method in ("describe", "get_state"):
                r = srv.handle(method, {})
                if not isinstance(r, dict) or "error" in r:
                    raise RuntimeError("%s answered %r" % (method, r if not isinstance(r, dict) else sorted(r)))
            try:
                finalize(srv)
            except Exception:  # noqa: BLE001 - the summary is not the lab's business
                LOG("self-test: finalize raised (ignored)\n%s" % traceback.format_exc())
            code = EXIT_OK
        except BaseException:  # noqa: BLE001
            LOG("self-test failed\n%s" % traceback.format_exc())
        finally:
            os._exit(code)
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            code = os.waitstatus_to_exitcode(status)
            LOG("self-test: exit %d after %.1f s" % (code, time.monotonic() - t0))
            return code == EXIT_OK
        if time.monotonic() - t0 > SELFTEST_TIMEOUT_S:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            LOG("self-test: no answer within %.0f s" % SELFTEST_TIMEOUT_S)
            return False
        time.sleep(0.1)


class Budget(object):
    """The practice episode counter, persisted (an atomic rename) so a restart of the server does not reset it."""

    def __init__(self, state_dir, maximum):
        self.path = os.path.join(state_dir, "episodes_used.json")
        self.max = int(maximum)
        self.used = self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                return max(0, int(json.load(f)["episodes_used"]))
        except FileNotFoundError:
            return 0
        except Exception:  # noqa: BLE001
            LOG("budget: %s unreadable, no further episodes\n%s" % (self.path, traceback.format_exc()))
            return self.max

    def _save(self):
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"episodes_used": self.used, "episodes_max": self.max}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except OSError:
            LOG("budget: cannot persist the counter\n%s" % traceback.format_exc())

    def available(self):
        return self.used < self.max

    def take(self):
        self.used += 1
        self._save()

    def refund(self):
        self.used = max(0, self.used - 1)
        self._save()

    def status(self):
        return {"episodes_used": int(self.used), "episodes_max": int(self.max)}


def make_listener(path):
    """Bind under a temporary name, open it to every account, then move it into place: the socket never exists under
    its public name with any other mode."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"                                     # (a socket path holds at most 107 bytes)
    for p in (tmp, path):
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            continue
        if not (stat.S_ISSOCK(st.st_mode) or stat.S_ISLNK(st.st_mode)):
            raise RuntimeError("%s exists and is not a socket" % p)
        os.unlink(p)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(tmp)
    os.chmod(tmp, 0o666)
    s.listen(MAX_CONNECTIONS)
    os.rename(tmp, path)
    s.setblocking(False)
    return s


def peer_uid(sock):
    try:
        _pid, uid, _gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                              struct.calcsize("3i")))
        return uid
    except OSError:
        return None


def encode(obj):
    return (json.dumps(obj) + "\n").encode("utf-8")


# ------------------------------------------------------------------------------------------------ the episode (child)
class LineTooLong(Exception):
    pass


class LineReader(object):
    def __init__(self, sock, initial=b""):
        self.sock = sock
        self.buf = bytearray(initial)
        self.eof = False

    def readline(self, timeout):
        """The next line (str), None at end of stream; raises TimeoutError, LineTooLong, OSError."""
        deadline = time.monotonic() + timeout
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line = bytes(self.buf[:i])
                del self.buf[:i + 1]
                if len(line) > MAX_LINE:
                    raise LineTooLong()
                return line.decode("utf-8", "replace")
            if len(self.buf) > MAX_LINE:
                raise LineTooLong()
            if self.eof:
                if self.buf:
                    line = bytes(self.buf)
                    self.buf.clear()
                    return line.decode("utf-8", "replace")
                return None
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError()
            self.sock.settimeout(left)
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                self.eof = True
            else:
                self.buf += chunk


def parse_request(line):
    """(id, method, params) as the grader's wire reads a request, or None for a line that is not one."""
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    if not isinstance(obj, dict) or "method" not in obj:
        return None
    params = obj.get("params") or {}
    return obj.get("id"), str(obj.get("method")), params if isinstance(params, dict) else {}


def run_episode(conn, pending, seed, number, cfg, status, CellServer):
    """Serve one episode on `conn` (blocking). Returns the child's exit code."""
    silence = float(cfg["silence_s"])
    program_cap = float(cfg["program_time_cap_s"])
    reader = LineReader(conn, pending)
    t_build = time.monotonic()

    def send(obj):
        try:
            data = encode(obj)
        except (TypeError, ValueError):
            LOG("episode %d: a reply is not JSON\n%s" % (number, traceback.format_exc()))
            data = encode({"id": obj.get("id"), "error": "internal_error"})
        try:
            conn.settimeout(silence)
            conn.sendall(data)
            return True
        except OSError:
            return False

    signal.signal(signal.SIGALRM, signal.SIG_DFL)          # the watchdog: a hung request ends the process
    signal.alarm(int(REQUEST_CAP_S))
    try:
        srv = make_cell(CellServer, seed, cfg)
    except Exception:  # noqa: BLE001
        signal.alarm(0)
        LOG("episode %d: the cell could not be built (seed %d)\n%s" % (number, seed, traceback.format_exc()))
        try:
            line = reader.readline(0.5)
            req = parse_request(line) if line is not None else None
        except Exception:  # noqa: BLE001
            req = None
        send({"id": req[0] if req else None, "error": "internal_error"})
        return EXIT_BUILD_FAILED
    signal.alarm(0)
    # the episode's clock starts once the cell stands, as the grader's wire starts it once the program runs
    t0_wall, t0 = time.time(), time.monotonic()
    build_s = t0 - t_build
    srv.wall_start = t0_wall

    def program_time():                                 # the grader's wire: the wall clock minus the cell's own work
        return (time.monotonic() - t0) - float(getattr(srv, "work_wall", 0.0))

    n_req, reason = 0, None
    while True:
        remaining = program_cap - program_time()
        if remaining <= 0:
            reason = "program_time_cap"
            break
        try:
            line = reader.readline(min(silence, remaining))
        except TimeoutError:
            reason = "program_time_cap" if program_time() >= program_cap else "silence"
            break
        except LineTooLong:
            send({"id": None, "error": "protocol"})
            reason = "protocol"
            break
        except OSError:
            reason = "eof"
            break
        if line is None:
            reason = "eof"
            break
        if not line.strip():
            continue
        req = parse_request(line)
        if req is None:
            send({"id": None, "error": "protocol"})
            reason = "protocol"
            break
        rid, method, params = req
        n_req += 1
        if method == "status":
            reply = {"id": rid, "result": dict(status)}
        elif method.startswith("_") or method in FORBIDDEN:
            reply = {"id": rid, "error": "forbidden"}
        else:
            signal.alarm(int(REQUEST_CAP_S))
            try:
                result = srv.handle(method, params)
            except Exception:  # noqa: BLE001
                LOG("episode %d: %s raised\n%s" % (number, method, traceback.format_exc()))
                result = {"error": "internal_error"}
            finally:
                signal.alarm(0)
            if is_error(result):
                reply = {"id": rid, "error": result["error"]}
                if "detail" in result:
                    reply["detail"] = result["detail"]
            else:
                reply = {"id": rid, "result": result}
        if not send(reply):
            reason = "eof"
            break
        if srv.done and method == "finish":
            reason = "finish"
            break
        if srv.done and srv.done_reason == "sim_cap" and method == "step":
            srv.steps_after_done = getattr(srv, "steps_after_done", 0) + 1
            if srv.steps_after_done > SIM_CAP_GRACE_STEPS:
                reason = "sim_cap"
                break
    srv.wall_used = time.time() - t0_wall
    srv.program_used = program_time()
    try:
        conn.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()
    rec = {}
    signal.alarm(int(REQUEST_CAP_S))
    try:
        if not srv.done:
            srv._end(reason)
        rec = finalize(srv) or {}                           # frees the renderers; the record stays in this process
    except Exception:  # noqa: BLE001
        LOG("episode %d: finalize raised\n%s" % (number, traceback.format_exc()))
    signal.alarm(0)
    LOG("episode %d: end %s after %d requests; build %.1f s, wall %.0f s, sim %.1f s, frames %s, steps %s, "
        "bad params %s, illegal %s, internal errors %s" % (
            number, reason, n_req, build_s, time.monotonic() - t0, float(rec.get("sim_time_s", float("nan"))),
            rec.get("camera_frames_total"), rec.get("step_calls"), rec.get("bad_param_calls"),
            rec.get("illegal_call_count"), rec.get("internal_errors")))
    return EXIT_OK


# ------------------------------------------------------------------------------------------------ the server (parent)
class Conn(object):
    def __init__(self, sock, number, uid):
        self.sock = sock
        self.number = number
        self.uid = uid
        self.buf = b""
        self.pending = b""
        self.last = time.monotonic()


class Lab(object):
    def __init__(self, args, cfg, CellServer):
        self.cfg = cfg
        self.CellServer = CellServer
        self.silence = float(cfg["silence_s"])
        # the longest legitimate episode: the program's time cap plus the cell's own work at its bounds
        self.episode_wall_max = (float(cfg["program_time_cap_s"]) + WORK_S_PER_SIM_S * float(cfg["sim_cap_s"])
                         + WORK_S_PER_FRAME * float(cfg["camera_budget_frames"]))
        self.budget = Budget(args.state, cfg["practice_episodes_max"])
        self.allowed = None
        if args.allow_user:
            self.allowed = {0}
            for name in args.allow_user:
                try:
                    self.allowed.add(pwd.getpwnam(name).pw_uid)
                except KeyError:
                    LOG("start-up: no account %r to allow" % name)
        self.path = args.socket
        self.sel = selectors.DefaultSelector()
        self.listener = make_listener(self.path)
        self.sel.register(self.listener, selectors.EVENT_READ, None)
        self.idle = {}                       # connections read by the server (no episode asked yet)
        self.waiting = collections.deque()   # connections that asked for an episode, in order
        self.active = None                   # the running episode: {"pid", "t0", "number", "conn"}
        self.n_conn = 0

    # -- connections before their episode
    def accept(self):
        while True:
            try:
                sock, _ = self.listener.accept()
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                LOG("accept failed\n%s" % traceback.format_exc())
                return
            self.n_conn += 1
            uid = peer_uid(sock)
            if self.allowed is not None and uid not in self.allowed:
                LOG("connection %d from uid %s refused" % (self.n_conn, uid))
                self._refuse(sock, "forbidden")
                continue
            if len(self.idle) + len(self.waiting) + (1 if self.active else 0) >= MAX_CONNECTIONS:
                LOG("connection %d refused: %d open" % (self.n_conn, MAX_CONNECTIONS))
                self._refuse(sock, "busy")
                continue
            sock.setblocking(False)
            c = Conn(sock, self.n_conn, uid)
            self.idle[sock.fileno()] = c
            self.sel.register(sock, selectors.EVENT_READ, c)
            LOG("connection %d from uid %s" % (c.number, uid))

    @staticmethod
    def _refuse(sock, code):
        try:
            sock.settimeout(1.0)
            sock.sendall(encode({"id": None, "error": code}))
        except OSError:
            pass
        sock.close()

    def _reply(self, c, obj):
        try:
            c.sock.settimeout(5.0)
            c.sock.sendall(encode(obj))
            return True
        except OSError:
            return False
        finally:
            try:
                c.sock.setblocking(False)
            except OSError:
                pass

    def close(self, c, why):
        if c.sock.fileno() in self.idle:
            del self.idle[c.sock.fileno()]
            try:
                self.sel.unregister(c.sock)
            except (KeyError, ValueError):
                pass
        try:
            c.sock.close()
        except OSError:
            pass
        LOG("connection %d closed: %s" % (c.number, why))

    def read(self, c):
        try:
            data = c.sock.recv(1 << 16)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self.close(c, "error")
            return
        if not data:
            self.close(c, "disconnected")
            return
        c.last = time.monotonic()
        c.buf += data
        self.lines(c)

    def lines(self, c):
        """Answer the complete requests of an idle connection: status at once; the first other call queues it."""
        while True:
            i = c.buf.find(b"\n")
            if i < 0:
                if len(c.buf) > PRE_LINE:
                    self._reply(c, {"id": None, "error": "protocol"})
                    self.close(c, "request too long")
                return
            line, rest = c.buf[:i], c.buf[i + 1:]
            if not line.strip():
                c.buf = rest
                continue
            req = parse_request(line.decode("utf-8", "replace"))
            if req is None:
                self._reply(c, {"id": None, "error": "protocol"})
                self.close(c, "protocol error")
                return
            rid, method, _params = req
            if method == "status":
                c.buf = rest
                if not self._reply(c, {"id": rid, "result": self.budget.status()}):
                    self.close(c, "disconnected")
                    return
                continue
            if method == "finish":                           # nothing to end: no episode is spent on it
                self._reply(c, {"id": rid, "result": {"ok": True}})
                self.close(c, "finish before any episode")
                return
            if not self.budget.available():
                c.buf = rest
                if not self._reply(c, {"id": rid, "error": "practice_budget"}):
                    self.close(c, "disconnected")
                    return
                continue
            c.pending, c.buf = c.buf, b""                     # this request and everything after it
            del self.idle[c.sock.fileno()]
            self.sel.unregister(c.sock)
            self.waiting.append(c)
            LOG("connection %d waits for an episode (%d waiting)" % (c.number, len(self.waiting)))
            return

    # -- episodes
    def start_next(self):
        while self.active is None and self.waiting:
            c = self.waiting.popleft()
            try:
                peek = c.sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                gone = peek == b""
            except (BlockingIOError, InterruptedError):
                gone = False
            except OSError:
                gone = True
            if gone:
                try:
                    c.sock.close()
                except OSError:
                    pass
                LOG("connection %d left before its episode" % c.number)
                continue
            if not self.budget.available():
                req = parse_request(c.pending.split(b"\n", 1)[0].decode("utf-8", "replace"))
                c.buf = c.pending.split(b"\n", 1)[1] if b"\n" in c.pending else b""
                c.pending = b""
                self.idle[c.sock.fileno()] = c
                self.sel.register(c.sock, selectors.EVENT_READ, c)
                if not self._reply(c, {"id": req[0] if req else None, "error": "practice_budget"}):
                    self.close(c, "disconnected")
                    continue
                c.last = time.monotonic()
                self.lines(c)
                continue
            self.budget.take()
            number, seed, status = self.budget.used, new_seed(), self.budget.status()
            pid = os.fork()
            if pid == 0:
                self._child(c, seed, number, status)       # never returns
            c.sock.close()
            self.active = {"pid": pid, "t0": time.monotonic(), "number": number, "conn": c.number}
            LOG("episode %d (connection %d, uid %s): pid %d, seed %d; %d/%d used" % (
                number, c.number, c.uid, pid, seed, self.budget.used, self.budget.max))

    def _child(self, c, seed, number, status):
        code = EXIT_ERROR
        try:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_DFL)
            self.listener.close()
            for other in list(self.idle.values()) + list(self.waiting):
                try:
                    other.sock.close()
                except OSError:
                    pass
            self.sel.close()
            c.sock.setblocking(True)
            code = run_episode(c.sock, c.pending, seed, number, self.cfg, status, self.CellServer)
        except BaseException:  # noqa: BLE001
            LOG("episode %d: the episode process failed\n%s" % (number, traceback.format_exc()))
        finally:
            os._exit(code)

    def reap(self):
        a = self.active
        if a is None:
            return
        try:
            pid, status = os.waitpid(a["pid"], os.WNOHANG)
        except ChildProcessError:
            pid, status = a["pid"], 0
        if pid == 0:
            if time.monotonic() - a["t0"] > self.episode_wall_max + BACKSTOP_S and not a.get("killed"):
                LOG("episode %d: past its longest legitimate run, killed" % a["number"])
                try:
                    os.kill(a["pid"], signal.SIGKILL)
                except OSError:
                    pass
                a["killed"] = True
            return
        code = os.waitstatus_to_exitcode(status)
        refund = code == EXIT_BUILD_FAILED or code == -signal.SIGALRM
        if refund:
            self.budget.refund()
        LOG("episode %d: process exit %d after %.0f s%s" % (
            a["number"], code, time.monotonic() - a["t0"],
            " (a request hung)" if code == -signal.SIGALRM else "") + ("; refunded" if refund else ""))
        self.active = None

    def expire_idle(self):
        now = time.monotonic()
        for c in list(self.idle.values()):
            if now - c.last > self.silence:
                self.close(c, "silent")

    def serve(self):
        LOG("serving %s; %d/%d practice episodes used" % (self.path, self.budget.used, self.budget.max))
        while not STOP[0]:
            try:
                for key, _ev in self.sel.select(0.5):
                    if key.data is None:
                        self.accept()
                    elif key.fileobj.fileno() in self.idle:
                        self.read(key.data)
                self.reap()
                self.start_next()
                self.expire_idle()
            except Exception:  # noqa: BLE001 - the lab outlives any single connection
                LOG("server loop\n%s" % traceback.format_exc())
                time.sleep(0.5)
        self.shutdown()

    def shutdown(self):
        LOG("shutting down")
        a = self.active
        if a is not None:
            try:
                os.kill(a["pid"], signal.SIGTERM)
                for _ in range(50):
                    if os.waitpid(a["pid"], os.WNOHANG)[0]:
                        break
                    time.sleep(0.1)
                else:
                    os.kill(a["pid"], signal.SIGKILL)
                    os.waitpid(a["pid"], 0)
            except OSError:
                pass
        for c in list(self.idle.values()) + list(self.waiting):
            try:
                c.sock.close()
            except OSError:
                pass
        try:
            self.listener.close()
            os.unlink(self.path)
        except OSError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--socket", default="/run/furniture-cell/practice.sock")
    ap.add_argument("--config", default=os.path.join(HERE, "lab_config.json"))
    ap.add_argument("--state", default=os.path.join(HERE, "state"))
    ap.add_argument("--log", default=None, help="the server's log file (default: stderr)")
    ap.add_argument("--allow-user", action="append", default=[],
                    help="an account allowed to connect besides root (repeatable; none given: every account)")
    ap.add_argument("--no-selftest", action="store_true")
    args = ap.parse_args(argv)
    os.umask(0o077)
    global LOG
    LOG = Log(args.log)
    warnings.filterwarnings("ignore", message=".*fork.*", category=DeprecationWarning)
    os.makedirs(args.state, exist_ok=True)
    os.chdir(args.state)                                    # MuJoCo writes MUJOCO_LOG.TXT into the working directory
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: STOP.__setitem__(0, True))
    try:
        cfg = load_config(args.config)
        CellServer = import_cell()
    except Exception:  # noqa: BLE001
        LOG("start-up failed\n%s" % traceback.format_exc())
        return 2
    if not args.no_selftest and not selftest(CellServer, cfg):
        return 2
    try:
        lab = Lab(args, cfg, CellServer)
    except Exception:  # noqa: BLE001
        LOG("start-up failed\n%s" % traceback.format_exc())
        return 2
    lab.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
