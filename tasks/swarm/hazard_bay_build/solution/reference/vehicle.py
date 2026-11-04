"""The reference ground vehicle's program. Its kinematics come from the scene it was built in (`scene.usda` beside
it): the joints' frames read with pxr, forward kinematics, a damped least-squares IK on the grip point (with the
tool's axis held, where it matters).

Two programs:

- `Privileged`, the reference's own: with the true state in each observation (`privileged`, which the grader adds
  for the reference's token only), it does the whole job on the day's draws: the valve pushed shut round the lever's
  true hub (ending as soon as the lever reads shut, backing off along the arc), then every part off the tables and
  into the crate (a stand at a table's edge per part, chosen so the arm reaches it from there, the vehicle stays
  outside the burning pan's ring and has room to turn at its approach; a path on a grid of the bay, driven backward
  when the way lies behind; a pick from above with the suction tool, checked from where the vehicle really stands; a
  drop into a free slot of the crate, let go only once the part is over the crate), then it parks clear of everything
  with its arm folded. It turns on the spot only where there is room: the skid-steer turns about a point some way off
  its root body, so a spin walks the vehicle.
- `Baseline`, for practice runs without that block: it closes the valve on its odometry and parks (the part routine
  is kept but switched off: on raw wheel odometry the vehicle drifts by tens of centimetres over the minutes the parts
  take and runs into the tables; doing the station without the true state needs localisation against what the
  cameras see, which is the agent's job).
"""
import math
import os
import sys

import numpy as np

# the bay, as the reference built it (world frame)
BAY_PARK = (-2.80, 2.20)
LEVER_STEM = np.array([-1.20, 3.20, 0.95])  # the valve's stem; the lever's grip is 0.22 m up it, closing swings it to +x
GRIP_ABOVE_STEM = 0.1955  # [m] where the cameras report the lever's grip, above the stem, with the lever open
LOOK_HEADING = 60.0  # [deg] the heading at the valve that puts the lever in the mast camera's view (it looks 62 deg right)
VALVE_STAND = (-1.20, 2.62)
CRATE = np.array([0.10, 1.50])
CRATE_TURN, CRATE_STAND = (0.10, 0.55), (0.10, 0.92)
WEST_LANE, SOUTH_LANE = -2.35, -0.55  # the routes round Table A (west of it) and along the tables' south side
TABLE_A_ITEMS = ("cracker_box", "mustard_bottle", "sugar_box")
TABLE_B_ITEMS = ("soup_can", "power_drill", "mug")
SPEED, TURN = 0.6, 1.2  # [m/s] cruise, [rad/s] the largest turn rate
ARM_RATE = 0.9  # [rad/s] the largest joint rate the arm targets move at
CLEAR_STATION = False  # (see the docstring)


def log(*a):
    print("[vehicle]", *a, file=sys.stderr, flush=True)
    if os.environ.get("ARB_REFERENCE_LOG"):  # (a maintainer's record of a run: the grader discards stderr)
        try:
            with open(os.environ["ARB_REFERENCE_LOG"], "a") as f:
                print("[vehicle]", *a, file=f)
        except OSError:
            pass


# --------------------------------------------------------------------------------------------- kinematics
def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_mat(q)
    T[:3, 3] = p
    return T


def axis_rot(axis, a):
    k = np.asarray(axis, dtype=float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return T


class Kinematics:
    """The vehicle's chain from its root body to the grip link, read from the scene's joints."""

    def __init__(self, scene, root_body, grip_link):
        from pxr import Usd, UsdPhysics
        st = Usd.Stage.Open(scene)
        edges = []
        for p in Usd.PrimRange(st.GetPrimAtPath("/World/Fleet/UGV"), Usd.TraverseInstanceProxies()):
            if not (p.IsA(UsdPhysics.Joint) and p.IsActive()):
                continue
            j = UsdPhysics.Joint(p)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if not b0 or not b1:
                continue

            def q4(v):
                return (v.GetReal(), *v.GetImaginary()) if v is not None else (1.0, 0.0, 0.0, 0.0)
            J0 = tf(j.GetLocalPos0Attr().Get() or (0, 0, 0), q4(j.GetLocalRot0Attr().Get()))
            J1 = tf(j.GetLocalPos1Attr().Get() or (0, 0, 0), q4(j.GetLocalRot1Attr().Get()))
            rev = p.IsA(UsdPhysics.RevoluteJoint)
            axis = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[p.GetAttribute("physics:axis").Get() or "X"] if rev else None
            edges.append(dict(name=p.GetName(), b0=b0[0], b1=b1[0], J0=J0, J1=J1, axis=axis))
        bodies = {e["b0"] for e in edges} | {e["b1"] for e in edges}
        # the root: the articulation root's body (an arm asset may carry a body of the same name)
        arts = [p.GetPath().pathString for p in Usd.PrimRange(st.GetPrimAtPath("/World/Fleet/UGV"), Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        root = next((b for b in bodies if b in arts), None) or next(b for b in sorted(bodies, key=len) if b.endswith("/" + root_body))
        grip = next(b for b in bodies if b.endswith("/" + grip_link))
        # the path root -> grip through the joint graph
        prev = {root: None}
        frontier = [root]
        while frontier and grip not in prev:
            nxt = []
            for b in frontier:
                for e in edges:
                    for a, c, fwd in ((e["b0"], e["b1"], True), (e["b1"], e["b0"], False)):
                        if a == b and c not in prev:
                            prev[c] = (b, e, fwd)
                            nxt.append(c)
            frontier = nxt
        chain = []
        b = grip
        while prev[b] is not None:
            a, e, fwd = prev[b]
            chain.append((e, fwd))
            b = a
        self.chain = chain[::-1]
        for e, _ in self.chain:
            e["J0i"], e["J1i"] = np.linalg.inv(e["J0"]), np.linalg.inv(e["J1"])
        self.joints = [e["name"] for e, _ in self.chain if e["axis"] is not None]
        # the wheels: revolute joints off the root that are not on the arm's chain
        self.wheels = {}
        for e in edges:
            if e["axis"] is not None and e["b0"] == root and e["name"] not in self.joints:
                T = e["J0"]
                axis_root = T[:3, :3] @ np.array(e["axis"], dtype=float)
                self.wheels[e["name"]] = dict(pos=T[:3, 3].copy(), axis=axis_root)
        p, w = self.points({})
        self.wrist_len = float(np.linalg.norm(p - w))  # (the grip point lies on the last joint's axis: a constant)

    def frames(self, q):
        """The grip link's frame in the root's frame and, for every revolute joint on the chain, its origin and its
        (signed) axis, for joint positions q (a dict by name)."""
        T = np.eye(4)
        joints = []
        for e, fwd in self.chain:
            a = q.get(e["name"], 0.0) if e["axis"] is not None else 0.0
            if fwd:
                Tj = T @ e["J0"]
                if e["axis"] is not None:
                    joints.append((Tj[:3, 3].copy(), Tj[:3, :3] @ np.asarray(e["axis"], dtype=float)))
                    T = Tj @ axis_rot(e["axis"], a) @ e["J1i"]
                else:
                    T = Tj @ e["J1i"]
            else:
                Tj = T @ e["J1"]
                if e["axis"] is not None:
                    joints.append((Tj[:3, 3].copy(), -(Tj[:3, :3] @ np.asarray(e["axis"], dtype=float))))
                    T = Tj @ np.linalg.inv(axis_rot(e["axis"], a)) @ e["J0i"]
                else:
                    T = Tj @ e["J0i"]
        return T, joints

    def points(self, q):
        """(the grip point, the last revolute joint's origin: the wrist) in the root's frame."""
        T, js = self.frames(q)
        return T[:3, 3].copy(), js[-1][0] if js else T[:3, 3].copy()

    def solve(self, target, q0, limits, wrist_dir=None, posture=None, iters=120, tol=0.002, wtol=0.006):
        """Damped least squares on the grip point and, with `wrist_dir` (a unit vector in the root's frame), on the
        wrist, held `wrist_len` from the grip point along that direction (the tool's axis); a light pull toward
        `posture` in the null space. Returns (q, grip residual, wrist residual)."""
        names = self.joints
        q = np.array([q0.get(n, 0.0) for n in names], dtype=float)
        lo = np.array([(limits.get(n) or [None, None])[0] if (limits.get(n) or [None, None])[0] is not None else -2 * math.pi for n in names])
        hi = np.array([(limits.get(n) or [None, None])[1] if (limits.get(n) or [None, None])[1] is not None else 2 * math.pi for n in names])
        lo, hi = np.maximum(lo, -2 * math.pi) + 1e-3, np.minimum(hi, 2 * math.pi) - 1e-3
        q = np.clip(q, lo, hi)
        pq = np.array([posture.get(n, 0.0) for n in names]) if posture else None
        target = np.asarray(target, dtype=float)
        wt = target + self.wrist_len * np.asarray(wrist_dir, dtype=float) if wrist_dir is not None else None
        best = (q.copy(), 1e9, 1e9)
        for _ in range(iters):
            T, js = self.frames(dict(zip(names, q)))
            p = T[:3, 3]
            e = target - p
            err = float(np.linalg.norm(e))
            werr = 0.0
            J = np.array([np.cross(ax, p - o) for o, ax in js]).T
            if wt is not None:
                w = js[-1][0]
                ew = wt - w
                werr = float(np.linalg.norm(ew))
                Jw = np.array([np.cross(ax, w - o) for o, ax in js]).T
                J = np.vstack([J, Jw])
                e = np.concatenate([e, ew])
            if err + 0.5 * werr < best[1] + 0.5 * best[2]:
                best = (q.copy(), err, werr)
            if err < tol and werr < wtol:
                break
            lam = 0.03
            JJ = J @ J.T + lam * lam * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJ, e)
            if pq is not None:
                N = np.eye(len(names)) - J.T @ np.linalg.solve(JJ, J)
                dq = dq + 0.15 * N @ (pq - q)
            dq = np.clip(dq, -0.2, 0.2)
            q = np.clip(q + dq, lo, hi)
        return dict(zip(names, best[0])), best[1], best[2]

    def fk(self, q):
        """The grip point in the root's frame for joint positions q (a dict by name)."""
        T = np.eye(4)
        for e, fwd in self.chain:
            a = q.get(e["name"], 0.0) if e["axis"] is not None else 0.0
            R = axis_rot(e["axis"], a) if e["axis"] is not None else np.eye(4)
            if fwd:
                T = T @ e["J0"] @ R @ np.linalg.inv(e["J1"])
            else:
                T = T @ e["J1"] @ np.linalg.inv(R) @ np.linalg.inv(e["J0"])
        return T[:3, 3].copy()

    def ik(self, target, q0, limits, iters=150, posture=None):
        """Damped least squares on the grip point (position only), within the joints' limits; a light pull toward
        `posture` in the null space. Returns (q, residual)."""
        names = self.joints
        q = np.array([q0.get(n, 0.0) for n in names], dtype=float)
        lo = np.array([limits.get(n, [-1e3, 1e3])[0] if limits.get(n, [None])[0] is not None else -1e3 for n in names])
        hi = np.array([limits.get(n, [-1e3, 1e3])[1] if limits.get(n, [None, None])[1] is not None else 1e3 for n in names])
        lo, hi = np.maximum(lo, -2 * math.pi), np.minimum(hi, 2 * math.pi)
        pq = np.array([posture.get(n, 0.0) for n in names]) if posture else None
        best = (q.copy(), 1e9)
        for _ in range(iters):
            qd = dict(zip(names, q))
            p = self.fk(qd)
            e = np.asarray(target) - p
            err = float(np.linalg.norm(e))
            if err < best[1]:
                best = (q.copy(), err)
            if err < 0.003:
                break
            J = np.zeros((3, len(names)))
            for i in range(len(names)):
                qd2 = dict(qd)
                qd2[names[i]] += 1e-4
                J[:, i] = (self.fk(qd2) - p) / 1e-4
            lam = 0.05
            JJ = J @ J.T + lam * lam * np.eye(3)
            dq = J.T @ np.linalg.solve(JJ, e)
            if pq is not None:
                N = np.eye(len(names)) - J.T @ np.linalg.solve(JJ, J)
                dq = dq + 0.1 * N @ (pq - q)
            dq = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + dq, lo, hi)
        return dict(zip(names, best[0])), best[1]


# --------------------------------------------------------------------------------------------- the program
class Baseline:
    """The vehicle's baseline program (no true state). `state` is a plain dict (the controller deep-copies it for
    repeated queries)."""

    def __init__(self):
        self.kin = None
        self.mis = None

    def setup(self, mis):
        self.mis = mis
        u = mis.get("ugv")
        if not u:
            return
        scene = os.path.join(os.getcwd(), "scene.usda")
        try:
            self.kin = Kinematics(scene, "base_link", u["grip_link"])
            log("arm joints", self.kin.joints, "wheels", {k: [round(float(x), 3) for x in v["pos"]] for k, v in self.kin.wheels.items()})
        except Exception as exc:  # noqa: BLE001
            log("no kinematics:", exc)
            self.kin = None
        self.limits = u.get("limits") or {}
        if self.kin and self.kin.wheels:
            ws = list(self.kin.wheels.values())
            self.track = float(np.mean([abs(w["pos"][1]) for w in ws])) * 2
            self.radius = 0.098

    def init_state(self):
        return dict(step=0, steps=self.plan(), sub=None, t_step=0.0, q_tgt=None, hold=None, done=False, lever_done_t=None,
                    log=[], grip=False, items_left=list(TABLE_A_ITEMS + TABLE_B_ITEMS))

    def plan(self):
        """The task list: (kind, args)."""
        st = []
        # the valve: east along the bay's lane, north to the lever, push it closed
        st += [("go", (BAY_PARK[0], 2.20, None)), ("go", (VALVE_STAND[0], 2.20, None)), ("face", 0.0),
               ("go", (VALVE_STAND[0], VALVE_STAND[1], 0.0)), ("face", LOOK_HEADING), ("look_lever", None), ("face", 0.0),
               ("lever", None), ("arm_home", None),
               ("go", (VALVE_STAND[0], 2.20, 0.0)), ("mark", "valve")]
        if not CLEAR_STATION:  # back to the bay and stay there
            st += [("go", (BAY_PARK[0], 2.20, None)), ("face", 0.0), ("arm_home", None), ("mark", "done")]
            return st
        # Table A: round its west end to the south lane; each part: in from the south, pick, out, east, north to the crate
        st += [("go", (WEST_LANE, 2.20, None)), ("go", (WEST_LANE, SOUTH_LANE, None))]
        for name in TABLE_A_ITEMS:
            st += [("part", name)]
        st += [("wait_hoist", None)]
        for name in TABLE_B_ITEMS:
            st += [("part", name)]
        st += [("arm_home", None), ("mark", "done")]
        return st

    # -- helpers ----------------------------------------------------------------------------------------------
    def wheels(self, v, w):
        """Wheel speed targets for a forward speed v [m/s] and a turn rate w [rad/s] (skid steering)."""
        out = {}
        for name, wh in self.kin.wheels.items():
            side = 1.0 if wh["pos"][1] > 0 else -1.0  # (left: +y in the root's frame)
            sign = 1.0 if wh["axis"][1] > 0 else -1.0
            out[name] = sign * (v - side * w * self.track * 0.75) / self.radius
        return out

    @staticmethod
    def pose(obs):
        x, y, yaw = obs["ugv"]["pose"]
        return np.array([x, y]), math.radians(yaw)

    def to_root(self, obs, p_world):
        """A world point in the root's frame (the root's +x is the vehicle's forward: heading + 90 degrees)."""
        xy, yaw = self.pose(obs)
        a = yaw + math.pi / 2
        d = np.asarray(p_world[:2]) - xy
        c, s = math.cos(-a), math.sin(-a)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], p_world[2] - self.root_z])

    def drive_to(self, obs, x, y, yaw_end):
        """One step toward (x, y) and then the heading yaw_end (degrees, None: as arrived). Returns (cmd, done)."""
        xy, yaw = self.pose(obs)
        d = np.array([x, y]) - xy
        dist = float(np.linalg.norm(d))
        if dist > 0.04:
            want = math.atan2(d[1], d[0]) - math.pi / 2
            back = False
            err = (want - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > math.radians(100) and dist < 0.8:  # (short moves straight back: reverse instead of turning)
                back = True
                err = (want + math.pi - yaw + math.pi) % (2 * math.pi) - math.pi
            w = float(np.clip(2.0 * err, -TURN, TURN))
            if abs(err) > math.radians(25):  # (turn in place, never slower than skid steering can turn)
                w = math.copysign(max(abs(w), 0.5), err)
                v = 0.0
            else:  # (drive and steer)
                v = min(SPEED, 0.9 * dist + 0.08) * math.cos(err)
            return self.wheels(-v if back else v, w), False
        if yaw_end is not None:
            err = (math.radians(yaw_end) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(err) > math.radians(3.0):
                w = float(np.clip(2.0 * err, -TURN, TURN))
                return self.wheels(0.0, math.copysign(max(abs(w), 0.35), err)), False
        return self.wheels(0.0, 0.0), True

    def arm_toward(self, s, obs, q_goal, rate=ARM_RATE):
        """Move the arm's targets toward q_goal at a bounded rate; True when the joints are there."""
        dt = self.mis["dt"]
        joints = obs["ugv"]["joints"]
        if s["q_tgt"] is None:
            s["q_tgt"] = {n: joints[n][0] for n in self.kin.joints}
        for n in self.kin.joints:
            d = q_goal[n] - s["q_tgt"][n]
            s["q_tgt"][n] += float(np.clip(d, -rate * dt, rate * dt))
        # there: the targets have arrived at the goal and the joints have followed them
        there = all(abs(q_goal[n] - s["q_tgt"][n]) < 1e-6 and abs(q_goal[n] - joints[n][0]) < 0.03 for n in self.kin.joints)
        return there

    # -- one step -------------------------------------------------------------------------------------------------
    def act(self, s, obs, hoist_phase):
        if self.kin is None or "ugv" not in obs:
            return {}
        g = obs["ugv"]
        dt = self.mis["dt"]
        if s["step"] == 0 and s["t_step"] == 0.0 and s.get("home") is None:
            s["home"] = {n: g["joints"][n][0] for n in self.kin.joints}
            self.root_z = self.mis["ugv"]["start"][2]
            p = self.kin.fk(s["home"])
            log("fk check: grip point", [round(float(v), 3) for v in p], "observed", g["grip_point"])
        if not hasattr(self, "root_z"):
            self.root_z = self.mis["ugv"]["start"][2]
        s["t_step"] += dt
        cmd = {"vel": {}, "pos": {}, "grip": s["grip"]}
        if s["step"] >= len(s["steps"]) or g.get("disabled"):
            cmd["vel"] = self.wheels(0.0, 0.0)
            if s["q_tgt"]:
                cmd["pos"] = dict(s["q_tgt"])
            return cmd
        kind, arg = s["steps"][s["step"]]
        done = False
        wheels = self.wheels(0.0, 0.0)
        if kind == "go":
            wheels, done = self.drive_to(obs, *arg)
        elif kind == "face":
            wheels, done = self.drive_to(obs, *self.pose(obs)[0], arg)
        elif kind == "mark":
            done = True
        elif kind == "arm_home":
            done = self.arm_toward(s, obs, s["home"]) or s["t_step"] > 12.0
        elif kind == "wait_hoist":
            done = hoist_phase in ("RECOVER",) or s["t_step"] > 150.0
        elif kind == "look_lever":  # the lever's grip as the camera sees it, kept in the odometry's frame
            seen = [d["rel"] for d in obs["ugv"].get("sees") or [] if d["what"] == "lever"]
            if seen:
                xy = self.pose(obs)[0]
                g = np.array([xy[0] + seen[0][0], xy[1] + seen[0][1], self.root_z + seen[0][2]])
                s.setdefault("lever_seen", []).append(g.tolist())
            done = len(s.get("lever_seen", [])) >= 30 or s["t_step"] > 3.0
            if done:
                if s.get("lever_seen"):
                    g = np.mean(np.array(s["lever_seen"]), axis=0)
                    s["stem"] = [float(g[0]), float(g[1]), float(g[2] - GRIP_ABOVE_STEM)]
                    log(f"lever seen: stem at {np.round(s['stem'], 3).tolist()} (odometry frame)")
                else:
                    log("lever not seen: pushing where the map says it is")
        elif kind == "lever":
            done = self.lever(s, obs)
            if done and s.get("lever_done_t") is None:
                s["lever_done_t"] = obs["time"]
                log(f"lever pushed round at t={obs['time']:.1f}")
        elif kind == "part":
            wheels, done = self.part(s, obs, arg)
        if s["q_tgt"]:
            cmd["pos"] = dict(s["q_tgt"])
        cmd["vel"] = wheels
        cmd["grip"] = s["grip"]
        if done:
            xy, yaw = self.pose(obs)
            log(f"t={obs['time']:.1f} done {kind} {arg} at ({xy[0]:.2f}, {xy[1]:.2f}) heading {math.degrees(yaw):.0f}")
            s["step"] += 1
            s["t_step"] = 0.0
            s["sub"] = None
        return cmd

    def lever(self, s, obs):
        """Push the lever round its quarter turn: the grip point along an arc 0.17 m from the stem, from just on the
        lever's -x side to past its closed stop."""
        if s["sub"] is None:
            s["sub"] = dict(k=0, q=None)
        sub = s["sub"]
        n_way = 12
        if sub["k"] > n_way + 2:
            return True
        a = math.radians(-22.0 + (100.0 + 22.0) * min(sub["k"], n_way) / n_way)
        r = 0.24  # (at the lever's grip end: the open and closed stops sit close round the stem)
        stem = np.array(s["stem"]) if s.get("stem") else LEVER_STEM
        p = stem + np.array([r * math.sin(a), 0.0, r * math.cos(a)])
        if sub["k"] == 0:
            p = p + np.array([-0.04, -0.12, 0.0])  # (first in front of it, then into the lever's plane)
        if sub["k"] == n_way + 1:  # (then straight back out of the lever's plane, away from the line ...)
            p = stem + np.array([r * math.sin(a), -0.25, r * math.cos(a)])
        if sub["k"] == n_way + 2:  # (... and up, before the arm folds)
            p = stem + np.array([0.0, -0.30, 0.35])
        if sub["q"] is None:
            q0 = s["q_tgt"] or {n: obs["ugv"]["joints"][n][0] for n in self.kin.joints}
            sub["q"], res = self.kin.ik(self.to_root(obs, p), q0, self.limits)
            if res > 0.03:
                log(f"lever waypoint {sub['k']}: IK residual {res:.3f} m")
        there = self.arm_toward(s, obs, sub["q"], rate=0.6)
        if there or s["t_step"] > 6.0:
            log(f"t={obs['time']:.1f} lever waypoint {sub['k']} {'reached' if there else 'timed out'}; lever seen at "
                f"{next((d['rel'] for d in obs['ugv'].get('sees') or [] if d['what'] == 'lever'), None)}")
            sub["k"] += 1
            sub["q"] = None
            s["t_step"] = 0.0
        return False

    def part(self, s, obs, name):
        """One part from its table to the crate: in from the south lane, pick it from above, back out, east along the
        lane, north to the crate, drop it in, back to the lane."""
        if s["sub"] is None:
            items = self.mis.get("items") or []
            if name not in items:
                return self.wheels(0.0, 0.0), True
            pos = None
            for d in obs["ugv"].get("sees") or []:
                if d["what"] == f"item:{name}":
                    pos = self.pose(obs)[0].tolist() + [0.0]
                    pos = np.array([pos[0] + d["rel"][0], pos[1] + d["rel"][1], self.root_z + d["rel"][2]])
            if pos is None:
                pos = np.array(ITEMS_AT.get(name, (0.0, 0.0, 0.8)))
            table_south = 0.40 if name in TABLE_A_ITEMS else 0.13
            top = pos[2] + ITEM_HALF_H.get(name, 0.08)
            s["sub"] = dict(k=0, item=pos.tolist(), top=top, x=float(pos[0]), stand_y=table_south - 0.32, q=None, t=0.0)
        sub = s["sub"]
        x, sy = sub["x"], sub["stand_y"]
        legs = [("go", (x, SOUTH_LANE, 90.0 - 90.0)), ("go", (x, sy, 0.0)),
                ("arm", "above"), ("arm", "touch"), ("grip", True), ("arm", "lift"), ("arm", "carry"),
                ("go", (x, SOUTH_LANE, None)), ("go", (CRATE_TURN[0], SOUTH_LANE, None)), ("go", (CRATE_TURN[0], CRATE_TURN[1], 0.0)),
                ("go", (CRATE_STAND[0], CRATE_STAND[1], 0.0)), ("arm", "crate"), ("grip", False), ("arm", "carry"),
                ("go", (CRATE_TURN[0], CRATE_TURN[1], None)), ("go", (CRATE_TURN[0], SOUTH_LANE, None))]
        if sub["k"] >= len(legs):
            return self.wheels(0.0, 0.0), True
        kind, arg = legs[sub["k"]]
        sub["tl"] = sub.get("tl", 0.0) + self.mis["dt"]
        wheels, done = self.wheels(0.0, 0.0), False
        if kind == "go":
            wheels, done = self.drive_to(obs, *arg)
        elif kind == "grip":
            s["grip"] = arg
            sub["t"] += self.mis["dt"]
            done = sub["t"] > 0.5
            if arg and obs["ugv"].get("holding") is None and 0.4 < sub["t"] <= 0.4 + self.mis["dt"]:
                xy, yaw = self.pose(obs)
                gp = np.array(obs["ugv"]["grip_point"])
                a = yaw + math.pi / 2
                gw = np.array([xy[0] + math.cos(a) * gp[0] - math.sin(a) * gp[1], xy[1] + math.sin(a) * gp[0] + math.cos(a) * gp[1], self.root_z + gp[2]])
                log(f"{name}: nothing held after the touch; tool at {np.round(gw, 3).tolist()}, part top at "
                    f"{np.round([sub['item'][0], sub['item'][1], sub['top']], 3).tolist()}")
        elif kind == "arm":
            if sub["q"] is None:
                item = np.array(sub["item"])
                if arg == "above":
                    p = np.array([item[0], item[1], sub["top"] + 0.12])
                elif arg == "touch":
                    p = np.array([item[0], item[1], sub["top"] + 0.01])
                elif arg == "lift":
                    p = np.array([item[0], item[1], sub["top"] + 0.30])
                elif arg == "carry":
                    xy, yaw = self.pose(obs)
                    fwd = np.array([-math.sin(yaw), math.cos(yaw)])
                    p = np.array([xy[0] + 0.30 * fwd[0], xy[1] + 0.30 * fwd[1], 1.15])
                else:  # crate
                    p = np.array([CRATE[0], CRATE[1] - 0.05, 0.62])
                q0 = s["q_tgt"] or {n: obs["ugv"]["joints"][n][0] for n in self.kin.joints}
                sub["q"], res = self.kin.ik(self.to_root(obs, p), q0, self.limits, posture=s["home"])
                if res > 0.03:
                    log(f"{name} {arg}: IK residual {res:.3f} m")
            done = self.arm_toward(s, obs, sub["q"]) or s["t_step"] > 8.0
            if done:
                sub["q"] = None
        if done:
            log(f"t={obs['time']:.1f} {name} leg {sub['k']} {kind} {arg if kind != 'go' else tuple(round(v, 2) if v is not None else None for v in arg)} "
                f"done in {sub['tl']:.1f} s")
            sub["k"] += 1
            sub["t"] = 0.0
            sub["tl"] = 0.0
            s["t_step"] = 0.0
        return wheels, False


# where the reference put the parts (its own scene: the program may know what it built), their half heights
ITEMS_AT = {"cracker_box": (-1.62, 0.62, 0.856), "mustard_bottle": (-1.18, 0.80, 0.847), "sugar_box": (-0.90, 0.58, 0.837),
            "soup_can": (1.30, 0.50, 0.501), "power_drill": (1.62, 0.66, 0.546), "mug": (1.42, 0.34, 0.493)}
ITEM_HALF_H = {"cracker_box": 0.107, "mustard_bottle": 0.096, "sugar_box": 0.088, "soup_can": 0.051, "power_drill": 0.093, "mug": 0.059}


# ============================================================================================= the privileged program
# the real bay as the reference built it (the reference's twin is the real bay; what moves per day comes from the block)
TANK_RECT = ((-1.90, -3.10), (0.80, 0.60), 0.0)
LINE_RECTS = [((0.0, 3.36), (7.40, 0.20), 0.0),  # the steam line: pipe, supports and feet, the valve's branch
              ((-0.60, 3.26), (0.08, 0.12), 0.0)]  # the gauge's stub
CRATE_RIBS = 0.025  # [m] the handle ribs stand out of the crate's long sides
FOOT = (0.51, 0.43)  # [m] the vehicle's footprint (length along its heading, width), the arm tucked above it
INFLATE = 0.45  # [m] the planner keeps the vehicle's centre this far from everything (its half-diagonal, and room for a
                # spin, which walks the root body: the skid-steer turns about a point some way off it)
RING = 2.30  # [m] the vehicle's centre keeps out of this circle round the burning pan (the rule: 2 m)
STAND = 0.37  # [m] the vehicle's centre outside a table's (or the crate's) edge when it works there
APPROACH = 0.28  # [m] further out: where it turns to face the edge before it drives in straight
GRID = (-4.0, 4.0, -0.70, 3.10, 0.05)  # the planner's grid: x0, x1, y0, y1, cell [m]
TUCK = (0.05, 0.0, 0.78)  # [m] the tool's point, the arm tucked above the deck (root frame); holding a part: CARRY
CARRY = (0.10, 0.0, 0.86)
V_DRIVE, V_SLOW, W_TURN = 0.7, 0.3, 0.8  # [m/s] cruise, the last straight stretch; [rad/s] turn rate (a skid-steer's spin walks)
ARM_FAST, ARM_SLOW = 1.0, 0.5  # [rad/s] joint target rates: free moves, moves near the parts and the lever
LEVER_R, LEVER_STEP = 0.24, math.radians(4.0)  # the push's radius round the hub; its arc step
SLOTS = [(sx, sy) for sy in (-0.09, 0.09) for sx in (-0.18, 0.0, 0.18)]  # the crate's 3 x 2 drop slots (crate frame)
SPIN_ROOM = 0.58  # [m] the vehicle spins on the spot only this far from everything (its half-diagonal and the spin's walk)
PARK = (-0.50, 2.05)  # where the vehicle stands still at the end: clear of the tables, the crate, the line and the pan
PRE_ABOVE, TOUCH = 0.16, 0.005  # [m] above a part's top: where the arm comes in; where the tool touches (the suction holds at 4 cm)


def rect_dist(px, py, c, size, yaw):
    """Distance from points to an oriented rectangle (0 inside)."""
    ca, sa = math.cos(yaw), math.sin(yaw)
    dx, dy = px - c[0], py - c[1]
    u, v = dx * ca + dy * sa, -dx * sa + dy * ca
    ox, oy = np.maximum(np.abs(u) - size[0] / 2, 0.0), np.maximum(np.abs(v) - size[1] / 2, 0.0)
    return np.hypot(ox, oy)


def rect_corners(c, size, yaw):
    ca, sa = math.cos(yaw), math.sin(yaw)
    out = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        u, v = sx * size[0] / 2, sy * size[1] / 2
        out.append((c[0] + u * ca - v * sa, c[1] + u * sa + v * ca))
    return np.array(out)


def rects_overlap(a, b, margin=0.0):
    """Do two oriented rectangles (corner arrays) come closer than `margin`? (separating axes)"""
    for poly in (a, b):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)
            pa, pb = a @ n, b @ n
            if pb.min() - pa.max() >= margin or pa.min() - pb.max() >= margin:
                return False
    return True


def heading_of(d):
    """The heading (0 facing +y, counter-clockwise) that faces the direction d."""
    return math.atan2(d[1], d[0]) - math.pi / 2


def wrap_a(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Planner:
    """A grid of the bay: the static obstacles (tables, crate, tank, steam line) and the day's (the pan, the burning
    pan's ring, the bucket while it hangs low), inflated; A* and a string-pulled path."""

    def __init__(self, tables, crate):
        x0, x1, y0, y1, r = GRID
        self.r, self.x0, self.y0 = r, x0, y0
        self.nx, self.ny = int(round((x1 - x0) / r)) + 1, int(round((y1 - y0) / r)) + 1
        xs, ys = x0 + r * np.arange(self.nx), y0 + r * np.arange(self.ny)
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")
        self.rects = [((t["center"][0], t["center"][1]), (t["size"][0], t["size"][1]), math.radians(t["yaw_deg"])) for t in tables]
        lo, hi = np.array(crate[0]), np.array(crate[1])
        self.crate = (((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2), (hi[0] - lo[0], hi[1] - lo[1] + 2 * CRATE_RIBS), 0.0)
        self.rects += [self.crate, TANK_RECT] + LINE_RECTS
        occ = np.zeros((self.nx, self.ny), dtype=bool)
        for c, size, yaw in self.rects:
            occ |= rect_dist(self.X, self.Y, c, size, yaw) < INFLATE
        self.static = occ

    def clearance(self, p, pan=None):
        """How far the point is from the nearest obstacle (the tables, the crate, the tank, the steam line, the pan)."""
        d = min(float(rect_dist(np.array([p[0]]), np.array([p[1]]), c, size, yaw)[0]) for c, size, yaw in self.rects)
        if pan is not None:
            d = min(d, math.hypot(p[0] - pan[0], p[1] - pan[1]) - 0.30)
        return d

    def occupancy(self, pan, fire, bucket):
        occ = self.static | (np.hypot(self.X - pan[0], self.Y - pan[1]) < 0.30 + INFLATE)
        if fire:
            occ |= np.hypot(self.X - pan[0], self.Y - pan[1]) < RING
        if bucket is not None:
            occ |= np.hypot(self.X - bucket[0], self.Y - bucket[1]) < 1.0
        return occ

    def cell(self, p):
        return (int(round((p[0] - self.x0) / self.r)), int(round((p[1] - self.y0) / self.r)))

    def inside(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny

    def free(self, occ, p):
        c = self.cell(p)
        return self.inside(c) and not occ[c]

    def line_free(self, occ, a, b):
        n = max(2, int(math.dist(a, b) / (self.r / 2)) + 1)
        for k in range(n + 1):
            f = k / n
            if not self.free(occ, (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))):
                return False
        return True

    def nearest_free(self, occ, p, rmax=0.6):
        c0 = self.cell(p)
        best = None
        k = int(rmax / self.r)
        for i in range(c0[0] - k, c0[0] + k + 1):
            for j in range(c0[1] - k, c0[1] + k + 1):
                if self.inside((i, j)) and not occ[i, j]:
                    d = (i - c0[0]) ** 2 + (j - c0[1]) ** 2
                    if best is None or d < best[0]:
                        best = (d, (i, j))
        return None if best is None else best[1]

    def plan(self, occ, a, b):
        """A path from a to b (both points), as a list of points, or None."""
        import heapq
        sa = self.cell(a) if self.free(occ, a) else self.nearest_free(occ, a)
        sb = self.cell(b) if self.free(occ, b) else None
        if sa is None or sb is None:
            return None
        h = lambda c: math.hypot(c[0] - sb[0], c[1] - sb[1])  # noqa: E731
        openq, g, prev = [(h(sa), 0.0, sa)], {sa: 0.0}, {sa: None}
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while openq:
            _, gc, c = heapq.heappop(openq)
            if c == sb:
                break
            if gc > g.get(c, 1e18) + 1e-9:
                continue
            for dx, dy, w in steps:
                nb = (c[0] + dx, c[1] + dy)
                if not self.inside(nb) or occ[nb]:
                    continue
                ng = gc + w
                if ng < g.get(nb, 1e18):
                    g[nb], prev[nb] = ng, c
                    heapq.heappush(openq, (ng + h(nb), ng, nb))
        if sb not in prev:
            return None
        cells = []
        c = sb
        while c is not None:
            cells.append(c)
            c = prev[c]
        pts = [(self.x0 + i * self.r, self.y0 + j * self.r) for i, j in reversed(cells)]
        pts[0], pts[-1] = tuple(a[:2]), tuple(b[:2])
        out = [pts[0]]  # (string pulling)
        k = 0
        while k < len(pts) - 1:
            j = len(pts) - 1
            while j > k + 1 and not self.line_free(occ, pts[k], pts[j]):
                j -= 1
            out.append(pts[j])
            k = j
        return [list(p) for p in out]

    def length(self, path):
        return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)) if path else 1e9


class Privileged:
    """The reference's program on the true state (obs['privileged']). `s` is a plain dict (the controller deep-copies
    it for repeated queries); every step is a dict in s['steps'] that keeps its own progress."""

    def __init__(self, kin, mis, wheels):
        self.kin, self.mis, self.wheels = kin, mis, wheels
        self.limits = (mis.get("ugv") or {}).get("limits") or {}
        self.dt = mis["dt"]
        self.planner = None

    def init_state(self):
        return dict(phase="start", steps=[], q_tgt=None, grip=False, home=None, lever_done_t=None, done=False, slots=[],
                    tries={}, gave_up=[], t=0.0, ik_seed=None, log_t=-1.0)

    # -- the true state -------------------------------------------------------------------------------------------
    def setup_world(self, pv):
        if self.planner is None:
            self.planner = Planner(pv["bay"]["tables"], pv["bay"]["crate"])

    @staticmethod
    def pose(pv):
        u = pv["ugv"]
        return np.array(u["pos"][:2]), math.radians(u["heading_deg"]), np.array(u["pos"]), quat_mat(u["quat"])

    def to_root(self, pv, p_world):
        _, _, p, R = self.pose(pv)
        return R.T @ (np.asarray(p_world, dtype=float) - p)

    def dir_root(self, pv, d_world):
        return self.pose(pv)[3].T @ np.asarray(d_world, dtype=float)

    def occ(self, pv):
        b = pv["bucket"]["pos"]
        return self.planner.occupancy(pv["pan"], pv["fire"], b[:2] if b[2] > 0.3 else None)

    def joints(self, obs):
        return {n: obs["ugv"]["joints"][n][0] for n in self.kin.joints}

    # -- arm paths -----------------------------------------------------------------------------------------------
    def reach(self, s, pv, obs, p, wrist):
        """The arm's best configuration for the grip point at p (world) with the tool's axis `wrist`, from several
        starting guesses (the current targets, the start, the arm straight up): (q, grip residual, wrist residual)."""
        pr, wr = self.to_root(pv, p), self.dir_root(pv, wrist)
        seeds = [dict(s["q_tgt"] or self.joints(obs)), dict(s["home"]), {n: 0.0 for n in self.kin.joints}]
        best = None
        for q0 in seeds:
            q, res, wres = self.kin.solve(pr, q0, self.limits, wrist_dir=wr, posture=s["home"], iters=200)
            if best is None or res + 0.5 * wres < best[1] + 0.5 * best[2]:
                best = (q, res, wres)
        return best

    def joint_line(self, q0, q1, n=8):
        """Joint-space waypoints from q0 to q1."""
        return [{k: q0[k] + (f / n) * (q1[k] - q0[k]) for k in self.kin.joints} for f in range(1, n + 1)]

    def line_path(self, s, pv, obs, p1, wrist1, p0=None, wrist0=None, step=0.03, root=None):
        """Joint waypoints that carry the grip point along a straight line (world) to p1, the tool's axis turning from
        wrist0 to wrist1 (world unit vectors, the wrist's side of the grip point). `root` = (p, R) of a planned pose."""
        if root is None:
            to_r = lambda x: self.to_root(pv, x)  # noqa: E731
            dir_r = lambda d: self.dir_root(pv, d)  # noqa: E731
        else:
            rp, rR = root
            to_r = lambda x: rR.T @ (np.asarray(x, dtype=float) - rp)  # noqa: E731
            dir_r = lambda d: rR.T @ np.asarray(d, dtype=float)  # noqa: E731
        q = dict(s["q_tgt"] or self.joints(obs))
        if p0 is None:
            tool_r, wr = self.kin.points(q)
            p0r = tool_r
            w0r = (wr - tool_r) / (np.linalg.norm(wr - tool_r) + 1e-9)
        else:
            p0r, w0r = to_r(p0), dir_r(wrist0 if wrist0 is not None else wrist1)
        p1r, w1r = to_r(p1), dir_r(wrist1)
        n = max(1, int(math.ceil(float(np.linalg.norm(p1r - p0r)) / step)))
        out, worst = [], 0.0
        for k in range(1, n + 1):
            f = k / n
            w = (1 - f) * w0r + f * w1r
            w = w / (np.linalg.norm(w) + 1e-9)
            q, res, wres = self.kin.solve(p0r + f * (p1r - p0r), q, self.limits, wrist_dir=w, posture=s["home"], iters=60)
            worst = max(worst, res)
            out.append(dict(q))
        return out, worst

    def joint_path(self, s, obs, q_goal):
        return [dict(q_goal)]

    # -- the program ---------------------------------------------------------------------------------------------
    def act(self, s, obs, hoist):
        pv = obs["privileged"]
        self.setup_world(pv)
        s["t"] = obs["time"]
        if s["home"] is None:
            s["home"] = self.joints(obs)
            s["q_tgt"] = dict(s["home"])
        if pv["ugv"].get("lost"):
            return dict(vel=self.wheels(0.0, 0.0), pos=dict(s["q_tgt"]), grip=False)
        if not s["steps"]:
            self.plan_next(s, pv, obs, hoist)
        if os.environ.get("ARB_REFERENCE_LOG") and obs["time"] - s["log_t"] >= 2.0:
            s["log_t"] = obs["time"]
            xy, psi, _, _ = self.pose(pv)
            st0 = s["steps"][0] if s["steps"] else {}
            log(f"status t={obs['time']:.1f} {st0.get('kind')} {st0.get('why', '')} phase={st0.get('phase')} at ({xy[0]:.2f}, {xy[1]:.2f}) "
                f"heading {math.degrees(psi):.1f} lever {pv['lever']['deg']:.1f} holding {obs['ugv'].get('holding')} grip {s['grip']}")
        wheels = self.wheels(0.0, 0.0)
        for _ in range(4):  # (a step that is done at once lets the next one act in the same control step)
            if not s["steps"]:
                break
            st = s["steps"][0]
            st.setdefault("t0", obs["time"])
            wheels, done = self.run_step(st, s, pv, obs, hoist)
            if not done and obs["time"] - st["t0"] > st.get("timeout", 30.0):
                log(f"t={obs['time']:.1f} step {st['kind']} timed out ({st.get('why', '')})")
                done = True
                if st.get("critical"):
                    s["steps"] = []
                    s.setdefault("aborts", 0)
                    s["aborts"] += 1
                    break
            if not done:
                break
            log(f"t={obs['time']:.1f} {st['kind']} {st.get('why', '')} done in {obs['time'] - st['t0']:.1f} s")
            s["steps"].pop(0)
            wheels = self.wheels(0.0, 0.0)
        return dict(vel=wheels, pos=dict(s["q_tgt"]), grip=bool(s["grip"]))

    def plan_next(self, s, pv, obs, hoist):
        if s["phase"] == "start":
            s["phase"] = "valve"
            s["steps"] = self.valve_steps(s, pv, obs)
            return
        if s["phase"] == "valve":
            if pv["lever"]["deg"] < 85.0 and s["tries"].get("valve", 0) < 2:  # (pushed but not shut: again)
                s["tries"]["valve"] = s["tries"].get("valve", 0) + 1
                s["steps"] = self.valve_steps(s, pv, obs, depth=0.02 * s["tries"]["valve"])
                return
            s["phase"] = "parts"
        if s["phase"] == "parts":
            steps = self.part_steps(s, pv, obs)
            if steps is None:  # (nothing reachable now: the burning pan's ring; again in a moment)
                s["steps"] = [dict(kind="wait", seconds=0.5)]
                return
            if steps:
                s["steps"] = steps
                return
            s["phase"] = "park"  # (well clear of everything, the arm tucked)
            s["steps"] = [dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=15.0),
                          dict(kind="drive", goal=list(PARK), heading=None, timeout=40.0, why="park"), dict(kind="stop", timeout=1e9)]
            return
        if s["phase"] == "park":
            s["phase"] = "done"
            s["done"] = True
        s["steps"] = [dict(kind="stop", timeout=1e9)]

    # -- the valve ------------------------------------------------------------------------------------------------
    def valve_steps(self, s, pv, obs, depth=0.0):
        stand, approach = (-1.20, 2.62), (-1.20, 2.25)  # (on the white square, facing the lever; turned out in front of it)
        return [dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=12.0, why="tuck"),
                dict(kind="drive", goal=approach, heading=0.0, timeout=40.0, why="valve approach"),
                dict(kind="straight", goal=stand, heading=0.0, timeout=15.0, why="valve stand"),
                dict(kind="lever", depth=depth, timeout=60.0, why="lever"),
                dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=12.0, why="tuck after the lever")]

    def lever_at(self, pv, a, depth, out=0.0):
        """A point of the push's arc: `a` round the lever's true hub from straight up (toward the closed stop), in the
        lever's plane (`depth` behind it), `out` in front of it."""
        hub = np.array(pv["lever"]["hub"])
        return hub + LEVER_R * np.array([math.sin(a), 0.0, math.cos(a)]) + np.array([0.0, depth - out, 0.0])

    def run_lever(self, st, s, pv, obs):
        """In front of the lever, into its plane behind the lever (20 degrees back from where it stands), round the
        true hub toward the closed stop; as soon as the lever reads shut, back along the arc and out of its plane (the
        arm never presses a lever that is at its stop)."""
        into = np.array([0.0, -1.0, 0.0])  # (the wrist on the vehicle's side: the tool points into the lever's plane)
        depth = st["depth"]
        if "path" not in st:
            a0 = math.radians(pv["lever"]["deg"] - 20.0)
            pts = [self.lever_at(pv, a0, depth, 0.25), self.lever_at(pv, a0, depth, 0.10), self.lever_at(pv, a0, depth)]
            a = a0
            while a < math.radians(95.0):
                a = min(math.radians(95.0), a + LEVER_STEP)
                pts.append(self.lever_at(pv, a, depth))
            q1, worst, _ = self.reach(s, pv, obs, pts[0], into)  # (in front of the lever: the best of several solutions,
            seg = self.joint_line(dict(s["q_tgt"]), q1)          # reached in joint space, clear of the valve)
            for k in range(1, len(pts)):
                more, w2 = self.line_path(dict(s, q_tgt=seg[-1]), pv, obs, pts[k], into, p0=pts[k - 1], wrist0=into, step=0.02)
                seg = seg + more
                worst = max(worst, w2)
            st.update(path=seg, k=0, held=0.0, phase="push")
            log(f"t={obs['time']:.1f} lever from {pv['lever']['deg']:.1f} deg: {len(seg)} waypoints, IK residual {worst:.3f} m")
        if st["phase"] == "push":
            done = self.follow(st, s, obs, ARM_SLOW, tol=0.04)
            st["held"] = st["held"] + self.dt if pv["lever"]["deg"] >= 88.0 else 0.0
            if st["held"] >= 0.3 or done:
                if pv["lever"]["deg"] >= 85.0 and s["lever_done_t"] is None:
                    s["lever_done_t"] = obs["time"]
                    log(f"t={obs['time']:.1f} the lever is shut ({pv['lever']['deg']:.1f} deg)")
                tool = self.tool_world(s, pv, obs)
                hub = np.array(pv["lever"]["hub"])
                a_tool = math.atan2(tool[0] - hub[0], tool[2] - hub[2])
                back = self.lever_at(pv, a_tool - math.radians(15.0), depth)
                b1, w1 = self.line_path(s, pv, obs, back, into, step=0.02)
                b2, w2 = self.line_path(dict(s, q_tgt=b1[-1]), pv, obs, self.lever_at(pv, a_tool - math.radians(15.0), depth, 0.25), into,
                                        p0=back, wrist0=into, step=0.03)
                st.update(path=b1 + b2, k=0, phase="retreat")
            return False
        return self.follow(st, s, obs, ARM_FAST, tol=0.05)

    # -- the parts -----------------------------------------------------------------------------------------------
    def pick_point(self, part):
        return np.array([part["center"][0], part["center"][1], part["hi"][2]])

    def clean(self, pv, name):
        """No other part's box within reach of the suction on the way down to this part's top."""
        p = self.pick_point(pv["parts"][name])
        for m, q in pv["parts"].items():
            if m == name or q["held"]:
                continue
            lo, hi = np.array(q["lo"]), np.array(q["hi"])
            for z in (p[2] + 0.015, p[2] + 0.04, p[2] + 0.07):
                x = np.array([p[0], p[1], z])
                if float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - x, x - hi)))) < 0.05:
                    return False
        return True

    def where(self, pv, name):
        part = pv["parts"][name]
        c = part["center"]
        lo, hi = pv["bay"]["crate"]
        if lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1] and c[2] <= hi[2] + 0.10:
            return "crate"
        for t in pv["bay"]["tables"]:
            if rect_dist(np.array([c[0]]), np.array([c[1]]), t["center"], t["size"], math.radians(t["yaw_deg"]))[0] < 0.01 and c[2] > t["height"]:
                return t["name"]
        return "elsewhere"

    def stands(self, rect, target_xy):
        """Stands outside each edge of an oriented rectangle, at the target's projection (kept 0.15 m inside the
        edge's ends), facing the edge: (stand, approach, heading)."""
        c, size, yaw = rect
        ax, ay = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
        d = np.asarray(target_xy, dtype=float) - np.asarray(c, dtype=float)
        out = []
        for n, t, hn, ht in ((ay, ax, size[1] / 2, size[0] / 2), (-ay, ax, size[1] / 2, size[0] / 2),
                             (ax, ay, size[0] / 2, size[1] / 2), (-ax, ay, size[0] / 2, size[1] / 2)):
            u = float(np.clip(d @ t, -(ht - 0.15), ht - 0.15)) if ht > 0.15 else 0.0
            base = np.asarray(c) + n * hn + t * u
            stand = base + n * STAND
            out.append((stand, stand + n * APPROACH, heading_of(-n), float((d @ n) - hn) * -1.0))
        return out

    def stand_ok(self, pv, occ, stand, approach, heading):
        x, y = stand
        if pv["fire"] and (math.hypot(x - pv["pan"][0], y - pv["pan"][1]) < RING or math.hypot(approach[0] - pv["pan"][0], approach[1] - pv["pan"][1]) < RING):
            return False
        if not self.planner.free(occ, approach) or self.planner.clearance(approach, pv["pan"]) < SPIN_ROOM:
            return False  # (the vehicle turns to face the edge there: it needs the room)
        foot = rect_corners(stand, (FOOT[1] + 0.04, FOOT[0] + 0.04), heading)  # (its width across, its length along the heading)
        for c, size, yaw in self.planner.rects:
            if rects_overlap(foot, rect_corners(c, size, yaw)):
                return False
        pan = pv["pan"]
        return math.hypot(x - pan[0], y - pan[1]) > 0.30 + 0.40

    def stand_root(self, stand, heading):
        """The root's pose (position, rotation) with the vehicle at a stand: level, its forward (+x) along the heading."""
        z = self.mis["ugv"]["start"][2]
        a = heading + math.pi / 2
        R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        return np.array([stand[0], stand[1], z]), R

    @staticmethod
    def wrist_options(stand, target):
        """The tool's axis for a pick (the wrist's side of the grip point): straight up, else leaning toward the
        vehicle (the arm then reaches further)."""
        u = np.array([stand[0] - target[0], stand[1] - target[1], 0.0])
        u = u / (np.linalg.norm(u) + 1e-9)
        out = []
        for lean in (0.0, 25.0, 40.0):
            d = np.array([0.0, 0.0, 1.0]) * math.cos(math.radians(lean)) + u * math.sin(math.radians(lean))
            out.append(d / np.linalg.norm(d))
        return out

    def reach_ok(self, s, pv, obs, stand, heading, pts, wrists):
        """The first tool axis with which the arm reaches every point from the stand (None: none does)."""
        rp, rR = self.stand_root(stand, heading)
        for wrist in wrists:
            q, ok = dict(s["home"]), True
            for p in pts:
                q, res, wres = self.kin.solve(rR.T @ (np.asarray(p) - rp), q, self.limits, wrist_dir=rR.T @ np.asarray(wrist), posture=s["home"], iters=150)
                if res > 0.008 or wres > 0.02:
                    ok = False
                    break
            if ok:
                return [float(x) for x in wrist]
        return None

    def part_steps(self, s, pv, obs):
        """The next part: Table A's first, then Table B's; the cheapest reachable stand outside the burning pan's
        ring. None: parts left, none reachable now; []: none left."""
        left = [n for n in pv["parts"] if self.where(pv, n) not in ("crate", "elsewhere") and n not in s["gave_up"]]
        if not left:
            return []
        occ = self.occ(pv)
        xy = self.pose(pv)[0]
        tables = {t["name"]: t for t in pv["bay"]["tables"]}
        best = None
        for name in sorted(left, key=lambda n: (self.where(pv, n) != "Table_A", not self.clean(pv, n))):
            tname = self.where(pv, name)
            t = tables[tname]
            rect = (t["center"], t["size"], math.radians(t["yaw_deg"]))
            top = self.pick_point(pv["parts"][name])
            for stand, appr, hd, depth in self.stands(rect, top[:2]):
                if not self.stand_ok(pv, occ, stand, appr, hd) or [name] + [round(float(v), 2) for v in stand] in s.get("bad_stands", []):
                    continue
                path = self.planner.plan(occ, xy, appr)
                if path is None:
                    continue
                cost = self.planner.length(path) + 2.0 * max(0.0, depth) + (0.0 if self.clean(pv, name) else 5.0) + (0.0 if tname == "Table_A" else 3.0)
                if best is not None and cost >= best[0]:
                    continue
                wrist = self.reach_ok(s, pv, obs, stand, hd, [top + [0, 0, PRE_ABOVE], top + [0, 0, 0.10], top + [0, 0, TOUCH]],
                                      self.wrist_options(stand, top))
                if wrist is None:
                    continue
                best = (cost, name, stand, appr, hd, wrist)
            if best is not None and self.where(pv, best[1]) == "Table_A" and tname != "Table_A":
                break
        if best is None:
            s["unreached"] = s.get("unreached", 0) + (0 if pv["fire"] else 1)
            if s["unreached"] >= 6:  # (no ring to wait for, for 3 s: what no stand reaches is given up)
                for n in left:
                    s["gave_up"].append(n)
                    log(f"t={obs['time']:.1f} {n}: no stand reaches it; given up")
                return self.part_steps(s, pv, obs)
            return None
        s["unreached"] = 0
        _, name, stand, appr, hd, wrist = best
        log(f"t={obs['time']:.1f} next part {name} on {self.where(pv, name)} from ({stand[0]:.2f}, {stand[1]:.2f}) heading {math.degrees(hd):.0f}, "
            f"tool leaning {math.degrees(math.acos(max(-1.0, min(1.0, wrist[2])))):.0f} deg")
        return [dict(kind="arm_line", to="tuck", speed=ARM_FAST, timeout=12.0, why="tuck"),
                dict(kind="drive", goal=list(appr), heading=hd, timeout=60.0, why=f"to {name}"),
                dict(kind="straight", goal=list(stand), heading=hd, timeout=15.0, why="in"),
                dict(kind="pick", part=name, wrist=wrist, stand=list(stand), timeout=40.0, why=f"pick {name}"),
                dict(kind="straight", goal=list(appr), heading=hd, timeout=15.0, why="out"),
                dict(kind="crate", part=name, timeout=120.0, why=f"crate {name}")]

    def run_pick(self, st, s, pv, obs):
        name = st["part"]
        part = pv["parts"][name]
        down = np.array(st.get("wrist") or [0.0, 0.0, 1.0])
        if "phase" not in st:
            top = self.pick_point(part)
            xy, psi, _, _ = self.pose(pv)
            wrist = self.reach_ok(s, pv, obs, xy, psi, [top + [0, 0, PRE_ABOVE], top + [0, 0, 0.10], top + [0, 0, TOUCH]],
                                  [np.array(st.get("wrist") or [0.0, 0.0, 1.0])] + self.wrist_options(xy, top))  # (from where it stands)
            if wrist is None:  # (out of the arm's reach from here: this stand is no good for this part)
                log(f"t={obs['time']:.1f} {name}: out of reach from ({xy[0]:.2f}, {xy[1]:.2f}), {math.dist(xy, st['stand']):.2f} m off the stand")
                s.setdefault("bad_stands", []).append([name] + [round(float(v), 2) for v in st["stand"]])
                s["steps"] = s["steps"][:1] + [x for x in s["steps"][1:] if x["kind"] != "crate"]
                return True
            st["wrist"] = wrist
            down = np.array(wrist)
            a, w1 = self.line_path(s, pv, obs, top + [0, 0, PRE_ABOVE], down, step=0.04)
            b, w2 = self.line_path(dict(s, q_tgt=a[-1]), pv, obs, top + [0, 0, 0.10], down, p0=top + [0, 0, PRE_ABOVE], step=0.03)
            c, w3 = self.line_path(dict(s, q_tgt=b[-1]), pv, obs, top + [0, 0, TOUCH], down, p0=top + [0, 0, 0.10], step=0.015)
            st.update(phase="down", path=a + b + c, k=0, n_grip=len(a + b), n_fast=len(a), top=top.tolist())
            if max(w1, w2, w3) > 0.015:
                log(f"t={obs['time']:.1f} {name}: IK residual {max(w1, w2, w3):.3f} m on the way down")
        if st["phase"] == "down":
            if st["k"] >= st["n_grip"] - 2:
                s["grip"] = True
            done = self.follow(st, s, obs, ARM_FAST if st["k"] < st.get("n_fast", 0) else ARM_SLOW, tol=0.006)
            if st["k"] >= len(st["path"]) - 1:
                st.setdefault("t_last", obs["time"])
                done = done or obs["time"] - st["t_last"] > 1.5
            if obs["ugv"].get("holding") is not None or done:
                st["phase"] = "held" if obs["ugv"].get("holding") is not None else "missed"
                st["t_held"] = obs["time"]
                tool = self.tool_world(s, pv, obs)
                log(f"t={obs['time']:.1f} {name}: {st['phase']} ({obs['ugv'].get('holding')}) {obs['time'] - st['t0']:.1f} s into the pick; "
                    f"tool at {np.round(tool, 3).tolist()}, part top {np.round(st['top'], 3).tolist()}, box {part['lo']} {part['hi']}")
            return False
        if st["phase"] == "missed":
            s["grip"] = False
            log(f"t={obs['time']:.1f} {name}: nothing held after the touch")
            s["tries"][name] = s["tries"].get(name, 0) + 1
            if s["tries"][name] >= 2:
                s["gave_up"].append(name)
            st["phase"] = "up"
            top = np.array(st["top"])
            st["path"], w = self.line_path(s, pv, obs, top + [0, 0, 0.30], down, step=0.03)
            st["k"] = 0
            st["fail"] = True
            st.pop("t_last", None)
        if st["phase"] == "held":
            top = np.array(st["top"])
            up, w = self.line_path(s, pv, obs, top + [0, 0, 0.30], down, step=0.03)
            st.update(phase="up", path=up, k=0)
        if st["phase"] == "up":
            if self.follow(st, s, obs, ARM_SLOW, tol=0.04):
                if st.get("fail"):
                    s["steps"] = s["steps"][:1] + [x for x in s["steps"][1:] if x["kind"] != "crate"]
                st["phase"] = "carry"
                st["path"], w = self.line_path(s, pv, obs, self.carry_point(pv), down, step=0.04)
                st["k"] = 0
            return False
        if st["phase"] == "carry":
            return self.follow(st, s, obs, ARM_FAST, tol=0.05)
        return False

    def carry_point(self, pv, which=CARRY):
        _, _, p, R = self.pose(pv)
        return p + R @ np.asarray(which, dtype=float)

    def run_crate(self, st, s, pv, obs):
        """To the crate's nearest free side, the part over a free slot, let go, check it landed inside."""
        held = obs["ugv"].get("holding")
        if "phase" not in st:
            if held is None:
                return True
            lo, hi = np.array(pv["bay"]["crate"][0]), np.array(pv["bay"]["crate"][1])
            c = (lo + hi) / 2
            occ = self.occ(pv)
            xy = self.pose(pv)[0]
            free = [k for k in range(len(SLOTS)) if k not in s["slots"]] or list(range(len(SLOTS)))
            best = None
            for stand, appr, hd, _ in self.stands(self.planner.crate, c[:2]):
                if not self.stand_ok(pv, occ, stand, appr, hd):
                    continue
                path = self.planner.plan(occ, xy, appr)
                if path is None:
                    continue
                for k in free:
                    slot = np.array([c[0] + SLOTS[k][0], c[1] + SLOTS[k][1], hi[2] + 0.10])
                    cost = self.planner.length(path) + 0.5 * math.dist(stand, slot[:2])
                    if best is None or cost < best[0]:
                        best = (cost, stand, appr, hd, k, slot)
            if best is None:
                st.setdefault("waits", 0)
                st["waits"] += 1
                return False
            _, stand, appr, hd, k, slot = best
            log(f"t={obs['time']:.1f} crate: slot {k} from ({stand[0]:.2f}, {stand[1]:.2f}) heading {math.degrees(hd):.0f}")
            st.update(phase="drive", stand=list(stand), appr=list(appr), hd=hd, slot_k=k, slot=slot.tolist(),
                      sub=[dict(kind="drive", goal=list(appr), heading=hd, timeout=60.0), dict(kind="straight", goal=list(stand), heading=hd, timeout=15.0)])
        if st["phase"] == "drive":
            sub = st["sub"][0]
            sub.setdefault("t0", obs["time"])
            wh, done = self.run_step(sub, s, pv, obs, None)
            if done or obs["time"] - sub["t0"] > sub["timeout"]:
                log(f"t={obs['time']:.1f} crate: {sub['kind']} {'done' if done else 'timed out'} in {obs['time'] - sub['t0']:.1f} s")
                st["sub"].pop(0)
                if not st["sub"]:
                    st["phase"] = "place"
            st["_wheels"] = wh
            return False
        if st["phase"] == "place":
            part = pv["parts"].get(held) if held else None
            if part is None:
                st["phase"] = "check"
                st["t_rel"] = obs["time"]
                return False
            tool = self.tool_world(s, pv, obs)
            below = float(tool[2] - part["lo"][2])
            lo, hi = np.array(pv["bay"]["crate"][0]), np.array(pv["bay"]["crate"][1])
            c = (lo + hi) / 2
            order = [st["slot_k"]] + [k for k in range(len(SLOTS)) if k != st["slot_k"] and k not in s["slots"]] + list(range(len(SLOTS)))
            xy, psi, _, _ = self.pose(pv)
            for k in order:  # (the planned slot, else the first free one the arm reaches from where the vehicle stands)
                slot = np.array([c[0] + SLOTS[k][0], c[1] + SLOTS[k][1], hi[2] + 0.10])
                if self.reach_ok(s, pv, obs, xy, psi, [slot + [0, 0, below]], [np.array([0.0, 0.0, 1.0])]) is not None:
                    st["slot_k"] = k
                    break
            p1 = np.array([slot[0], slot[1], slot[2] + below])
            st["path"], w = self.line_path(s, pv, obs, p1, np.array([0.0, 0.0, 1.0]), step=0.03)
            st.update(k=0, phase="lower", part=held)
            return False
        if st["phase"] == "lower":
            if self.follow(st, s, obs, ARM_SLOW, tol=0.03):
                part = pv["parts"].get(st["part"]) if st.get("part") else None
                lo, hi = np.array(pv["bay"]["crate"][0]), np.array(pv["bay"]["crate"][1])
                inner_lo, inner_hi = lo[:2] + 0.035, hi[:2] - 0.035
                inside = part is None or (np.all(np.array(part["lo"][:2]) >= inner_lo) and np.all(np.array(part["hi"][:2]) <= inner_hi))
                if not inside and st.get("aims", 0) < 3:  # (the vehicle slid while the arm reached: aim again from here)
                    st["aims"] = st.get("aims", 0) + 1
                    c_in = (inner_lo + inner_hi) / 2
                    pc = np.array(part["center"][:2])
                    shift = np.clip(c_in - pc, -(inner_hi - inner_lo) / 2, (inner_hi - inner_lo) / 2)
                    tool = self.tool_world(s, pv, obs)
                    log(f"t={obs['time']:.1f} {st['part']} not over the crate yet: aiming again ({shift[0]:+.2f}, {shift[1]:+.2f})")
                    st["path"], w = self.line_path(s, pv, obs, tool + np.array([shift[0], shift[1], 0.0]), np.array([0.0, 0.0, 1.0]), step=0.02)
                    st["k"] = 0
                    return False
                s["grip"] = False
                st["phase"], st["t_rel"] = "check", obs["time"]
            return False
        if st["phase"] == "check":
            if obs["time"] - st["t_rel"] < 1.2:
                return False
            name = st.get("part")
            ok = name is not None and self.where(pv, name) == "crate"
            log(f"t={obs['time']:.1f} {name} {'in the crate' if ok else 'missed the crate: ' + self.where(pv, name) if name else 'gone'}")
            if ok:
                s["slots"].append(st["slot_k"])
            st["phase"] = "away"
            p = self.tool_world(s, pv, obs) + np.array([0.0, 0.0, 0.12])
            st["path"], w = self.line_path(s, pv, obs, p, np.array([0.0, 0.0, 1.0]), step=0.03)
            st["k"] = 0
            return False
        if st["phase"] == "away":
            if self.follow(st, s, obs, ARM_FAST, tol=0.05):
                st["phase"] = "back"
                st["sub"] = [dict(kind="straight", goal=st["appr"], heading=st["hd"], timeout=15.0)]
            return False
        if st["phase"] == "back":
            sub = st["sub"][0]
            sub.setdefault("t0", obs["time"])
            wh, done = self.run_step(sub, s, pv, obs, None)
            st["_wheels"] = wh
            return bool(done or obs["time"] - sub["t0"] > sub["timeout"])
        return False

    def tool_world(self, s, pv, obs):
        tool_r, _ = self.kin.points(self.joints(obs))
        _, _, p, R = self.pose(pv)
        return p + R @ tool_r

    # -- steps ---------------------------------------------------------------------------------------------------
    def follow(self, st, s, obs, rate, tol=0.03):
        """Move the arm's targets along st['path'] (joint waypoints) at `rate`; True once at the end and there."""
        path = st["path"]
        if not path:
            return True
        k = min(st.get("k", 0), len(path) - 1)
        goal = path[k]
        step = rate * self.dt
        q = s["q_tgt"]
        for n in self.kin.joints:
            d = goal[n] - q[n]
            q[n] += float(np.clip(d, -step, step))
        arrived = all(abs(goal[n] - q[n]) < 1e-6 for n in self.kin.joints)
        joints = obs["ugv"]["joints"]
        if arrived and k < len(path) - 1:
            if all(abs(goal[n] - joints[n][0]) < 0.12 for n in self.kin.joints):
                st["k"] = k + 1
            return False
        return arrived and k == len(path) - 1 and all(abs(goal[n] - joints[n][0]) < tol for n in self.kin.joints)

    def run_step(self, st, s, pv, obs, hoist):
        """One control step of a step: (wheels, done)."""
        k = st["kind"]
        stop = self.wheels(0.0, 0.0)
        if k == "stop":
            return stop, False
        if k == "wait":
            return stop, obs["time"] - st["t0"] >= st["seconds"]
        if k == "arm_line":
            if "path" not in st:
                target = CARRY if (st["to"] == "tuck" and obs["ugv"].get("holding")) else TUCK
                st["path"], w = self.line_path(s, pv, obs, self.carry_point(pv, target), np.array([0.0, 0.0, 1.0]), step=0.05)
                st["k"] = 0
            return stop, self.follow(st, s, obs, st.get("speed", ARM_FAST), tol=0.06)
        if k == "drive":
            return self.drive(st, s, pv, obs)
        if k == "straight":
            return self.straight(st, pv)
        if k == "lever":
            return stop, self.run_lever(st, s, pv, obs)
        if k == "pick":
            return stop, self.run_pick(st, s, pv, obs)
        if k == "crate":
            done = self.run_crate(st, s, pv, obs)
            return st.pop("_wheels", stop), done
        return stop, True

    def spin(self, pv, err, st=None):
        """Turn on the spot toward `err` where there is room; where there is not, first straight on or back, whichever
        way gains room (the spin walks the root body), at most 0.2 m in a step."""
        xy, psi, _, _ = self.pose(pv)
        room = self.planner.clearance(xy, pv["pan"])
        if room < SPIN_ROOM and (st is None or st.get("crept", 0.0) < 0.2):
            fwd = np.array([-math.sin(psi), math.cos(psi)])
            ahead, behind = self.planner.clearance(xy + 0.1 * fwd, pv["pan"]), self.planner.clearance(xy - 0.1 * fwd, pv["pan"])
            if st is not None:
                st["crept"] = st.get("crept", 0.0) + V_SLOW * self.dt
            return self.wheels(V_SLOW if ahead >= behind else -V_SLOW, 0.0)
        w = float(np.clip(2.0 * err, -W_TURN, W_TURN))
        return self.wheels(0.0, math.copysign(max(abs(w), 0.3), err))

    def drive(self, st, s, pv, obs):
        xy, psi, _, _ = self.pose(pv)
        goal = np.array(st["goal"], dtype=float)
        if st.get("arrived"):
            return self.turn(pv, st["heading"], st) if st.get("heading") is not None else (self.wheels(0.0, 0.0), True)
        if "path" not in st or obs["time"] - st.get("t_plan", -9) > 2.0:
            occ = self.occ(pv)
            path = self.planner.plan(occ, xy, goal)
            st["t_plan"] = obs["time"]
            if path is None:
                st.pop("path", None)
                return self.wheels(0.0, 0.0), False
            st["path"] = path
        path = [np.array(p) for p in st["path"]]
        fwd = np.array([-math.sin(psi), math.cos(psi)])
        dg = float(np.linalg.norm(goal - xy))
        if dg < st.get("tol", 0.10):  # (close enough: a straight drive in, or the arm's IK on the true pose, takes up the rest)
            st["arrived"] = True
            return self.wheels(0.0, 0.0), st.get("heading") is None
        if dg < 0.30:  # near: straight on along the heading until level with it (a spin on the spot would move the root:
            along = float((goal - xy) @ fwd)  # the skid-steer's turning centre is not the root body's origin)
            if abs(along) < 0.02:
                st["arrived"] = True
                return self.wheels(0.0, 0.0), st.get("heading") is None
            return self.wheels(math.copysign(min(V_SLOW, 0.8 * abs(along) + 0.03), along), 0.0), False
        # pure pursuit: the point on the path 0.35 m ahead of the nearest one
        best, bi, bf = 1e9, 0, 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ab = b - a
            f = float(np.clip((xy - a) @ ab / (ab @ ab + 1e-12), 0.0, 1.0))
            d = float(np.linalg.norm(a + f * ab - xy))
            if d < best:
                best, bi, bf = d, i, f
        look, i, f = 0.35, bi, bf
        pt = path[-1]
        while i < len(path) - 1:
            a, b = path[i], path[i + 1]
            seg = float(np.linalg.norm(b - a))
            rest = (1.0 - f) * seg
            if rest >= look:
                pt = a + (f + look / max(seg, 1e-9)) * (b - a)
                break
            look -= rest
            i, f = i + 1, 0.0
        dist_goal = float(np.linalg.norm(goal - xy))
        err = wrap_a(heading_of(pt - xy) - psi)
        back = abs(err) > math.radians(110.0) and self.planner.length([list(xy)] + [list(p) for p in path[bi + 1:]]) < 2.5
        if back:  # (the way lies behind and is short: drive it backward instead of turning round)
            err = wrap_a(err - math.pi)
        if abs(err) > math.radians(35.0):  # (turn in place, where there is room)
            return self.spin(pv, err, st), False
        v = min(V_DRIVE, 0.8 * dist_goal + 0.05) * math.cos(err) ** 2
        w = float(np.clip(2.5 * err, -W_TURN, W_TURN))
        return self.wheels(-v if back else v, w), False

    def turn(self, pv, heading, st=None):
        """Turn in place to the heading (skid steering needs a firm turn: never slower than 0.3 rad/s), where there is
        room."""
        psi = self.pose(pv)[1]
        err = wrap_a(heading - psi)
        if abs(err) < math.radians(2.0):
            return self.wheels(0.0, 0.0), abs(pv["ugv"]["yaw_rate"]) < 5.0
        return self.spin(pv, err, st), False

    def straight(self, st, pv):
        """Straight to the goal, forward or back, the heading held."""
        xy, psi, _, _ = self.pose(pv)
        goal = np.array(st["goal"], dtype=float)
        d = goal - xy
        fwd = np.array([-math.sin(psi), math.cos(psi)])
        along = float(d @ fwd)
        if abs(along) < 0.012:
            return self.wheels(0.0, 0.0), abs(pv["ugv"]["vel"][0]) + abs(pv["ugv"]["vel"][1]) < 0.03
        err = wrap_a(st["heading"] - psi)
        v = math.copysign(min(V_SLOW, 1.2 * abs(along) + 0.02), along)
        return self.wheels(v, float(np.clip(2.0 * err, -0.5, 0.5))), False


# ============================================================================================= the facade
class Vehicle:
    """The vehicle's program: `Privileged` with the true state, `Baseline` without."""

    def __init__(self):
        self.base = Baseline()
        self.prog = None
        self.privileged = False

    def setup(self, mis, privileged=False):
        self.base.setup(mis)
        self.privileged = bool(privileged) and self.base.kin is not None
        if self.privileged:
            self.prog = Privileged(self.base.kin, mis, self.base.wheels)

    @property
    def mis(self):
        return self.base.mis

    def init_state(self):
        return self.prog.init_state() if self.privileged else self.base.init_state()

    def act(self, s, obs, hoist_phase):
        if self.privileged and "privileged" in obs and "ugv" in obs and obs["privileged"].get("ugv"):
            return self.prog.act(s, obs, hoist_phase)
        if self.privileged:
            return {}
        return self.base.act(s, obs, hoist_phase)
