"""The reference ground vehicle's program: drive on its odometry, work the valve's lever with the arm, park. Its
kinematics come from the scene it was built in (`scene.usda` beside it): the joints' frames read with pxr, forward
kinematics, a damped least-squares IK on the grip point.

A baseline, not a solution. It closes the valve and goes back to its bay; it does NOT clear the station. The part
routine below (fixed lanes round the tables, a pick from above with the suction tool, a drop into the crate) is kept
but switched off (CLEAR_STATION): on raw wheel odometry the vehicle drifts by tens of centimetres over the minutes
the parts take and runs into the tables, so the baseline would score differently every run. Doing the station needs
localisation against what the cameras see (`sees`), which is the agent's job.
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
        self.joints = [e["name"] for e, _ in self.chain if e["axis"] is not None]
        # the wheels: revolute joints off the root that are not on the arm's chain
        self.wheels = {}
        for e in edges:
            if e["axis"] is not None and e["b0"] == root and e["name"] not in self.joints:
                T = e["J0"]
                axis_root = T[:3, :3] @ np.array(e["axis"], dtype=float)
                self.wheels[e["name"]] = dict(pos=T[:3, 3].copy(), axis=axis_root)

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
class Vehicle:
    """The vehicle's task program. `state` is a plain dict (the controller deep-copies it for repeated queries)."""

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
