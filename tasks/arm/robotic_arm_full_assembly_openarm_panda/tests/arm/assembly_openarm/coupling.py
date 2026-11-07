"""Coupling state machine of the OpenArm assembly cell (one state per mate of the kit; see the full-arm cell's
`assembly_full/coupling.py`, whose states and rules this follows for the OpenArm's two interfaces).

Mates (`kit.json` "mates"):
  seat     the J2 motor in J1's yoke: the motor's frame within tol_pos / tol_ang of its design pose in the carrier
  flange   a unit bolted onto a motor's output flange (rotor): the two mate frames (the flange face's centre, z along the
           face normal, one in each body) on one axis within tol_ang, the axis lines within tol_pos, the faces within
           axial_tol, and the rotation about the axis on a bolt position: within tol_ang of a multiple of the bolt
           pitch (60 deg). ACHIEVED only on the design's bolt position (`design_clock_deg`, the rotor's index mark):
           another bolt position fastens (welded) but counts 0

States: WAITING (a required mate missing) -> FREE -> ENGAGED (inside the capture region: a soft weld holds the part on
its feature; a seat pulled to its design pose from at most `capture_seat_depth_m` short of home, a flange onto the
axis and the face with its rotation about the axis left free, re-anchored every tick) -> HOLDING (the predicate holds
and the part is at rest, `hold_ticks` ticks) -> LOCKED (the weld turned rigid at the bolt position = the screws) or
LOCKED_OFF (fastened on another bolt position). A fastened flange's joint stays free: the motors are unpowered, the
hinge turns against its back-drive friction; at the fastening its range is set to the design's joint limits (mapped
through the as-built angle). Score = the achieved mates' weight / the total weight.
"""

import math

import numpy as np

STATES = ("WAITING", "FREE", "ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def rotation_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def body_pose(d, b):
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CouplingStateMachine(object):
    def __init__(self, model, data, kit, hold_ticks=50, rel_speed_max=0.005, rest_ang_speed_max=math.radians(2.0),
                 capture_pos_m=0.004, capture_ang_rad=math.radians(8.0), capture_axial_m=0.004, capture_seat_depth_m=0.003,
                 release_factor=2.0, soft_solref=(0.04, 1.0), insertion=None):
        self.m, self.d, self.kit = model, data, kit
        self.hold_ticks, self.rel_speed_max, self.rest_ang = int(hold_ticks), float(rel_speed_max), float(rest_ang_speed_max)
        self.cap_pos, self.cap_ang, self.cap_axial, self.seat_depth = float(capture_pos_m), float(capture_ang_rad), float(capture_axial_m), float(capture_seat_depth_m)
        self.release_factor = float(release_factor)
        self.soft_solref = [float(v) for v in soft_solref]
        self.insertion = dict(insertion or {})          # mate id -> unit insertion direction in the carrier's frame (seats)
        self.mates = []
        for mm in kit["mates"]:
            x = dict(mm)
            x["Tpc"] = np.array(mm["T_part_in_carrier"], float)
            x["Fc"] = np.array(mm["frame_in_carrier"], float)
            x["Fp"] = np.array(mm["frame_in_part"], float)
            x["tol_pos"] = mm["tol_pos_mm"] * 1e-3
            x["tol_ang"] = math.radians(mm["tol_ang_deg"])
            x["axial_tol"] = mm.get("axial_tol_mm", 0.0) * 1e-3
            x["pitch"] = math.radians(mm.get("bolt_pitch_deg", 360.0))
            x["design_k"] = int(round(mm.get("design_clock_deg", 0.0) / mm.get("bolt_pitch_deg", 360.0))) if mm["kind"] == "flange" else 0
            x["bc"], x["bp"] = model.body(mm["carrier"]).id, model.body(mm["part"]).id
            x["eq"] = model.equality("weld_" + mm["id"]).id
            x["cap"] = [model.equality("cap_%s_%d" % (mm["id"], i)).id for i in range(2)] if mm["kind"] == "flange" else []
            # the welds the mate makes besides its own pair (the forearm's flange fastens J4_D): each must sit within the
            # capture of its design pose to be held, within the tolerance to fasten
            x["co"] = [{"bc": model.body(cw["carrier"]).id, "bp": model.body(cw["part"]).id, "Tpc": np.array(cw["T_part_in_carrier"], float),
                        "eq": model.equality("weld_%s__%s" % (mm["id"], cw["part"])).id, "name": cw["part"]} for cw in mm.get("co_welds", [])]
            self.mates.append(x)
        self.by_id = {x["id"]: x for x in self.mates}
        self.hard_solref = {x["id"]: model.eq_solref[x["eq"]].copy() for x in self.mates}
        self.co_hard = {c["eq"]: model.eq_solref[c["eq"]].copy() for x in self.mates for c in x["co"]}
        self.state = {x["id"]: "WAITING" for x in self.mates}
        self.rec = {x["id"]: {"id": x["id"], "kind": x["kind"], "weight": x["weight"], "fastened": False, "achieved": False, "t_s": None,
                              "pos_err_mm": None, "ang_err_deg": None, "axial_err_mm": None, "clock_deg": None, "bolt_k": None,
                              "joint_offset_rad": None} for x in self.mates}
        self.hold = {x["id"]: 0 for x in self.mates}
        self.captured = set()
        self.events = []
        self.total_w = sum(x["weight"] for x in self.mates)
        self.limits_set = {}
        # joints: hinge id, the bodies on either side (proximal, distal) and the joint's design limits
        self.joint_info = {}
        for jn, js in kit["joints"].items():
            pd = kit.get("joint_links", {}).get(jn)
            hinge = [b for b, bb in kit["bodies"].items() if (bb.get("joint") or {}).get("name") == jn]
            if not pd or not hinge:
                continue
            hb = hinge[0]
            # the hinge's sense: +1 when turning it by +theta turns the distal link by +theta about the joint axis (the
            # hinge axis along the joint axis, its child on the distal side); a rotor bolted to the PROXIMAL link (J3's,
            # inside the upper arm, bolted to L2) turns the proximal side instead: -1
            ax_h = np.asarray(kit["bodies"][hb]["joint"]["axis"], float)
            s0 = 1.0 if float(ax_h @ np.asarray(js["axis_arm"], float)) > 0 else -1.0
            bolted_to = [mm["part"] if mm["carrier"] == hb else mm["carrier"] for mm in kit["mates"] if hb in (mm["carrier"], mm["part"])]
            unit_of = lambda b: kit["bodies"][b]["unit"]
            if bolted_to and unit_of(bolted_to[0]) != unit_of(pd[1]):
                s0 = -s0
            self.joint_info[jn] = {"hinge": model.joint("hinge_" + hb).id, "prox": model.body(pd[0]).id, "dist": model.body(pd[1]).id,
                                   "axis": np.asarray(js["axis_arm"], float), "range": [float(v) for v in js["range"]], "sense": s0}

    # ------------------------------------------------------------------ predicates
    def score(self):
        return sum(x["weight"] for x in self.mates if self.rec[x["id"]]["achieved"]) / self.total_w

    def _vel(self, b):
        import mujoco
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
        return v

    def evaluate(self, x):
        d = self.d
        Tc, Tp = body_pose(d, x["bc"]), body_pose(d, x["bp"])
        if x["kind"] == "seat":
            E = _inv(Tc @ x["Tpc"]) @ Tp                     # the part relative to its design pose (carrier frame axes)
            r = {"pos_err": float(np.linalg.norm(E[:3, 3])), "ang_err": rotation_angle(E[:3, :3])}
            ins = self.insertion.get(x["id"])
            if ins is not None:
                ins_w = Tc[:3, :3] @ np.asarray(ins, float)
                dw = Tp[:3, 3] - (Tc @ x["Tpc"])[:3, 3]
                r["axial"] = float(dw @ ins_w)
                r["lateral"] = float(np.linalg.norm(dw - r["axial"] * ins_w))
            else:
                r["axial"], r["lateral"] = 0.0, r["pos_err"]
            r["holds"] = r["pos_err"] <= x["tol_pos"] and r["ang_err"] <= x["tol_ang"]
        else:
            A, B = Tc @ x["Fc"], Tp @ x["Fp"]
            E = _inv(A) @ B
            za = A[:3, 2]
            ang = math.acos(max(-1.0, min(1.0, float(za @ B[:3, 2]))))
            dv = B[:3, 3] - A[:3, 3]
            axial = float(dv @ za)
            line = float(np.linalg.norm(dv - axial * za))
            # the rotation about the axis: the twist of E about A's z (swing-twist), not atan2 of E's x axis: with the axes a
            # little apart the latter re-anchors the soft hold with a small twist every tick, and a free rotor spun
            qe = quat_wxyz(E[:3, :3])
            clock = wrap(2.0 * math.atan2(float(qe[3]), float(qe[0])))
            k = int(round(clock / x["pitch"]))
            res = wrap(clock - k * x["pitch"])
            nk = int(round(2 * math.pi / x["pitch"]))
            r = {"pos_err": line, "ang_err": ang, "axial": axial, "clock": clock, "k": k % nk, "clock_res": res,
                 "holds": ang <= x["tol_ang"] and line <= x["tol_pos"] and abs(axial) <= x["axial_tol"] and abs(res) <= x["tol_ang"]}
            r["design_ok"] = (k - x["design_k"]) % nk == 0
        for c in x["co"]:
            E = _inv(body_pose(d, c["bc"]) @ c["Tpc"]) @ body_pose(d, c["bp"])
            pe, ae = float(np.linalg.norm(E[:3, 3])), rotation_angle(E[:3, :3])
            r.setdefault("co", []).append((pe, ae))
            r["holds"] = r["holds"] and pe <= x["tol_pos"] and ae <= x["tol_ang"]
        if r["holds"]:
            vc, vp = self._vel(x["bc"]), self._vel(x["bp"])
            r["rel_speed"] = float(np.linalg.norm(vp[3:] - vc[3:]))
            r["rel_ang_speed"] = float(np.linalg.norm(vp[:3] - vc[:3]))
        return r

    def _inside(self, x, r, f=1.0):
        if any(pe > self.cap_pos * f or ae > self.cap_ang * f for pe, ae in r.get("co", [])):
            return False
        if x["kind"] == "seat":
            return (r["ang_err"] <= self.cap_ang * f and r["lateral"] <= self.cap_pos * f and -self.seat_depth * f <= r["axial"] <= self.cap_pos * f)
        return r["pos_err"] <= self.cap_pos * f and r["ang_err"] <= self.cap_ang * f and abs(r["axial"]) <= self.cap_axial * f

    # ------------------------------------------------------------------ welds
    def _capture_flange(self, x):
        """the flange's soft hold: the two point constraints on the axis (rotation about the axis free); the co-welds soft"""
        for i, e in enumerate(x["cap"]):
            # the anchors on the axis in both bodies (the compiler's anchor2 comes from qpos0: the part on the bench)
            h = (0.0, 0.02)[i]
            self.m.eq_data[e, 0:3] = (x["Fc"] @ np.array([0, 0, h, 1.0]))[:3]
            self.m.eq_data[e, 3:6] = (x["Fp"] @ np.array([0, 0, h, 1.0]))[:3]
            self.d.eq_active[e] = 1
        for c in x["co"]:
            ce = c["eq"]
            self.m.eq_data[ce, 0:3] = 0.0
            self.m.eq_data[ce, 3:6] = c["Tpc"][:3, 3]
            self.m.eq_data[ce, 6:10] = quat_wxyz(c["Tpc"][:3, :3])
            self.m.eq_data[ce, 10] = 1.0
            self.m.eq_solref[ce] = self.soft_solref
            self.d.eq_active[ce] = 1

    def _weld(self, x, rel, soft):
        for e in x["cap"]:
            self.d.eq_active[e] = 0
        e = x["eq"]
        self.m.eq_data[e, 0:3] = 0.0
        self.m.eq_data[e, 3:6] = rel[:3, 3]
        self.m.eq_data[e, 6:10] = quat_wxyz(rel[:3, :3])
        self.m.eq_data[e, 10] = 1.0
        self.m.eq_solref[e] = self.soft_solref if soft else self.hard_solref[x["id"]]
        self.d.eq_active[e] = 1
        for c in x["co"]:
            ce = c["eq"]
            self.m.eq_data[ce, 0:3] = 0.0
            self.m.eq_data[ce, 3:6] = c["Tpc"][:3, 3]
            self.m.eq_data[ce, 6:10] = quat_wxyz(c["Tpc"][:3, :3])
            self.m.eq_data[ce, 10] = 1.0
            self.m.eq_solref[ce] = self.soft_solref if soft else self.co_hard[ce]
            self.d.eq_active[ce] = 1

    def _target(self, x, clock):
        """the part's pose in the carrier's frame: a seat at its design, a flange on the axis and face at `clock`"""
        if x["kind"] == "seat":
            return x["Tpc"]
        return x["Fc"] @ _rotz(clock) @ _inv(x["Fp"])

    def _release(self, x):
        for e in x["cap"]:
            self.d.eq_active[e] = 0
        self.d.eq_active[x["eq"]] = 0
        self.m.eq_solref[x["eq"]] = self.hard_solref[x["id"]]
        for c in x["co"]:
            self.d.eq_active[c["eq"]] = 0
            self.m.eq_solref[c["eq"]] = self.co_hard[c["eq"]]
        self.captured.discard(x["id"])

    # ------------------------------------------------------------------ joints
    def joint_angle(self, jn):
        """the as-built joint angle: the distal link's frame relative to the proximal one about the joint axis (the
        frames are parallel at the design zero)"""
        ji = self.joint_info[jn]
        R = self.d.xmat[ji["prox"]].reshape(3, 3).T @ self.d.xmat[ji["dist"]].reshape(3, 3)
        ax = ji["axis"] / np.linalg.norm(ji["axis"])
        # the rotation's angle about the axis (swing-twist decomposition)
        q = quat_wxyz(R)
        tw = float(q[0]), float(np.dot(q[1:], ax))
        return wrap(2.0 * math.atan2(tw[1], tw[0]))

    def _limit(self, jn):
        """the joint's design range on its hinge: theta = theta_now + (q - q_now) * s, s = +-1 the hinge's sense"""
        ji = self.joint_info.get(jn)
        if ji is None:
            return
        m, d = self.m, self.d
        j = ji["hinge"]
        a = m.jnt_qposadr[j]
        th0, q0 = float(d.qpos[a]), self.joint_angle(jn)
        s = ji["sense"]
        lo, hi = ji["range"]
        th = sorted([th0 + (lo - q0) * s, th0 + (hi - q0) * s])
        m.jnt_range[j] = [min(th[0], th0), max(th[1], th0)]
        m.jnt_limited[j] = 1
        self.limits_set[jn] = tuple(m.jnt_range[j])

    # ------------------------------------------------------------------ per control tick
    def tick(self, sim_time):
        for x in self.mates:
            mid = x["id"]
            st = self.rec[mid]
            if st["fastened"]:
                self.state[mid] = "LOCKED" if st["achieved"] else "LOCKED_OFF"
                continue
            if not all(self.rec[q]["achieved"] for q in x["requires"]):
                self.state[mid] = "WAITING"
                self.hold[mid] = 0
                continue
            r = self.evaluate(x)
            was = mid in self.captured
            if self._inside(x, r, self.release_factor if was else 1.0):
                if x["kind"] == "flange":
                    self._capture_flange(x)
                else:
                    self._weld(x, self._target(x, 0.0), soft=True)
                self.captured.add(mid)
                if not was:
                    self.events.append((sim_time, "ENGAGED %s" % mid))
            elif was:
                self._release(x)
                self.events.append((sim_time, "released from capture %s" % mid))
            if mid not in self.captured:
                self.state[mid] = "FREE"
                self.hold[mid] = 0
                continue
            still = r["holds"] and r.get("rel_speed", 1.0) < self.rel_speed_max and r.get("rel_ang_speed", 1.0) < self.rest_ang
            if not still:
                self.state[mid] = "ENGAGED"
                self.hold[mid] = 0
                continue
            self.hold[mid] += 1
            self.state[mid] = "HOLDING"
            if self.hold[mid] >= self.hold_ticks:
                self._lock(x, r, sim_time)

    def _lock(self, x, r, sim_time):
        mid = x["id"]
        st = self.rec[mid]
        st["fastened"], st["t_s"] = True, float(sim_time)
        st["pos_err_mm"], st["ang_err_deg"] = r["pos_err"] * 1e3, math.degrees(r["ang_err"])
        st["axial_err_mm"] = r.get("axial", 0.0) * 1e3
        if x["kind"] == "flange":
            st["achieved"] = bool(r["design_ok"])
            st["clock_deg"], st["bolt_k"] = math.degrees(r["clock"]), int(r["k"])
            self._weld(x, self._target(x, r["k"] * x["pitch"]), soft=False)
        else:
            st["achieved"] = True
            self._weld(x, self._target(x, 0.0), soft=False)
        self.captured.discard(mid)
        if x["kind"] == "flange":
            import mujoco
            mujoco.mj_kinematics(self.m, self.d)
            self._limit(x["joint"])
            st["joint_offset_rad"] = None if x["joint"] not in self.joint_info else self.joint_angle(x["joint"])
        self.state[mid] = "LOCKED" if st["achieved"] else "LOCKED_OFF"
        self.events.append((sim_time, ("LOCKED %s" % mid) if st["achieved"] else "LOCKED_OFF %s (bolt position %d, design %d)" % (mid, st["bolt_k"], x["design_k"])))

    def record(self):
        return [dict(self.rec[x["id"]]) for x in self.mates]
