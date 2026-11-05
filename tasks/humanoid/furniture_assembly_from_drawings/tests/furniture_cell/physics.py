"""The cell's physical rules beyond rigid-body contact: hand grips, dowel joints, the back panel's seat, safety.

Grip (assisted pinch, documented in the cell manual): a hand that closes while its grasp site straddles a panel edge
(the panel's thickness along the hand's pinch axis within 35 deg, the grasp point inside the board's slab and at most
45 mm inside one of its edges) holds that panel rigidly at the relative pose of the closing instant; the finger
contacts are then off. It lets go when the hand opens, or slips when the hold needs more than the hand's limits
(force 30 N, torque 5 N m, for 50 ms).

Dowel joint (never reported): every dowel of a guest's end group within 5 mm of a distinct hole's axis of a host's
group, axes within 4 deg, the tips between 3 mm outside and fully inside the holes -> ENGAGED: a soft weld pulls the
guest to the seated pose those holes define (the dowels' chamfers and the cam's pull); seated within 1.5 mm / 1.5 deg
and at rest for 0.3 s -> LOCKED (rigid). An engaged guest pulled 12 mm / 8 deg away is released.

Back panel seat (never reported): the back panel within 4 mm across, 8 mm up / down, 3 mm in depth and 3 deg of its
place in a side panel's rabbet -> ENGAGED (soft) -> LOCKED like a dowel joint.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import furniture as F

MM = 1e-3
TICK = 0.01

GRIP_MAX_FORCE = 30.0
GRIP_MAX_TORQUE = 5.0
GRIP_SLIP_TICKS = 5
GRIP_SOLREF = (0.02, 1.0)          # a firm hold; the hands' squeeze between them is not a load (see Grips.tick)
GRIP_REACH = 0.045                 # m: the edge at most this far inside the hand
GRIP_SLAB = 0.015                  # m: the grasp point at most this far outside the board's faces
GRIP_AXIS_COS = math.cos(math.radians(35.0))

DOWEL_LATERAL = 5.0 * MM
DOWEL_AXIS_COS = math.cos(math.radians(4.0))
DOWEL_DEPTH = (-3.0 * MM, F.DOWEL_OUT * MM + 2.0 * MM)
SEAT = {"lateral": 4.0 * MM, "vertical": 8.0 * MM, "depth": 3.0 * MM, "angle": math.radians(3.0)}
LOCK_POS, LOCK_ANG, LOCK_TICKS = 1.5 * MM, math.radians(1.5), 30
LOCK_SPEED = 0.005
RELEASE_POS, RELEASE_ANG = 12.0 * MM, math.radians(8.0)
SOFT_SOLREF = (0.03, 1.0)
HARD_SOLREF = (0.004, 1.0)

SAFETY = {"part_speed": 3.0, "robot_contact_N": 150.0, "robot_contact_ticks": 20, "robot_robot_N": 80.0,
          "part_force_N": 400.0, "part_force_ticks": 20}


def _mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def rel_pose(d, b1, b2):
    """Pose of body b2 in body b1's frame: (p, R)."""
    R1 = d.xmat[b1].reshape(3, 3)
    R2 = d.xmat[b2].reshape(3, 3)
    return R1.T @ (d.xpos[b2] - d.xpos[b1]), R1.T @ R2


def pose_err(p_a, R_a, p_b, R_b):
    dR = R_a.T @ R_b
    ang = math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))
    return float(np.linalg.norm(p_a - p_b)), ang


def set_weld(m, d, eq, relpos, relR, solref, anchor=None):
    """Activate weld `eq` holding body2 at (relpos, relR) in body1's frame; `anchor` (body2 frame) is the point the
    constraint force acts at (its torque is about that point)."""
    # MuJoCo's weld: body2's point data[0:3] (body2 frame) is held on body1's point data[3:6] (body1 frame) and
    # body2's orientation at body1's times data[6:10]
    a = np.zeros(3) if anchor is None else np.asarray(anchor, float)
    m.eq_data[eq, 0:3] = a
    m.eq_data[eq, 3:6] = np.asarray(relpos, float) + np.asarray(relR, float) @ a
    m.eq_data[eq, 6:10] = _quat(relR)
    m.eq_data[eq, 10] = 1.0
    m.eq_solref[eq] = solref
    d.eq_active[eq] = 1


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| (rows are points)."""
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


class Bodies:
    """Body ids of the parts and the robots' hands."""

    def __init__(self, m, robots):
        self.part = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"part_{k}") for k in F.PARTS}
        self.key_of = {v: k for k, v in self.part.items()}
        self.part_geoms = {k: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
                           for k, b in self.part.items()}
        self.robots = robots


# ================================================================================================ grips

class Grips:
    """Assisted pinch grips of every hand."""

    def __init__(self, m, d, bodies: Bodies):
        self.m, self.d, self.b = m, d, bodies
        self.state = {}          # (robot, side) -> {"part": key | None, "eq": id, "over": n}
        self.eq = {}
        self.hand_geoms = {}
        for name, r in bodies.robots.items():
            for side in ("left", "right"):
                self.state[(name, side)] = {"part": None, "eq": -1, "over": 0, "closed": False}
                for key in F.PARTS:
                    self.eq[(name, side, key)] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                                                  f"grip_{name}_{side}_{key}")
                hb = set(int(x) for x in r.hand_bodies[side])
                gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in hb and (m.geom_contype[g] or m.geom_conaffinity[g])]
                self.hand_geoms[(name, side)] = (gs, m.geom_contype[gs].copy(), m.geom_conaffinity[gs].copy())
        self.events = []          # (time, robot, side, what, part) for the record

    def _edge_ok(self, key: str, site_pos, pinch_axis) -> float | None:
        """Distance score of a valid straddle of part `key`'s edge by the hand, or None."""
        d, b = self.d, self.b.part[key]
        R = d.xmat[b].reshape(3, 3)
        gl = R.T @ (site_pos - d.xpos[b])
        al = R.T @ pinch_axis
        part = F.PARTS[key]
        box = part.boxes[0]
        c = np.array(box.center) * MM
        h = np.array(part.half) * MM
        dl = gl - np.array([0.0, 0.0, 0.0])
        k = int(np.argmin(part.size))
        if abs(al[k]) < GRIP_AXIS_COS:
            return None
        if abs(dl[k] - 0.0) > h[k] + GRIP_SLAB:
            return None
        u, v = [i for i in range(3) if i != k]
        inside_u = h[u] - abs(dl[u])
        inside_v = h[v] - abs(dl[v])
        best = None
        if 0.0 <= inside_u <= GRIP_REACH and abs(dl[v]) <= h[v] - 0.005:
            best = inside_u
        if 0.0 <= inside_v <= GRIP_REACH and abs(dl[u]) <= h[u] - 0.005:
            best = inside_v if best is None else min(best, inside_v)
        _ = c
        return best

    def command(self, name: str, side: str, close: bool, t: float) -> None:
        st = self.state[(name, side)]
        if close and not st["closed"]:
            st["closed"] = True
            r = self.b.robots[name]
            sid = r.grasp_site[side]
            site_pos = self.d.site_xpos[sid].copy()
            pinch = self.d.site_xmat[sid].reshape(3, 3)[:, 1].copy()
            cands = []
            for key in F.PARTS:
                sc = self._edge_ok(key, site_pos, pinch)
                if sc is not None:
                    cands.append((sc, key))
            if cands:
                _, key = min(cands)
                self._engage(name, side, key, t)
        elif not close and st["closed"]:
            st["closed"] = False
            if st["part"] is not None:
                self._release(name, side, t, "opened")

    def _engage(self, name, side, key, t):
        st = self.state[(name, side)]
        eq = self.eq[(name, side, key)]
        r = self.b.robots[name]
        hb = r.hand_body[side]
        bp = self.b.part[key]
        p, R = rel_pose(self.d, hb, bp)
        # the grip's force acts at the grasp point: its torque is the torque the hand must hold
        anchor = self.d.xmat[bp].reshape(3, 3).T @ (self.d.site_xpos[r.grasp_site[side]] - self.d.xpos[bp])
        set_weld(self.m, self.d, eq, p, R, GRIP_SOLREF, anchor=anchor)
        st.update(part=key, eq=eq, over=0)
        gs, _, _ = self.hand_geoms[(name, side)]
        self.m.geom_contype[gs] = 0
        self.m.geom_conaffinity[gs] = 0
        self.events.append((t, name, side, "grip", key))

    def _release(self, name, side, t, why):
        st = self.state[(name, side)]
        if st["eq"] >= 0:
            self.d.eq_active[st["eq"]] = 0
        gs, ct, ca = self.hand_geoms[(name, side)]
        self.m.geom_contype[gs] = ct
        self.m.geom_conaffinity[gs] = ca
        self.events.append((t, name, side, why, st["part"]))
        st.update(part=None, eq=-1, over=0)

    def holding(self, name, side):
        return self.state[(name, side)]["part"]

    def weld_wrench(self, eq: int) -> tuple[np.ndarray, float]:
        """(force vector, torque norm) the weld `eq` transmits; the force's frame is the same for every grip weld (the
        world), so the forces of several hands on one board add up to the board's net load."""
        d = self.d
        rows = np.where((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == eq))[0]
        if len(rows) < 6:
            return np.zeros(3), 0.0
        f = d.efc_force[rows]
        return f[:3].copy(), float(np.linalg.norm(f[3:6]))

    def tick(self, t: float) -> None:
        """Slip rule: a board held by one hand slips when the hand's force exceeds GRIP_MAX_FORCE or its torque about the
        grasp point exceeds GRIP_MAX_TORQUE; a board held by k >= 2 hands slips from all of them when its net load
        (the vector sum of the hands' forces) exceeds k x GRIP_MAX_FORCE (the hands' squeeze on the board between them
        is not a load), for GRIP_SLIP_TICKS ticks in a row."""
        by_part = {}
        for (name, side), st in self.state.items():
            if st["part"] is None:
                continue
            fvec, torque = self.weld_wrench(st["eq"])
            st["force"], st["torque"] = float(np.linalg.norm(fvec)), torque
            by_part.setdefault(st["part"], []).append(((name, side), fvec, torque))
        for key, hands in by_part.items():
            k = len(hands)
            if k == 1:
                (hand, fvec, torque), = hands
                share = float(np.linalg.norm(fvec))
                over = share > GRIP_MAX_FORCE or torque > GRIP_MAX_TORQUE
            else:
                share = float(np.linalg.norm(sum(h[1] for h in hands))) / k
                over = share > GRIP_MAX_FORCE
            for hand, _, _ in hands:
                st = self.state[hand]
                st["over"] = st["over"] + 1 if over else 0
                st["load"] = share
                st["share"] = share
            if over and all(self.state[h[0]]["over"] >= GRIP_SLIP_TICKS for h in hands):
                for hand, _, _ in hands:
                    self._release(hand[0], hand[1], t, "slipped")

    def tactile(self, name, side) -> float:
        """The hand's pressure reading (N): its share of the held board's load when holding, else 0."""
        st = self.state[(name, side)]
        return float(st.get("load", 0.0)) if st["part"] is not None else 0.0


# ================================================================================================ joints

def _groups(part: F.Part, which: str):
    out = {}
    for peg in (part.holes if which == "holes" else part.dowels):
        out.setdefault(peg.group, []).append(peg)
    return out


class Coupling:
    """Dowel joints between the side panels (hosts) and the horizontal panels (guests), and the back panel's seat."""

    HOSTS = ("L", "R")
    GUESTS = ("T", "C", "D")

    def __init__(self, m, d, bodies: Bodies):
        self.m, self.d, self.b = m, d, bodies
        self.host_groups = {h: _groups(F.PARTS[h], "holes") for h in self.HOSTS}
        self.guest_groups = {g: _groups(F.PARTS[g], "dowels") for g in self.GUESTS}
        # one weld per (host, guest) pair; state per pair
        self.state = {}
        for h in self.HOSTS:
            for g in self.GUESTS + ("K",):
                eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"joint_{h}_{g}")
                self.state[(h, g)] = {"state": "FREE", "eq": eq, "target": None, "hold": 0, "match": None,
                                      "t_engaged": None, "t_locked": None}
        # the back panel's design pose in each side panel's frame
        self.k_in_host = {}
        for h in self.HOSTS:
            ph = np.array(F.PARTS[h].design_center) * MM
            pk = np.array(F.PARTS["K"].design_center) * MM
            self.k_in_host[h] = (pk - ph, np.eye(3))
        self.events = []

    # ---- dowel alignment
    def _dowel_match(self, h, g):
        """(host_group, guest_group, seated relative pose (p, R) of the guest in the host) when a guest end group is
        engaged in a host hole group, else None."""
        d, b = self.d, self.b
        bh, bg = b.part[h], b.part[g]
        Rh, ph = d.xmat[bh].reshape(3, 3), d.xpos[bh]
        Rg, pg = d.xmat[bg].reshape(3, 3), d.xpos[bg]
        for gg, dowels in self.guest_groups[g].items():
            tips = [(pg + Rg @ (np.array(dw.pos) * MM + np.array(dw.axis) * F.DOWEL_OUT * MM), Rg @ np.array(dw.axis))
                    for dw in dowels]
            for hg, holes in self.host_groups[h].items():
                mouths = [(ph + Rh @ (np.array(hl.pos) * MM), Rh @ np.array(hl.axis)) for hl in holes]
                used, pairs = set(), []
                for (tp, ta) in tips:
                    best = None
                    for j, (mp, ma) in enumerate(mouths):
                        if j in used or float(ta @ ma) < DOWEL_AXIS_COS:
                            continue
                        depth = float((tp - mp) @ ma)
                        if not (DOWEL_DEPTH[0] <= depth <= DOWEL_DEPTH[1]):
                            continue
                        lat = float(np.linalg.norm((tp - mp) - depth * ma))
                        if lat <= DOWEL_LATERAL and (best is None or lat < best[0]):
                            best = (lat, j)
                    if best is None:
                        break
                    used.add(best[1])
                    pairs.append(best[1])
                if len(pairs) != len(tips):
                    continue
                # seated pose: dowel bases on the hole mouths, tips at depth DOWEL_OUT (host frame)
                P, Q = [], []
                for dw, j in zip(dowels, pairs):
                    hl = holes[j]
                    base = np.array(dw.pos) * MM
                    tip = base + np.array(dw.axis) * F.DOWEL_OUT * MM
                    mouth = np.array(hl.pos) * MM
                    deep = mouth + np.array(hl.axis) * F.DOWEL_OUT * MM
                    P += [base, tip]
                    Q += [mouth, deep]
                Rs, ts = kabsch(np.array(P), np.array(Q))
                return hg, gg, (ts, Rs)
        return None

    def _k_match(self, h):
        d, b = self.d, self.b
        p, R = rel_pose(d, b.part[h], b.part["K"])
        pd, _ = self.k_in_host[h]
        dp = p - pd
        if abs(dp[0]) > SEAT["lateral"] or abs(dp[2]) > SEAT["vertical"] or abs(dp[1]) > SEAT["depth"]:
            return None
        # the back panel is a plain board: any of its four box symmetries seats it (either face to the front)
        for Rd in (np.eye(3), np.diag([-1.0, -1.0, 1.0]), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0])):
            ang = math.acos(float(np.clip((np.trace(Rd.T @ R) - 1) / 2, -1.0, 1.0)))
            if ang <= SEAT["angle"]:
                return pd.copy(), Rd
        return None

    def tick(self, t: float, held_parts: set) -> None:
        d, m = self.d, self.m
        for (h, g), st in self.state.items():
            bh, bg = self.b.part[h], self.b.part[g]
            if st["state"] == "LOCKED":
                continue
            if st["state"] == "FREE":
                if g == "K":
                    mt = self._k_match(h)
                    if mt is None:
                        continue
                    st["target"] = mt
                    st["match"] = (f"{h}.rabbet", "K.panel")
                else:
                    mt = self._dowel_match(h, g)
                    if mt is None:
                        continue
                    hg, gg, pose = mt
                    st["target"] = pose
                    st["match"] = (hg, gg)
                p_t, R_t = st["target"]
                set_weld(m, d, st["eq"], p_t, R_t, SOFT_SOLREF)
                st.update(state="ENGAGED", hold=0, t_engaged=t)
                self.events.append((t, "engaged", h, g, st["match"]))
                continue
            # ENGAGED: lock when seated and at rest, release when pulled away
            p, R = rel_pose(d, bh, bg)
            p_t, R_t = st["target"]
            ep, ea = pose_err(p, R, p_t, R_t)
            if ep > RELEASE_POS or ea > RELEASE_ANG:
                d.eq_active[st["eq"]] = 0
                st.update(state="FREE", hold=0, target=None, match=None)
                self.events.append((t, "released", h, g, None))
                continue
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bg, v, 0)
            vh = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bh, vh, 0)
            rel_speed = float(np.linalg.norm(v[3:] - vh[3:]))
            if ep <= LOCK_POS and ea <= LOCK_ANG and rel_speed <= LOCK_SPEED:
                st["hold"] += 1
                if st["hold"] >= LOCK_TICKS:
                    set_weld(m, d, st["eq"], p_t, R_t, HARD_SOLREF)
                    st.update(state="LOCKED", t_locked=t)
                    self.events.append((t, "locked", h, g, st["match"]))
            else:
                st["hold"] = 0

    def locked(self):
        return {k: dict(v) for k, v in self.state.items() if v["state"] == "LOCKED"}


# ================================================================================================ safety

class Safety:
    """Per-tick safety watch: a violation zeroes the episode's score (the episode goes on)."""

    def __init__(self, m, d, bodies: Bodies):
        self.m, self.d, self.b = m, d, bodies
        self.robot_of_body = {}
        for name, r in bodies.robots.items():
            for bid in r.body_ids:
                self.robot_of_body[int(bid)] = name
        self.hand_bodies = set()
        for r in bodies.robots.values():
            for side in ("left", "right"):
                self.hand_bodies.update(int(x) for x in r.hand_bodies[side])
        self.part_bodies = set(bodies.part.values())
        self.count = {}
        self.violations = []
        self.peaks = {"robot_contact_N": 0.0, "robot_robot_N": 0.0, "part_force_N": 0.0, "part_speed": 0.0}
        self.peak_who = {}

    def tick(self, t: float) -> None:
        m, d = self.m, self.d
        robot_c, rr_c, part_c = {}, 0.0, {}
        self._who = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = self.robot_of_body.get(b1), self.robot_of_body.get(b2)
            if r1 is None and r2 is None and b1 not in self.part_bodies and b2 not in self.part_bodies:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            if r1 is not None and r2 is not None and r1 != r2:
                rr_c = max(rr_c, fn)
            elif r1 is not None or r2 is not None:
                rb = b1 if r1 is not None else b2
                if rb not in self.hand_bodies:
                    robot_c[rb] = robot_c.get(rb, 0.0) + fn
                    self._who[("robot", rb)] = (b1, b2)
            else:
                pair = (min(b1, b2), max(b1, b2))
                part_c[pair] = part_c.get(pair, 0.0) + fn
                self._who[("part", pair)] = (b1, b2)
        worst_robot = max(robot_c.values(), default=0.0)
        worst_part = max(part_c.values(), default=0.0)
        self._worst_robot = max(robot_c, key=robot_c.get) if robot_c else None
        self._worst_part = max(part_c, key=part_c.get) if part_c else None
        if worst_robot > self.peaks["robot_contact_N"] and self._worst_robot is not None:
            pair = self._who.get(("robot", self._worst_robot))
            self.peak_who["robot_contact_N"] = (round(t, 2), [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "world"
                                                             for b in pair])
        if worst_part > self.peaks["part_force_N"] and self._worst_part is not None:
            pair = self._who.get(("part", self._worst_part))
            self.peak_who["part_force_N"] = (round(t, 2), [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "world"
                                                          for b in pair])
        self.peaks["robot_contact_N"] = max(self.peaks["robot_contact_N"], worst_robot)
        self.peaks["robot_robot_N"] = max(self.peaks["robot_robot_N"], rr_c)
        self.peaks["part_force_N"] = max(self.peaks["part_force_N"], worst_part)
        self._streak("robot_contact", worst_robot > SAFETY["robot_contact_N"], SAFETY["robot_contact_ticks"], t,
                     worst_robot)
        self._streak("part_force", worst_part > SAFETY["part_force_N"], SAFETY["part_force_ticks"], t, worst_part)
        if rr_c > SAFETY["robot_robot_N"]:
            self._violate(t, "robot_robot_contact", rr_c)
        for key, b in self.b.part.items():
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            sp = float(np.linalg.norm(v[3:]))
            self.peaks["part_speed"] = max(self.peaks["part_speed"], sp)
            if sp > SAFETY["part_speed"]:
                self._violate(t, f"part_speed:{key}", sp)

    def _streak(self, what, cond, ticks, t, val):
        n = self.count.get(what, 0) + 1 if cond else 0
        self.count[what] = n
        if n == ticks:
            key = ("robot", self._worst_robot) if what == "robot_contact" else ("part", self._worst_part)
            pair = self._who.get(key)
            if pair:
                names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "world" for b in pair]
                what = "%s:%s/%s" % (what, names[0], names[1])
            self._violate(t, what, val)

    def _violate(self, t, what, val):
        if len(self.violations) < 50:
            self.violations.append({"t": round(t, 3), "what": what, "value": round(float(val), 2)})

    @property
    def ok(self) -> bool:
        return not self.violations
