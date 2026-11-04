"""The two Pandas assemble the OpenArm pair from its loose units: the privileged program of the OpenArm cell.

Privileged state only: the units' exact poses, the design poses from the kit, the Pandas' own kinematics. Nothing but a
Panda's gripper moves a unit (no jig, no feeder). The arms being built are physical: their motors are unpowered, a
fastened joint turns against the motor's back-drive friction within the design range (`coupling.py`), J4_D turns on its
bearing; every target is computed from the arm as it is when the unit's turn comes and is followed while it goes in.

Sequence (`sequence`), per arm by its own Panda (`ORDER`: the left arm up to HB, the right arm, the rest of the left
arm; the idle Panda parked): L1 onto
J1's flange, the J2 motor into L1's yoke, L2 onto J2's flange, the upper arm onto L2 (J3's rotor inside it), HB onto
J4_D (on its bearing in the upper arm), the forearm onto J4's flange (it fastens J4_D), HA onto the forearm's J4_E, the
wrist onto J5's flange, the gripper onto J7's flange (`ARM_ORDER`, `STEPS`).
Per unit (after the full-arm task's `assembly_full/reference.py`):
  grasp     candidates from rays through the unit's rigid collision geoms (two antipodal pads, closing width 4-75 mm,
            approach depths 4-48 mm), checked at the pick (every IK solution tried) and from the pre-insertion pose
            inwards (the fingers, closed to the grasp's width, keep 1 mm from the arm, the links 3 mm)
  motion    every free-space move planned (`motion.Planner`: the Panda grown by 8 mm, the fingers 2 mm, the held units
            4 mm, against the bench, the fixture, the arms as they are, the units on the bench and the other Panda);
            every Cartesian move checked before it runs
  pick      pre-grasp, straight approach, close (15 N per finger) until both fingers touch, lift 60 mm; the insertion
            re-checked with the grasp as it is (a unit that moved in the fingers is put back and picked again)
  insert    closed loop on the privileged state: align at the pre-insertion pose, then along the insertion at 15 mm/s,
            the target re-computed every tick from the carrier's current pose; stopped when a joint of the arm turns
            1.5 deg (a jam pushing the unpowered arm), backed off and retried; held until the cell has the mate LOCKED
  retreat   open, back out along a checked direction, plan the way to the next pick.
A contact monitor logs every contact of a Panda with anything but the units it holds (`unplanned_contacts`).
`bench_plan(...)` computes the bench layout the program is tuned on (reverse assembly order, stable rests, a grasp per
unit that works at its pick and at its insertion).
"""
import json
import math
import os
import time

import numpy as np

READY = [0.0, -0.4, 0.0, -2.3, 0.0, 1.9, 0.785]
PARK = [0.0, -1.76, 0.0, -3.0, 0.0, 1.5, 0.785]
# between units a Panda waits folded back (READY, the IK's seed, puts the hand at the arms' shoulders for these
# Panda placements); a planned move only, never an unchecked one
HOME = PARK
INSERT_FINGER_CLEAR_M = 0.001
INSERT_LINK_CLEAR_M = 0.002
PRE_LINK_CLEAR_M = 0.004          # at the pre-insertion pose (the planner's 8 mm elsewhere)
# picking a flat part (the cable holders, 11-16 mm thick, the J2 motor by its 64 mm faces) the fingertips, 9 mm beyond
# the tcp, come down onto the bench: they may touch it (not go into it); the links keep 4 mm at the pick
PICK_BENCH_FINGER_CLEAR_M = 0.0
PICK_LINK_CLEAR_M = 0.004
PICK_OPEN_M = 0.012
PICK_OPENS = (0.025, 0.012, 0.006)  # the fingers come down open to the grasp's width + 25, 12 or 6 mm (the forearm's fork; the upper arm with HB)
PICK_FINGER_CLEAR_M = 0.0         # the open fingers against the unit itself, coming down around it: touching, not into it
GRASP_CAP = 150
# the fingers' tendon (half the width) at 2500 N/m, each finger with half the tendon's force: a command dw below the
# width presses each finger with 2500 x dw / 4 (15 N: 24 mm)
GRIP_N_PER_KG = 65.0              # a heavier unit is held harder: 65 N/kg per finger, 15-60 N (the upper arm, 0.92 kg: 60 N)
GRIP_MAX_N = 60.0
# the fingers hold a unit against turning about their closing axis only by the pads' torsional friction (0.01 m x 2 x
# the grip): the unit's weight may lever it about that axis by at most half of that (the upper arm, held 6 cm from its
# centre of mass, turned 25 deg in the fingers as it was lifted; 0.35 left no bench grasp of either forearm)
TORSION_M = 0.01
TORSION_SAFETY = 0.5
# the left upper arm, mounted with HB on its J4_D side: only grasps on the C3 end work (80 mm from its centre of mass)
TORSION_SAFETY_UNIT = {"l_UA": 0.7}
MIN_WIDTH_HEAVY_M = 0.030         # a unit over 0.5 kg is held across 30 mm or more (the 0.65 kg end effector, held
                                  # across a 22 mm feature 14 mm deep, turned out of the fingers on the way three times
                                  # and fell; held across 47 mm it went on)
MIN_DEPTH_HEAVY_M = 0.006         # (a 4 mm grasp left the 0.43 kg forearm with HA on the bench; the J2 motor in its
                                  # cradle is held by its faces at 3 mm and more: exempt)
ANTIPODAL_DEG = 15.0
ANTIPODAL_AXIS_DEG = 35.0
RELEASE_OPENS = (0.012, 0.006, 0.003)   # how far the fingers open to let go of a unit on the arm, first free
BEARING_TOL_DEG = 0.3            # a free bearing the mate fastens too (J4_D) off by more than this is pushed round first
BEARING_AIM_DEG = 0.1            # ... to within this
PRE_M = 0.04                      # the pre-insertion pose: this far back along the insertion
VIA_M = 0.06                      # the transport ends this much further back, then goes straight to the pre-insertion
HOLD_LOCK_TICKS = 1000
# a seat (the J2 motor in its yoke, the cable holders on their plates) is held until the cell has it ENGAGED (on its
# seat within 3 mm of home) and let go there, its fingers need only reach that far: the holders lie flush on J4_D /
# J4_E and fingers 21 mm wide wrapping their 11 mm edges would meet the arm in the last millimetre
SEAT_STOP_M = 0.0025            # a flange or seat held until LOCKED, at most 10 s
ARM_GUARD_DEG = 3.0
HOLD_GUARD_DEG = 6.0
# the units that rest on a riser block on the bench: the cable holders (11-16 mm thick: fingers wrapping their edges
# need room below them) and L2 (no grasp that works at its flange comes from above in any of its rests: the hand meets
# the bench)
RISER_UNITS = ("l_HA", "l_HB", "r_HA", "r_HB", "l_L2", "r_L2", "l_M2", "r_M2")
# the J2 motor (a 99 mm disc, 64 mm thick) can be held only across its faces: lying flat no finger gets under it, so
# it stands on its rim in the riser's two blocks (a cradle), its axis level
CRADLE_UNITS = ("l_M2", "r_M2")
# every unit onto the arm. The holder facing the other arm (the right arm's HA, the left arm's HB) cannot be reached
# once that arm's elbow is built (0 grasps, reach_kit.py), so the right arm comes first: its HA goes on with the left
# arm not built, the left arm's HB onto J4_D next to the finished right arm (2 grasps). HA on the forearm lying on the
# bench was tried: pushed from the side the loose forearm tipped up (5-17 mm short, FREE, 5 tries)
# The arm's motors J1-J3 are powered while the arm is built (the owner, 2026-10-27): each holds the angle it was
# fastened at (`POWER_HOLD`), so the hanging arm is not pushed away by the next unit going in (unpowered, the J2 motor's
# 0.2 N m let a finger's 1 N on J4_D turn the arm 3 deg and the forearm stopped 16 mm short); J4-J7 stay unpowered.
POWER_HOLD = {"j1": (40.0, 1.5, 10.0), "j2": (40.0, 1.5, 10.0), "j3": (25.0, 1.0, 5.0)}   # kp N m/rad, kd N m s/rad, max N m
# Order: the right arm first, all on the arm (its HA faces the left arm: no Panda reaches it once the left elbow is
# there); then the left arm, whose HB (facing the right arm: no Panda reaches its place on the arm now) goes onto J4_D
# in the air (the owner, 2026-10-28): the left Panda holds the upper arm, the right Panda puts HB on, the left Panda
# mounts both (`place_air`; no rest of the upper arm on the bench both takes HB and leaves a grasp that mounts it).
# (HB after the forearm on the arm: J4_D, free on its bearing until the forearm fastens it, turned 11 deg under HB going
# on, beyond the forearm's capture of it)
STEPS = [("flange_j1", "arm"), ("seat_M2", "arm"), ("flange_j2", "arm"), ("flange_j3", "arm"),
         ("flange_j4", "arm"), ("seat_HA", "arm"), ("seat_HB", "arm"), ("flange_j5", "arm"), ("flange_j7", "arm")]
HANDOVER = ()
ORDER = ([("right", m_, w_) for m_, w_ in STEPS]
         + [("left", "flange_j1", "arm"), ("left", "seat_M2", "arm"), ("left", "flange_j2", "arm"), ("left", "seat_HB", "air@other"),
            ("left", "flange_j3", "arm"), ("left", "flange_j4", "arm"), ("left", "seat_HA", "arm"), ("left", "flange_j5", "arm"),
            ("left", "flange_j7", "arm")])


def sequence(kit, pandas):
    """[(mate id, Panda name, where)] in the order the program places them"""
    arms = kit.get("arms", ["left"])
    out = []
    order = ORDER
    if os.environ.get("PANDA_ORDER"):
        # (development: e.g. "left:flange_j1:arm,left:seat_M2:arm,...")
        order = [tuple(x.split(":")) for x in os.environ["PANDA_ORDER"].split(",")]
    for side, mid, where in order:
        if side in arms:
            pn = [p["name"] for p in pandas if p.get("arm") == side][0]
            if where.endswith("@other"):
                pn = [p["name"] for p in pandas if p["name"] != pn][0]
                where = where.split("@")[0]
            out.append((side[0] + "_" + mid, pn, where))
    return out


def grip_force(mass):
    """the force per finger the hand holds a unit of `mass` kg with: a heavy unit (> 0.3 kg) with the full 60 N, the
    grasps being chosen for GRIP_N_PER_KG (the 0.65 kg end effector, held 80 mm from its centre of mass with 42 N, turned
    in the fingers on the way and fell)"""
    return GRIP_MAX_N if mass > 0.3 else min(GRIP_MAX_N, max(15.0, GRIP_N_PER_KG * mass))


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rotation_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Assembler(object):
    """the program's state on one cell: the Pandas' kinematics and planners, the units' geometry, the motion
    primitives. `run(seq)` executes a sequence; the bench plan uses the same checks on a probe cell."""

    def __init__(self, cell, log=print):
        import mujoco
        from ..assembly_full.panda_ik import PandaKin
        from .motion import Planner
        self.mj, self.cell, self.log = mujoco, cell, log
        m, d = cell.m, cell.d
        self.m, self.d, self.sm, self.kit = m, d, cell.sm, cell.kit
        self.kin, self.planner, self.ctrl, self.pc = {}, {}, {}, {}
        for pc in cell.pandas:
            n = pc["name"]
            self.kin[n] = PandaKin(model=m, prefix=n + "/", tcp_site=pc["tcp"])
            self.planner[n] = Planner(cell, n)
            self.ctrl[n] = cell.ctrls[n]
            self.pc[n] = pc
        self.cur = cell.pandas[0]["name"]
        self.mates = cell.mates
        self.unit_of = {bn: bb["unit"] for bn, bb in self.kit["bodies"].items()}
        self.col0 = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom)}
        # per unit: its rigid bodies (the root and the fixed children: motor cases, the bearing), the rest hinged
        self.rigid = {}
        for u in cell.loose:
            rb, todo = {u}, [u]
            while todo:
                x = todo.pop()
                for bn, bb in self.kit["bodies"].items():
                    if bb.get("parent") == x and bb.get("joint") is None:
                        rb.add(bn); todo.append(bn)
            self.rigid[u] = rb
        self.rigid_ids = {u: set(m.body(b).id for b in v) for u, v in self.rigid.items()}
        self.box = {u: self._box(u) for u in cell.loose}
        self.work = {"held": [], "phase": "start", "touch_ok": set()}
        self.contact_log = []
        self.why = {"last": None}
        cell.hooks.append(self.monitor)
        self.TIP = self.tip_extent()
        self.powered = True
        self.hold_q = {}
        self.graze_mm = 0.5            # how far a finger may graze a co-welded part (no grasp at 0.5: 0.8, see `place`)
        cell.hooks.insert(0, self.power_hold)

    # ------------------------------------------------------------------ geometry of the units
    def _box(self, u):
        """(lo, hi) of the unit's rigid collision geometry in its root frame (m)"""
        m, d = self.m, self.d
        Ti = _inv(self.cell.body_T(u))
        pts = []
        for g in self.cell.geoms_of[u]:
            if m.geom_bodyid[g] not in self.rigid_ids[u] or m.geom_type[g] != self.mj.mjtGeom.mjGEOM_MESH:
                continue
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pts.append(w @ Ti[:3, :3].T + Ti[:3, 3])
        P = np.concatenate(pts)
        return P.min(0), P.max(0)

    def collision_points(self, u):
        """the vertices of the unit's collision geoms (its hinged children at their current angles) in its root frame"""
        m, d = self.m, self.d
        Ti = _inv(self.cell.body_T(u))
        pts = []
        for g in self.cell.geoms_of[u]:
            if m.geom_type[g] != self.mj.mjtGeom.mjGEOM_MESH:
                continue
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            pts.append(w @ Ti[:3, :3].T + Ti[:3, 3])
        return np.concatenate(pts)

    def com_local(self, u):
        m, d = self.m, self.d
        mass, c = 0.0, np.zeros(3)
        for b in range(m.nbody):
            if self.cell.unit_of_body.get(b) == u:
                mass += m.body_mass[b]
                c += m.body_mass[b] * d.xipos[b]
        Ti = _inv(self.cell.body_T(u))
        return Ti[:3, :3] @ (c / mass) + Ti[:3, 3]

    def stable_rests(self, u, n=6, min_prob=0.02):
        """[(R, probability, hull vertices)]: the unit's stable resting orientations on a flat bench (the convex hull of
        its collision geometry, its centre of mass; trimesh.poses.compute_stable_poses)"""
        import trimesh
        mesh = trimesh.Trimesh(vertices=self.collision_points(u)).convex_hull
        Ts, probs = trimesh.poses.compute_stable_poses(mesh, center_mass=self.com_local(u), n_samples=1, threshold=0.02)
        hv = np.asarray(mesh.vertices, float)
        return [(np.asarray(T_, float)[:3, :3], float(p_), hv) for T_, p_ in list(zip(Ts, probs))[:n] if p_ >= min_prob]

    def group_of(self, u):
        """the units fastened to u through LOCKED mates (not through the base): they move with it"""
        out, todo = {u}, [u]
        while todo:
            x = todo.pop()
            for mm in self.kit["mates"]:
                if not self.sm.rec[mm["id"]]["fastened"]:
                    continue
                a_, b_ = self.unit_of[mm["carrier"]], self.unit_of[mm["part"]]
                for p_, q_ in ((a_, b_), (b_, a_)):
                    if p_ == x and q_ not in out and q_ != "base":
                        out.add(q_); todo.append(q_)
        return out

    # ------------------------------------------------------------------ the Panda in use
    def use(self, name):
        self.cur = name

    @property
    def K(self):
        return self.kin[self.cur]

    @property
    def C(self):
        return self.ctrl[self.cur]

    @property
    def PL(self):
        return self.planner[self.cur]

    def tcp_T(self):
        s = self.m.site(self.pc[self.cur]["tcp"]).id
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[s].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[s]
        return T

    def cmd_T(self):
        p0, R0 = self.C.fk_cmd()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R0, p0
        return T

    def tip_extent(self):
        """how far the closed fingertips reach beyond the tcp along the approach (m)"""
        m, d = self.m, self.d
        T = self.tcp_T()
        z = 0.0
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            if not bn.startswith(self.cur + "/") or "finger" not in bn or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        w = p + R @ (c + h * np.array([sx, sy, sz]))
                        z = max(z, float((w - T[:3, 3]) @ T[:3, 2]))
        return z

    # ------------------------------------------------------------------ debug snapshots (env PANDA_SNAP=<dir>)
    def snap(self, tag, unit=None):
        out = os.environ.get("PANDA_SNAP")
        if not out:
            return
        import mujoco
        from PIL import Image
        os.makedirs(out, exist_ok=True)
        if not hasattr(self, "_snap_r"):
            self._snap_r = mujoco.Renderer(self.m, 480, 640)
            self._snap_n = 0
        rr = self._snap_r
        imgs = []
        c_ = self.cell.d.xipos[self.m.body(unit).id] if unit else np.array([-0.2, 0.0, 0.3])
        for az, el, dist in ((0.0, -20.0, 0.35), (90.0, -15.0, 0.35), (200.0, -50.0, 0.4)):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = c_
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            rr.update_scene(self.d, cam)
            imgs.append(rr.render())
        self._snap_n += 1
        Image.fromarray(np.concatenate(imgs, axis=1)).save(os.path.join(out, "snap_%03d_%s.png" % (self._snap_n, tag)))

    # ------------------------------------------------------------------ the powered joints
    def power_hold(self):
        """J1-J3 of each arm, once fastened, hold the angle they were fastened at (PD on the rotor's hinge, the motor's
        torque limit); off when self.powered is False"""
        m, d = self.m, self.d
        for jn in self.sm.limits_set:
            key = jn.split("_")[-1]
            if key not in POWER_HOLD:
                continue
            ji = self.sm.joint_info[jn]
            j = ji["hinge"]
            dof = m.jnt_dofadr[j]
            if jn not in self.hold_q:
                self.hold_q[jn] = float(d.qpos[m.jnt_qposadr[j]])
            if not self.powered:
                d.qfrc_applied[dof] = 0.0
                continue
            kp, kd, tmax = POWER_HOLD[key]
            tau = kp * (self.hold_q[jn] - float(d.qpos[m.jnt_qposadr[j]])) - kd * float(d.qvel[dof])
            d.qfrc_applied[dof] = float(np.clip(tau, -tmax, tmax))

    # ------------------------------------------------------------------ monitor
    def monitor(self):
        m, d = self.m, self.d
        cell = self.cell
        f6 = np.zeros(6)
        held_b = set()
        for u in list(self.work["held"]) + list(self.work["touch_ok"]) + list(self.work.get("held_other", [])):
            held_b |= set(b for b in range(m.nbody) if cell.unit_of_body.get(b) == u)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            p1, p2 = b1 in cell.panda_bodies, b2 in cell.panda_bodies
            if p1 == p2:
                continue
            other = b2 if p1 else b1
            if other in held_b:
                continue
            if m.body(other).name == "world" and m.body(b1 if p1 else b2).name.endswith("link0"):
                continue
            self.mj.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) > 0.5:
                self.contact_log.append((round(float(d.time), 2), m.body(b1 if p1 else b2).name, m.body(other).name, round(abs(float(f6[0])), 1),
                                         self.work["phase"]))

    @staticmethod
    def contact_summary(cs):
        out = {}
        for t_, a_, b_, f_, ph in cs:
            o = out.setdefault((a_, b_, ph), {"a": a_, "b": b_, "phase": ph, "ticks": 0, "max_N": 0.0, "t0": t_, "t1": t_})
            o["ticks"] += 1
            o["max_N"] = max(o["max_N"], f_)
            o["t1"] = t_
        return sorted(out.values(), key=lambda o: -o["ticks"])

    # ------------------------------------------------------------------ motion primitives
    def wait_arrived(self, tol=0.004, max_ticks=300):
        C = self.C
        for _ in range(max_ticks):
            if float(np.max(np.abs(C.q() - C.q_goal))) < tol and float(np.max(np.abs(C.q_cmd - C.q_goal))) < 1e-6:
                return True
            self.cell.tick()
        return False

    def move_joint(self, q_t, speed=0.4):
        C = self.C
        q0 = np.array(C.q_cmd, float)
        q_t = np.asarray(q_t, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q_t - q0))) / (speed * self.cell.tick_s))))
        for i in range(1, n + 1):
            C.command_joint(q0 + (q_t - q0) * i / n)
            self.cell.tick()
        self.wait_arrived()

    def sync(self, held=(), allow=(), poses=None, width=None, touch=(), q_other=None):
        # a unit allowed or touched: its rigid bodies only; its hinged children (J4_D on its bearing, the rotors) are kept
        # clear (a finger on one turns it)
        def rigid(ns):
            return tuple(b for n in ns for b in (sorted(self.rigid[n]) if n in self.rigid else [n]))
        if q_other is None:
            q_other = self.work.get("q_other")         # (planning in the air: where the other Panda will be)
        self.PL.sync(held=held, allow=rigid(allow), poses=poses, width=width, touch=rigid(touch), q_other=q_other)

    def go(self, q_t, held=(), allow=(), speed=0.4, link_clear=None):
        """a planned free-space move to q_t; False if no path. A start inside the clearance (the hand just let go)
        first opens and backs straight up 50 mm."""
        self.sync(held=held, allow=allow)
        q_now = np.array(self.C.q_cmd, float)
        if self.PL.hits(q_now) and not held:
            # (the hand just let go, inside the clearance: back off 50 mm first, up or out of the grasp, checked against
            # penetration; never an unchecked move -- one pushed the hand through the arm with 150 N)
            T0 = self.cmd_T()
            for v in (np.array([0.0, 0.0, 1.0]), -T0[:3, 2], T0[:3, 0], -T0[:3, 0], T0[:3, 1], -T0[:3, 1]):
                T_up = T0.copy()
                T_up[:3, 3] += v * 0.05
                if self.move_line(T_up, speed=0.04, check=((), tuple(allow), 0.0, 3, 0.0, None, 0.0, ())):
                    break
            else:
                self.why["last"] = "stuck: no free way to back off (%s)" % self.why["last"]
                return False
            self.sync(held=held, allow=allow)
            q_now = np.array(self.C.q_cmd, float)
        clear0 = self.PL.clear
        if link_clear is not None:
            self.PL.clear = link_clear                  # (near the arm: the pre-insertion's 4 mm)
        try:
            # the fingers 8 mm clear of everything first: at 0.6 rad/s a Panda lags its joint path by millimetres at the
            # fingertips (planned 2 mm clear, the right hand going home past the upper arm's HB struck it with 31 N,
            # over the cell's 30 N limit); 2 mm only where nothing else is free
            # (tried only from and to postures that clear, and briefly: the wall clock)
            path = None
            if not any(self.PL.hits(q_, finger_clear=0.008) for q_ in (q_now, np.asarray(q_t, float))):
                self.PL.finger_clear = 0.008
                path = self.PL.plan(q_now, np.asarray(q_t, float), max_iter=500)
            if path is None:
                self.PL.finger_clear = 0.002
                path = self.PL.plan(q_now, np.asarray(q_t, float))
        finally:
            self.PL.clear = clear0
            self.PL.finger_clear = 0.002
        if path is None:
            hs, h0 = self.PL.hits(np.asarray(q_t, float)), self.PL.hits(q_now)
            self.why["last"] = "goal: %s; start: %s" % (sorted(set((h[0], h[1], round(h[2] * 1000, 1)) for h in hs))[:6],
                                                       sorted(set((h[0], h[1], round(h[2] * 1000, 1)) for h in h0))[:6])
            return False
        for q in path[1:]:
            self.move_joint(q, speed=speed)
        return True

    def go_home(self):
        if not self.go(HOME, speed=0.6):
            self.log("[panda]     %s: no path home (%s)" % (self.cur, self.why["last"]))
            return False
        return True

    def line_qs(self, T1, speed=0.03, ang_speed=math.radians(15.0), T0=None):
        from scipy.spatial.transform import Rotation, Slerp
        if T0 is None:
            T0 = self.cmd_T()
        dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        ang = rotation_angle(T0[:3, :3].T @ T1[:3, :3])
        dur = max(dist / speed, ang / ang_speed, 2 * self.cell.tick_s)
        n = max(1, int(math.ceil(dur / self.cell.tick_s)))
        sl = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
        q = np.array(self.C.q_cmd, float)
        qs, ok_all = [], True
        for i in range(1, n + 1):
            s_ = i / n
            q, ok = self.K.solve_one((1 - s_) * T0[:3, 3] + s_ * T1[:3, 3], sl([s_]).as_matrix()[0], q, q, margin=0.0)
            ok_all = ok_all and ok
            qs.append(q.copy())
        return qs, ok_all

    def line_free(self, qs, held=(), allow=(), held_clear=None, every=3, link_clear=None, world_finger_clear=None, finger_clear=0.002, touch=()):
        self.sync(held=held, allow=allow, touch=touch)
        sub = qs[::every] + qs[-1:]
        kw = dict(held_clear=held_clear, link_clear=link_clear, world_finger_clear=world_finger_clear, finger_clear=finger_clear)
        h0 = {(h[0], h[1]): h[2] for h in self.PL.hits(np.array(self.C.q_cmd, float), **kw)}
        for i_, q in enumerate(sub):
            hs = [h for h in self.PL.hits(q, **kw) if not ((h[0], h[1]) in h0 and h[2] >= h0[(h[0], h[1])] - 0.0005)]
            if hs:
                self.why["last"] = "at %d / %d: %s" % (i_ * every, len(qs), sorted(set((h[0], h[1], round(h[2] * 1000, 1)) for h in hs))[:6])
                return False
        return True

    def run_qs(self, qs):
        for q in qs:
            self.C.command_joint(q)
            self.cell.tick()
        self.wait_arrived()

    def move_line(self, T1, speed=0.03, ang_speed=math.radians(15.0), check=None):
        qs, ok = self.line_qs(T1, speed=speed, ang_speed=ang_speed)
        if check is not None and not self.line_free(qs, *check):
            return False
        self.run_qs(qs)
        return ok

    def held_spec(self, u):
        """[(unit, G)] of the unit's group as it sits in the hand now"""
        T_t = self.tcp_T()
        return [(x, _inv(T_t) @ self.cell.body_T(x)) for x in sorted(self.group_of(u))]

    # ------------------------------------------------------------------ targets
    def target(self, mid):
        """the live target: {"T": the unit's root pose at home, "ins": the insertion (world), "carrier", "unit"}"""
        tg = self.cell.live_target(mid)
        mm = self.mates[mid]
        Rc = self.cell.body_T(mm["carrier"])[:3, :3]
        tg["ins"] = Rc @ np.asarray(mm["insertion_in_carrier"], float)
        tg["ins"] /= np.linalg.norm(tg["ins"])
        return tg

    def remaining(self, mid):
        tg = self.target(mid)
        dv = tg["T"][:3, 3] - self.cell.body_T(tg["unit"])[:3, 3]
        ax = float(dv @ tg["ins"])
        return ax, float(np.linalg.norm(dv - ax * tg["ins"]))

    def arm_joints(self, mid):
        """the hinge angles of the arm's fastened joints and J4_D's bearing (the side of the mate)"""
        m, d = self.m, self.d
        P = mid[:2] if mid[1] == "_" else ""
        out = []
        for jn in sorted(self.sm.limits_set):
            if jn.startswith(P):
                out.append(float(d.qpos[m.jnt_qposadr[self.sm.joint_info[jn]["hinge"]]]))
        return np.array(out)

    # ------------------------------------------------------------------ grasps
    def part_ray(self, u, data, with_units=()):
        m = self.m
        ok_b = set(self.rigid_ids[u]).union(*[self.rigid_ids[x] for x in with_units])

        def ray(o, v):
            geomid = np.zeros(1, np.int32)
            for _ in range(60):                              # (through the other parts' convex pieces on the way)
                dist = self.mj.mj_ray(m, data, o, v, None, 1, -1, geomid)
                if dist < 0:
                    return None
                g = int(geomid[0])
                if m.geom_bodyid[g] in ok_b and self.col0.get(g, (0, 0)) != (0, 0):
                    return o + v * dist
                o = o + v * (dist + 1e-4)
            return None
        return ray

    def grasp_candidates(self, u, T_part, cap=2 * GRASP_CAP, toward=None, with_units=()):
        """[(G = tcp pose in the unit's root frame, width m, desc)]: per closing axis and approach axis of the unit's box
        (and the closing axis turned +-30 deg about the approach), approach depths 4-48 mm, offsets across; two antipodal
        pads (>= 3 of 5 ray pairs, widths within 6 mm), width 4-75 mm; the `cap` best by width and nearness to the
        centre of mass. Rays on the unit's rigid bodies only (a hinged child would swing in the fingers)."""
        from .coupling import quat_wxyz
        m = self.m
        K = self.K
        K.d.qpos[:] = self.d.qpos
        a = m.jnt_qposadr[m.joint("free_" + u).id]
        K.d.qpos[a:a + 3] = T_part[:3, 3]
        K.d.qpos[a + 3:a + 7] = quat_wxyz(T_part[:3, :3])
        self.mj.mj_kinematics(m, K.d)
        ray = self.part_ray(u, K.d, with_units)
        lo, hi = self.box[u]
        R = T_part[:3, :3]
        com = self.com_local(u)
        mass = sum(float(m.body_mass[b]) for b in range(m.nbody) if self.cell.unit_of_body.get(b) == u)
        heavy = mass > 0.3
        f_grip = min(GRIP_MAX_N, max(15.0, GRIP_N_PER_KG * mass))
        # the longest lever of the unit's weight about the closing axis the pads hold
        ts_ = dict(TORSION_SAFETY_UNIT)
        ts_.update(dict(x.split(":") for x in os.environ.get("PANDA_TORSION_UNITS", "").split(",") if x))   # (development)
        arm_max = float(ts_.get(u, TORSION_SAFETY)) * TORSION_M * 2.0 * f_grip / (mass * 9.81)
        out = []
        for ci in range(3):
            for ai in range(3):
                if ai == ci:
                    continue
                bi = 3 - ci - ai
                for sgn in (1.0, -1.0):
                    a_loc = np.zeros(3)
                    a_loc[ai] = sgn
                    ext_a = hi[ai] - lo[ai]
                    for depth in (0.003, 0.004, 0.006, 0.010, 0.014, 0.020, 0.028, 0.036, 0.048):
                        if depth > ext_a - 0.003 or (heavy and depth < MIN_DEPTH_HEAVY_M and u not in CRADLE_UNITS):
                            continue
                        for fb in (-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
                            for rot in [math.radians(r_) for r_ in (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0)]:
                                g = 0.5 * (lo + hi)
                                g[bi] += fb * (hi[bi] - lo[bi])
                                g[ai] = (lo[ai] + depth) if sgn > 0 else (hi[ai] - depth)
                                c0, b0 = np.zeros(3), np.zeros(3)
                                c0[ci], b0[bi] = 1.0, 1.0
                                c_loc = math.cos(rot) * c0 + math.sin(rot) * b0
                                b_loc = np.cross(a_loc, c_loc)
                                cw = R @ c_loc
                                hits = []
                                for off_a, off_b in ((0, 0), (-0.005, 0), (0.005, 0), (0, -0.005), (0, 0.005)):
                                    gg = g + a_loc * off_a + b_loc * off_b
                                    gw = R @ gg + T_part[:3, 3]
                                    p1 = ray(gw - cw * 0.15, cw)
                                    p2 = ray(gw + cw * 0.15, -cw)
                                    if p1 is not None and p2 is not None:
                                        hits.append((p1, p2))
                                if len(hits) < 3:
                                    continue
                                ws = sorted(float((p2 - p1) @ cw) for p1, p2 in hits)
                                w = ws[len(ws) // 2]
                                if w > 0.075 or w < 0.004 or ws[-1] - ws[0] > 0.006:
                                    continue
                                if mass > 0.5 and w < MIN_WIDTH_HEAVY_M:
                                    continue
                                flat = True
                                for side in (0, 1):
                                    pts = np.array([h_[side] for h_ in hits])
                                    _u, _s, vt = np.linalg.svd(pts - pts.mean(axis=0))
                                    lim = ANTIPODAL_DEG if rot != 0.0 else ANTIPODAL_AXIS_DEG
                                    if _s[1] > 1e-4 and abs(float(vt[2] @ cw)) < math.cos(math.radians(lim)):
                                        flat = False
                                if not flat:
                                    continue
                                centre_w = np.mean([0.5 * (p1 + p2) for p1, p2 in hits], axis=0)
                                centre = R.T @ (centre_w - T_part[:3, 3])
                                r_ = com - centre
                                if float(np.linalg.norm(r_ - (r_ @ c_loc) * c_loc)) > arm_max:
                                    continue
                                z, y = a_loc, c_loc
                                x = np.cross(y, z)
                                G = np.eye(4)
                                G[:3, :3] = np.column_stack([x, y, z])
                                G[:3, 3] = centre
                                if any(np.linalg.norm(G[:3, 3] - o[0][:3, 3]) < 0.003 and rotation_angle(G[:3, :3].T @ o[0][:3, :3]) < 0.2 for o in out):
                                    continue
                                desc = "close %s%s approach %s%s depth %.0f off %+.1f" % (
                                    "xyz"[ci], "" if rot == 0 else " %+.0f" % math.degrees(rot), "+-"[sgn < 0], "xyz"[ai], depth * 1000, fb)
                                out.append((G, w, desc))
                                # the same grasp with the hand turned half way about the approach (the fingers are
                                # symmetric; the wrist's joint 7 range decides which of the two the Panda reaches)
                                Gf = G.copy()
                                Gf[:3, 0], Gf[:3, 1] = -G[:3, 0], -G[:3, 1]
                                out.append((Gf, w, desc + " flipped"))
        out.sort(key=lambda o: (0.0 if 0.015 <= o[1] <= 0.065 else 0.03) + float(np.linalg.norm(o[0][:3, 3] - com)))
        if toward is not None:
            # the grasps approaching along `toward` (world; a seat's insertion: the hand behind the unit) first
            out.sort(key=lambda o: 0 if float((R @ o[0][:3, 2]) @ toward) > 0.7 else 1)
        return out[:cap]

    # ------------------------------------------------------------------ checks
    def collides(self, q, held=(), allow=(), poses=None, width=None, finger_clear=0.002, link_clear=None, touch=()):
        self.sync(held=held, allow=allow, poses=poses, width=width, touch=touch)
        return [(h[1], round(h[2] * 1000, 2)) for h in self.PL.hits(np.asarray(q, float), finger_clear=finger_clear, link_clear=link_clear)]

    def fast_ik(self, T, seeds=None):
        from ..assembly_full.panda_ik import REFERENCE_POSTURES
        for seed in (seeds or (READY, REFERENCE_POSTURES[0], REFERENCE_POSTURES[1], REFERENCE_POSTURES[3])):
            q, ok = self.K.solve_one(T[:3, 3], T[:3, :3], np.asarray(seed, float), np.asarray(seed, float))
            if ok:
                return q, self.K.margin(q)
        return None, 0.0

    def place_checks(self, mid, tg, G, group, poses=None, width=None, fast=False, rel=None):
        """(q_pre, margin) when the pre-insertion pose is reachable with this grasp and the Panda keeps clear of the
        arm there and all the way in (the fingers 1 mm, the links 3 mm; 2 mm steps); else (None, reason).
        `group` the units the hand carries (the unit and those fastened to it), `rel` their poses in the unit's root
        frame (default: as they are now), `poses` a hypothetical scene."""
        u = tg["unit"]
        T_end = tg["T"]
        if rel is None:
            rel = {x: _inv(self.cell.body_T(u)) @ self.cell.body_T(x) for x in group}
        T_pre = T_end.copy()
        T_pre[:3, 3] -= tg["ins"] * PRE_M
        held_names = (u,)
        # the rest of the group (the forearm hanging on the upper arm) and the parts the mate fastens besides its own
        # pair (J4_D, on its bearing, with the forearm) may be touched by the fingers, not gone into (a forearm grasp
        # whose finger went 1-2 mm into J4_D pressed it with 8-16 N on the way in and turned the arm 3 deg; with J4_D
        # untouchable no grasp of the forearm both works going in and can be picked)
        cw_parts = tuple(c["part"] for c in self.mates[mid].get("co_welds", []))
        touch_g = tuple(x for x in group if x != u) + cw_parts
        # (and the carrier, on the powered arm: the left upper arm, turned 10 deg in the fingers in the lift, met L2
        # with a finger 2 mm before home)
        cw_parts = cw_parts + (self.mates[mid]["carrier"], self.unit_of[self.mates[mid]["carrier"]])

        def graze(h):
            # (a co-welded part may be grazed: 0.5 mm into it at most -- no grasp of the forearm keeps strictly off
            # J4_D and can be picked from the bench; 1-2 mm into it pressed it with 8-16 N and turned the arm 3 deg)
            return [x_ for x_ in h if not (x_[0] in cw_parts and x_[1] >= -self.graze_mm)]

        def scene(T_u):
            p = dict(poses or {})
            for x in group:
                p[x] = T_u @ rel[x]
            return p

        def along(q_pre):
            """None when the Panda keeps clear from q_pre all the way in, else the reason"""
            h = graze(self.collides(q_pre, allow=held_names, poses=scene(T_pre), width=width, link_clear=PRE_LINK_CLEAR_M, touch=touch_g))
            if h:
                return "Panda hits %s at the pre-insertion" % sorted(set(x[0] for x in h))[:4]
            q_x = q_pre
            x_stop = SEAT_STOP_M if self.mates[mid]["kind"] == "seat" else 0.0
            for x in list(np.arange(PRE_M - 0.002, x_stop + 1e-9, -0.002)) + [x_stop]:
                T_x = T_end.copy()
                T_x[:3, 3] -= tg["ins"] * x
                q_x2, ok = self.K.solve_one((T_x @ G)[:3, 3], (T_x @ G)[:3, :3], q_x, q_x, margin=0.0)
                if not ok:
                    return "insertion IK at %.0f mm" % (x * 1000)
                h = graze(self.collides(q_x2, allow=held_names, poses=scene(T_x), width=width, finger_clear=INSERT_FINGER_CLEAR_M, link_clear=INSERT_LINK_CLEAR_M, touch=touch_g))
                if h:
                    return "Panda hits %s going in at %.0f mm" % (sorted(set(x_[0] for x_ in h))[:4], x * 1000)
                q_x = q_x2
            if width is not None:
                # the release: the fingers open 12, 6 or 3 mm (the first that keeps them off everything but the unit)
                why_r = None
                for op in RELEASE_OPENS:
                    h = graze(self.collides(q_x, allow=held_names, poses=scene(T_end), width=min(0.08, width + op), finger_clear=0.0,
                                            link_clear=INSERT_LINK_CLEAR_M, touch=(u,) + touch_g))
                    if h:
                        why_r = why_r or "the fingers opening at the end hit %s" % sorted(set(x[0] for x in h))[:4]
                        continue
                    # and a way back from there: 60 mm straight out of the grasp, up, or back along the insertion,
                    # the open fingers touching the unit at most
                    d_ = self.retreat_dir(q_x, tg["ins"], held_names, scene(T_end), min(0.08, width + op), (u,) + touch_g, graze)
                    if d_ is None:
                        why_r = "no way back after the release (open %.0f mm)" % (op * 1000)
                        continue
                    self.last_release_open, self.last_release_dir = op, d_
                    break
                else:
                    return why_r
            return None
        # the arm's posture decides what it touches: the quick IK's solution first, then up to five others
        q_f, m_f = self.fast_ik(T_pre @ G)
        sols = [(q_f, m_f)] if q_f is not None else []
        why = "pre-insertion IK"
        tried = 0
        for q_pre, mrg in sols:
            r_ = along(q_pre)
            if r_ is None:
                return q_pre, min(mrg, self.K.margin(q_pre))
            why = r_
        if fast and not why.startswith("Panda hits") and sols:
            return None, why
        for _sc, q_pre, mrg in self.K.candidates(T_pre @ G, np.array(READY)):
            if q_f is not None and float(np.max(np.abs(q_pre - q_f))) < 0.05:
                continue
            tried += 1
            if tried > 5:
                break
            r_ = along(q_pre)
            if r_ is None:
                return q_pre, min(mrg, self.K.margin(q_pre))
            why = r_
        return None, why

    def pick_checks(self, u, T_pick, G, poses=None, w_open=None, fast=False, touch_units=None):
        """(q_pick, q_pg, margin) for a grasp of the unit where it rests: never from below, a pre-grasp 80 mm back along
        the approach; the Panda (fingers open to the grasp's width + 25 mm) clear of everything but the unit"""
        P = T_pick @ G
        if P[2, 2] > 0.2:
            return None, None, "approach from below"
        P_pg = P.copy()
        P_pg[:3, 3] -= P[:3, 2] * 0.08
        sols = []
        q_f, m_f = self.fast_ik(P)
        if q_f is not None:
            sols.append((q_f, m_f))
        if not fast:
            sols += [(c_[1], c_[2]) for c_ in self.K.candidates(P, np.array(READY))]
        if not sols:
            return None, None, "pick IK"
        ps = dict(poses or {})
        ps[u] = T_pick
        why_ = "pre-grasp IK"
        for q_pick, mrg in sols[:4]:
            q_pg, ok = self.K.solve_one(P_pg[:3, 3], P_pg[:3, :3], q_pick, q_pick, margin=0.0)
            if not ok:
                continue
            # the unit is an obstacle too: the fingers, open to the grasp's width + 25 mm (else + 12 mm), come down
            # around it (the hand may touch it, not go into it)
            for op in PICK_OPENS:
                width = min(0.08, w_open + op) if w_open else None
                self.sync(allow=(), poses=ps, width=width, touch=tuple(touch_units or (u,)))
                hs = []
                for k in range(6):
                    q_k, ok = self.K.solve_one(((1 - k / 5) * P_pg[:3, 3] + k / 5 * P[:3, 3]), P[:3, :3], q_pg, q_pg, margin=0.0)
                    hs = self.PL.hits(q_k, world_finger_clear=PICK_BENCH_FINGER_CLEAR_M, link_clear=PICK_LINK_CLEAR_M, finger_clear=PICK_FINGER_CLEAR_M)
                    if hs:
                        break
                if not hs:
                    self.last_pick_open = op
                    return q_pick, q_pg, mrg
                why_ = "Panda hits %s at the pick (%s; step %d of 5 down, open +%.0f mm)" % (
                    sorted(set(h[1] for h in hs))[:4], sorted(set((h[0], h[1], round(h[2] * 1000, 1)) for h in hs))[:3], k, op * 1000)
        return None, None, why_

    def plan_grasps(self, mid, tg, pinned=None):
        """feasible grasps of the unit where it rests now, best first: [(score, G, w, desc, q_pick, q_pg)]"""
        u = tg["unit"]
        T_pick = self.cell.body_T(u)
        cands = self.grasp_candidates(u, T_pick, toward=tg["ins"] if self.mates[mid]["kind"] == "seat" else None)
        if pinned is not None and (pinned["desc"] in [c_[2] for c_ in cands] or u in CRADLE_UNITS):
            # (a bench plan's grasp the candidate rules now exclude is not tried: e.g. one too shallow for the unit's weight)
            cands = [(np.asarray(pinned["G"], float), float(pinned["w"]), pinned["desc"] + " (bench plan)")] + cands
        group = sorted(self.group_of(u))
        com = self.com_local(u)
        feasible, rejected = [], {}
        for G, w, desc in cands:
            q_pick, q_pg, mrg1 = self.pick_checks(u, T_pick, G, w_open=w, touch_units=tuple(group))
            if q_pick is None:
                rejected[desc] = mrg1
                continue
            q_pre, mrg2 = self.place_checks(mid, tg, G, group, width=w)
            if q_pre is None:
                rejected[desc] = mrg2
                continue
            score = min(mrg1, mrg2) - 2.0 * float(np.linalg.norm(G[:3, 3] - com)) - (0.3 if w < 0.012 else 0.0)
            feasible.append((score, G, w, desc, q_pick, q_pg))
            if len(feasible) >= 8:
                break
        feasible.sort(key=lambda x: -x[0])
        return feasible, len(cands), rejected

    # ------------------------------------------------------------------ one mate
    def insert(self, mid, x0, speed=0.015, guard_deg=ARM_GUARD_DEG, powered_guard_deg=None):
        """the held unit from x0 before home along the insertion to home, the target re-computed every tick from the
        carrier's current pose, the hand corrected for the unit's pose in it and the controller's lag; stops when a
        joint of the arm turns guard_deg (a powered one, J1-J3, powered_guard_deg if given), or when the mate is
        ENGAGED"""
        tg = self.target(mid)
        u = tg["unit"]
        T_part, T_act, T_cmd = self.cell.body_T(u), self.tcp_T(), self.cmd_T()
        G_act, lag = _inv(T_part) @ T_act, _inv(T_act) @ T_cmd
        x = x0
        q = np.array(self.C.q_cmd, float)
        qa0 = self.arm_joints(mid)
        P_ = mid[:2] if mid[1] == "_" else ""
        powered = np.array([jn[-3:] in ("_j1", "_j2", "_j3") for jn in sorted(self.sm.limits_set) if jn.startswith(P_)], bool)
        scale = np.where(powered, guard_deg / (powered_guard_deg or guard_deg), 1.0)
        moved = 0.0
        m_, d_ = self.m, self.d
        ub = set(b for b in range(m_.nbody) if self.cell.unit_of_body.get(b) == u)
        pb = self.cell.panda_bodies_of[self.cur]
        f6 = np.zeros(6)
        cmax = {}

        def contacts_now():
            for i_ in range(d_.ncon):
                c_ = d_.contact[i_]
                b1, b2 = m_.geom_bodyid[c_.geom1], m_.geom_bodyid[c_.geom2]
                if (b1 in ub or b1 in pb) != (b2 in ub or b2 in pb):
                    self.mj.mj_contactForce(m_, d_, i_, f6)
                    k_ = (m_.body(b1).name, m_.body(b2).name)
                    cmax[k_] = max(cmax.get(k_, 0.0), abs(float(f6[0])))
        self._insert_contacts = cmax
        while x > 0.0:
            contacts_now()
            if len(qa0):
                dq_ = np.abs(self.arm_joints(mid) - qa0)
                moved = math.degrees(float(np.max(dq_)))
                if math.degrees(float(np.max(dq_ * scale))) > guard_deg:
                    m_, d_ = self.m, self.d
                    ub = set(b for b in range(m_.nbody) if self.cell.unit_of_body.get(b) == u)
                    cs = {}
                    f6 = np.zeros(6)
                    pb = self.cell.panda_bodies_of[self.cur]
                    for i_ in range(d_.ncon):
                        c_ = d_.contact[i_]
                        b1, b2 = m_.geom_bodyid[c_.geom1], m_.geom_bodyid[c_.geom2]
                        mine1, mine2 = (b1 in ub or b1 in pb), (b2 in ub or b2 in pb)
                        if mine1 != mine2:
                            self.mj.mj_contactForce(m_, d_, i_, f6)
                            k_ = (m_.body(b1).name, m_.body(b2).name)
                            cs[k_] = max(cs.get(k_, 0.0), abs(float(f6[0])))
                    self.snap("%s_guard" % mid, u)
                    P_ = mid[:2] if mid[1] == "_" else ""
                    jn_ = [jn for jn in sorted(self.sm.limits_set) if jn.startswith(P_)]
                    self.log("[panda]     joints moved (deg): %s" % {jn: round(math.degrees(float(v)), 2) for jn, v in zip(jn_, self.arm_joints(mid) - qa0)})
                    self.log("[panda]     insert %s: the arm moved %.1f deg at %.0f mm short, stop; contacts of the hand / unit with the rest (N) now: %s; most on the way in: %s" % (
                        mid, moved, x * 1000, sorted(((round(v, 1), k_) for k_, v in cs.items()), reverse=True)[:6],
                        sorted(((round(v, 1), k_) for k_, v in cmax.items()), reverse=True)[:8]))
                    return x, moved
            x = max(0.0, x - speed * self.cell.tick_s)
            tg_ = self.target(mid)
            T_des = tg_["T"].copy()
            T_des[:3, 3] -= tg_["ins"] * x
            T_c = T_des @ G_act @ lag
            q, _ok = self.K.solve_one(T_c[:3, 3], T_c[:3, :3], q, q, margin=0.0)
            self.C.command_joint(q)
            self.cell.tick()
            if self.sm.state[mid] in ("ENGAGED", "HOLDING", "LOCKED"):
                break
        return x, moved

    def wait_arm_rest(self, mid, max_s=4.0, tol_deg_s=0.2):
        """hold the hand still until the arm's fastened joints turn slower than tol_deg_s (at most max_s); seconds waited"""
        q0 = self.arm_joints(mid)
        if not len(q0):
            return 0.0
        n = int(max_s / (5 * self.cell.tick_s))
        for i in range(n):
            self.cell.tick(5)
            q1 = self.arm_joints(mid)
            if math.degrees(float(np.max(np.abs(q1 - q0)))) / (5 * self.cell.tick_s) < tol_deg_s:
                return round((i + 1) * 5 * self.cell.tick_s, 2)
            q0 = q1
        return max_s

    def hold_until_locked(self, mid, ticks=HOLD_LOCK_TICKS):
        """keep the hand on the unit's home (re-computed from the carrier) until the mate is LOCKED; given up when the
        arm turns ARM_GUARD_DEG (the hand pushing the unpowered arm) or the mate leaves its capture"""
        tg = self.target(mid)
        u = tg["unit"]
        G_act = _inv(self.cell.body_T(u)) @ self.tcp_T()
        lag = _inv(self.tcp_T()) @ self.cmd_T()
        q = np.array(self.C.q_cmd, float)
        qa0 = self.arm_joints(mid)
        for i in range(ticks):
            if self.sm.rec[mid]["fastened"]:
                return True
            if self.sm.state[mid] not in ("ENGAGED", "HOLDING"):
                self.log("[panda]     hold %s: %s, given up" % (mid, self.sm.state[mid]))
                return False
            # (no guard on the arm's joints here: the unit is on its feature, the capture holds it and the arm swings
            # with it as it settles; a guard at 1.5 / 6 deg gave up on L2 every time, which then fastened on its own)
            if i % 10 == 0:
                tg_ = self.target(mid)
                T_c = tg_["T"] @ G_act @ lag
                q, _ok = self.K.solve_one(T_c[:3, 3], T_c[:3, :3], q, q, margin=0.0)
                self.C.command_joint(q)
            self.cell.tick()
        return bool(self.sm.rec[mid]["fastened"])

    def retreat_dir(self, q, ins, allow, poses, width, touch, graze=lambda h: h):
        """the first of (out of the grasp, up, back along the insertion) the hand can go 60 mm from posture q (5 mm
        steps; the fingers touching `touch` at most, 0 mm off the rest, the links 2 mm): its index, or None"""
        p0, R0 = self.K.fk(np.asarray(q, float))
        for i_, v in enumerate((-R0[:, 2], np.array([0.0, 0.0, 1.0]), -np.asarray(ins, float))):
            q_k, ok = np.asarray(q, float), True
            for k in range(1, 13):
                q_k, ok = self.K.solve_one(p0 + v * 0.005 * k, R0, q_k, q_k, margin=0.0)
                if not ok or graze(self.collides(q_k, allow=allow, poses=poses, width=width, finger_clear=0.0, link_clear=INSERT_LINK_CLEAR_M, touch=touch)):
                    ok = False
                    break
            if ok:
                return i_
        return None

    def retreat(self, mid, allow=()):
        self.work["phase"] = "release"
        # open as far as the place checks found free (12, 6 or 3 mm: the fingers off the unit, not swung into its
        # neighbours), open fully once backed off
        self.C.command_width(min(0.08, self.C.width() + self.work.get("release_open", RELEASE_OPENS[0])))
        self.cell.tick(40)
        self.work["held"] = []
        self.work["touch_ok"] = set()
        self.work["phase"] = "retreat"
        T = self.cmd_T()
        u = self.mates[mid]["unit"]
        # the unit let go of is an obstacle now (on the arm, or on its carrier); a contact present at the start may stay
        # while it does not get closer (line_free)
        vs = [-T[:3, 2], np.array([0.0, 0.0, 1.0]), -self.target(mid)["ins"]]
        k0 = self.work.pop("release_dir", None)
        if k0 is not None:
            vs = [vs[k0]] + [v_ for i_, v_ in enumerate(vs) if i_ != k0]
        # the hand next to the unit it let go of may touch it, and the parts the mate fastened with it (J4_D) and its
        # carrier, not go into them; the rest not touched
        tch = (u,) + tuple(c["part"] for c in self.mates[mid].get("co_welds", [])) + (self.unit_of[self.mates[mid]["carrier"]],)
        w0 = self.C.width()
        for op in (0.0, 0.006, 0.012, 0.025):
            if op > 0.0:
                # (no way back at this opening: the left forearm's finger, open 6 mm, dug 1.6 mm into it backing off)
                # wider, if the fingers opening further touch nothing but those
                self.sync(allow=tuple(allow), width=min(0.08, w0 + op), touch=tch)
                if self.PL.hits(np.array(self.C.q_cmd, float), finger_clear=0.0, link_clear=0.0):
                    continue
                self.C.command_width(min(0.08, w0 + op))
                self.cell.tick(30)
            for dist in (0.06, 0.03):
                for v in vs:
                    T_b = T.copy()
                    T_b[:3, 3] += v * dist
                    if self.move_line(T_b, speed=0.05, check=((), tuple(allow), None, 3, 0.002, None, 0.0, tch)):
                        self.C.command_width(0.08)
                        return True
        self.log("[panda]     %s: no free way back from the unit (%s)" % (mid, self.why["last"]))
        return False

    def put_back(self, q_pick, u):
        """carry the unit back to where it was picked and let go"""
        self.work["phase"] = "put back"
        hs = self.held_spec(u)
        if not self.go(q_pick, held=hs, allow=("world",)):
            # never let go in the air: the unit stays in the hand
            self.log("[panda]     put back %s: no way back to where it was picked (%s); still held" % (u, self.why["last"]))
            return False
        self.C.command_width(0.08)
        self.cell.tick(40)
        self.work["held"] = []
        T_b = self.cmd_T()
        T_b[:3, 3] -= T_b[:3, 2] * 0.06
        self.move_line(T_b, speed=0.04)

    def place(self, mid, pinned=None):
        """one mate by the Panda in use: plan, pick, transport, insert, hold until LOCKED, release. Returns the record."""
        cell, sm, C = self.cell, self.sm, self.C
        tg = self.target(mid)
        u = tg["unit"]
        rec = {"unit": u, "panda": self.cur, "placed_by": None}
        t0 = float(self.d.time)
        missing = [r_ for r_ in self.mates[mid]["requires"] if not sm.rec[r_]["achieved"]]
        if missing:
            rec.update(placed_by="nobody", skipped="needs %s" % missing)
            return rec
        if self.mates[mid].get("co_welds"):
            # a free bearing the mate also fastens (J4_D in the upper arm, turned when the upper arm was handled):
            # a fingertip pushes it round to the forearm's angle first
            rec["bearing"] = self.align_bearing(mid)
            tg = self.target(mid)
        t_plan = time.time()
        feasible, ncand, rejected = self.plan_grasps(mid, tg, pinned)
        if not feasible and self.mates[mid].get("co_welds"):
            # (the left forearm, planned grazing J4_D by 0.5 mm, went further into it on the built arm; 0.8 mm for
            # every forearm let a finger press J4_D with 8 N and turn the arm 3 deg: only when nothing grazes less)
            self.graze_mm = 0.8                        # (for this mate's attempts; back to 0.5 below)
            feasible, ncand, rejected = self.plan_grasps(mid, tg, pinned)
            rec["graze_mm"] = 0.8
        rec.update(grasp_candidates=ncand, grasps_feasible=len(feasible), plan_wall_s=round(time.time() - t_plan, 1),
                   rejected=dict(list(rejected.items())[:30]))
        if not feasible:
            self.log("[panda] %s: no feasible grasp (%d candidates): %s" % (mid, ncand, list(rejected.items())[:6]))
        n_c0 = len(self.contact_log)
        attempts, ok = [], False
        tries = [(f_, 0.3) for f_ in feasible[:5]] + [(f_, 0.1) for f_ in feasible[:2]]
        for (_s, G, w, desc, q_pick, q_pg), v_tr in tries:
            if ok:
                break
            att = {"grasp": desc, "width_mm": round(w * 1000, 1), "transport_rad_s": v_tr}
            attempts.append(att)
            q_pick = self.pick(mid, u, G, w, att)
            if q_pick is None:
                continue
            ok = self.deliver(mid, u, att, v_tr, back=lambda q_=q_pick: self.put_back(q_, u))
        st = sm.rec[mid]
        rec.update(grasp_attempts=attempts, placed_by="panda" if ok else "nobody", state=sm.state[mid], achieved=bool(st["achieved"]),
                   t_start_s=round(t0, 2), t_locked_s=st["t_s"], pos_err_mm=st["pos_err_mm"], ang_err_deg=st["ang_err_deg"],
                   score_after=sm.score(), n_unplanned_contacts=len(self.contact_log) - n_c0,
                   unplanned_contacts=self.contact_summary(self.contact_log[n_c0:])[:20])
        self.graze_mm = 0.5
        return rec

    def pick(self, mid, u, G, w, att):
        """the Panda in use picks the unit where it rests with grasp G (in the unit's frame), lifts it 60 mm; q_pick or
        None (att["result"] says why)"""
        cell, C = self.cell, self.C
        T_present = cell.body_T(u)
        q_pick, q_pg, _mrg = self.pick_checks(u, T_present, G, w_open=w, touch_units=tuple(self.group_of(u)))
        if q_pick is None:
            att["result"] = "no longer feasible where the unit rests (%s)" % _mrg
            return None
        self.work.update(phase="to the pick", held=[])
        C.command_width(min(0.08, w + getattr(self, "last_pick_open", PICK_OPEN_M)))
        if not self.go(q_pg):
            att["result"] = "no path to the pre-grasp (%s)" % self.why["last"]
            return None
        self.work["phase"] = "approach"
        # the open fingers come down around the unit without touching it (nor anything else)
        if not self.move_line(T_present @ G, speed=0.04, check=((), (), None, 3, PICK_LINK_CLEAR_M, PICK_BENCH_FINGER_CLEAR_M, PICK_FINGER_CLEAR_M, (u,))):
            att["result"] = "approach not free (%s)" % self.why["last"]
            self.go_home()
            return None
        self.work["phase"] = "grasp"
        self.work["touch_ok"] = {u}                   # the fingers closing on it
        mass = sum(float(self.m.body_mass[b]) for b in range(self.m.nbody) if cell.unit_of_body.get(b) == u)
        f_finger = grip_force(mass)
        C.command_width(max(0.0, w - 4.0 * f_finger / 2500.0))
        both, w_prev = False, None
        fl, fr = self.cur + "/left_finger", self.cur + "/right_finger"
        for _k in range(20):
            cell.tick(10)
            touching = self.touching(u)
            both = fl in touching and fr in touching
            w_now = C.width()
            if both and w_prev is not None and abs(w_now - w_prev) < 2e-4:
                break
            w_prev = w_now
        att["both_fingers"] = both
        if not both:
            self.retreat(mid)
            att["result"] = "the fingers did not close on the unit"
            self.go_home()
            return None
        group = sorted(self.group_of(u))
        self.work["held"] = group
        G_grip = _inv(cell.body_T(u)) @ self.tcp_T()       # the hand in the unit's frame as it closed
        self.snap("%s_grasped" % mid, u)
        self.work["phase"] = "lift"
        P = self.cmd_T()
        P[2, 3] += 0.06
        hs = self.held_spec(u)
        if not self.move_line(P, speed=0.04, check=(hs, ("world",))):
            self.move_line(P, speed=0.04)          # straight up off the bench (its own check failed on the bench)
        # a sub-assembly with a part on an unpowered joint (the forearm on J4) swings until it hangs still
        cell.tick(150 if len(group) > 1 else 20)
        # slip at the fingers: the hand's pose in the unit's frame (a unit's root can lie far from the grasp: the
        # upper arm's, on the J3 axis, moved 5 mm for a 2 deg turn in the fingers)
        E_s = _inv(G_grip) @ (_inv(cell.body_T(u)) @ self.tcp_T())
        self.last_lift_E = E_s
        att["lift_slip"] = [round(float(np.linalg.norm(E_s[:3, 3])) * 1000, 1), round(math.degrees(rotation_angle(E_s[:3, :3])), 1)]
        self.snap("%s_lifted" % mid, u)
        # which way it turned (in the hand's frame: about the closing axis y?)
        from scipy.spatial.transform import Rotation as _R
        att["lift_turn_axis_hand"] = np.round(_R.from_matrix(E_s[:3, :3]).as_rotvec(), 3).tolist()
        # a long unit lifted by one end turns in the fingers until its far end leaves the bench (the upper arm: 10
        # deg); the insertion is re-checked below with the grasp as it is now: only a unit falling out is refused
        # (the upper arm: 3 mm / 11 deg and held; the end effector held at 14 mm: 5 mm / 16 deg, then it turned out of
        # the fingers when the hand tilted on the way and fell from 0.8 m): beyond 4.5 mm / 13 deg it is put back
        if att["lift_slip"][0] > 4.5 or att["lift_slip"][1] > 13.0:
            att["result"] = "slipped in the lift (%.1f mm / %.1f deg): put back" % tuple(att["lift_slip"])
            self.put_back(q_pick, u)
            self.go_home()
            return None
        # held, or still turning? the hand still for 0.6 s: a unit the fingers do not hold keeps turning (the end
        # effector: 16 deg in the lift, then 120 deg on the way, and fell into the hand)
        T_h = _inv(cell.body_T(u)) @ self.tcp_T()
        cell.tick(max(1, int(round(0.6 / cell.tick_s))))
        E_c = _inv(T_h) @ (_inv(cell.body_T(u)) @ self.tcp_T())
        att["creep_0p6s"] = [round(float(np.linalg.norm(E_c[:3, 3])) * 1000, 2), round(math.degrees(rotation_angle(E_c[:3, :3])), 2)]
        if att["creep_0p6s"][0] > 1.0 or att["creep_0p6s"][1] > 1.0:
            att["result"] = "still turning in the fingers after the lift (%.1f mm / %.1f deg in 0.6 s): put back" % tuple(att["creep_0p6s"])
            self.put_back(q_pick, u)
            self.go_home()
            return None
        return q_pick

    def deliver(self, mid, u, att, v_tr, back):
        """the unit in the hand of the Panda in use: insertion re-checked with the grasp as it is, transport, insertion,
        hold or let go, home; True when the mate fastened. `back()` puts the unit back when it cannot go in."""
        cell, sm, C = self.cell, self.sm, self.C
        group = sorted(self.group_of(u))
        G_act = _inv(cell.body_T(u)) @ self.tcp_T()
        self.last_release_open, self.last_release_dir = RELEASE_OPENS[0], None
        chk = self.place_checks(mid, self.target(mid), G_act, group, width=C.width())
        self.work["release_open"] = self.last_release_open
        self.work["release_dir"] = self.last_release_dir
        att["release_open_mm"] = round(self.last_release_open * 1000)
        if chk[0] is None:
            att["result"] = "with the grasp as it is: %s" % chk[1]
            back()
            self.go_home()
            return False
        # transport to the via pose, then straight to the pre-insertion
        self.work["phase"] = "transport"
        hs = self.held_spec(u)
        tg = self.target(mid)
        T_pre = tg["T"].copy()
        T_pre[:3, 3] -= tg["ins"] * PRE_M
        # the via posture: of the IK solutions the first collision-free one (with the unit in the hand) from which the
        # straight move to the pre-insertion stays within the joint ranges (the first solution alone put link 7 into
        # the gallows' base)
        q_via = None
        self.sync(held=hs)
        vwhy = {}
        # (60 mm out along the insertion first; closer when that is in the way: the right forearm comes in from
        # between the arms, where the gallows' stub is)
        cands_v = []
        for vd in (VIA_M, 0.03, 0.0):
            T_via = T_pre.copy()
            T_via[:3, 3] -= tg["ins"] * vd
            cands_v += [(vd, q_c) for _sc, q_c, _m in self.K.candidates(T_via @ G_act, np.array(C.q_cmd))]
        for vd, q_c in cands_v:
            q2_, ok2_ = self.K.solve_one((T_pre @ G_act)[:3, 3], (T_pre @ G_act)[:3, :3], q_c, q_c, margin=0.0)
            if not ok2_ or float(np.max(np.abs(q2_ - q_c))) > 1.2 or self.K.margin(q2_) < 0.02:
                vwhy["to the pre-insertion"] = vwhy.get("to the pre-insertion", 0) + 1
                continue
            h1 = self.PL.hits(q_c, link_clear=PRE_LINK_CLEAR_M)
            h2 = self.PL.hits(q2_, link_clear=PRE_LINK_CLEAR_M, held_clear=0.0)
            if h1 or h2:
                k_ = "hits " + ",".join(sorted(set("%s>%s" % (h[0].split("/")[-1], h[1]) for h in (h1 or h2)))[:3])
                vwhy[k_] = vwhy.get(k_, 0) + 1
                continue
            q_via = q_c
            break
        if q_via is None:
            att["via_rejected"] = vwhy
            self.log("[panda]     %s: no via posture: %s" % (mid, vwhy))
        if q_via is None or not self.go(q_via, held=hs, speed=v_tr, link_clear=PRE_LINK_CLEAR_M) or not self.move_line(T_pre @ G_act, speed=0.03, check=(hs, (), 0.0, 3, PRE_LINK_CLEAR_M)):
            att["result"] = "no path to the pre-insertion (%s)" % (self.why["last"] if q_via is not None else "IK")
            back()
            self.go_home()
            return False
        G_now = _inv(cell.body_T(u)) @ self.tcp_T()
        E_t = _inv(G_act) @ G_now
        att["grasp_moved_in_transport"] = [round(float(np.linalg.norm(E_t[:3, 3])) * 1000, 1), round(math.degrees(rotation_angle(E_t[:3, :3])), 1)]
        moved_ = att["grasp_moved_in_transport"][0] > 5.0 or att["grasp_moved_in_transport"][1] > 5.0
        if moved_ and att["grasp_moved_in_transport"][0] < 15.0 and att["grasp_moved_in_transport"][1] < 20.0:
            # turned a little in the fingers on the way (the upper arm: 8 deg): the insertion again with the grasp as
            # it is now
            if self.place_checks(mid, self.target(mid), G_now, group, width=C.width())[0] is not None:
                moved_ = False
        if moved_:
            # it slipped in the fingers (or fell): no insertion with it; still held: put back (never let go in the air)
            att["result"] = "the unit moved %.0f mm / %.0f deg in the fingers on the way" % tuple(att["grasp_moved_in_transport"])
            if any(self.cur + "/" in b_ and "finger" in b_ for b_ in self.touching(u)):
                back()
            else:
                self.retreat(mid)
            self.go_home()
            return False
        self.align(mid, PRE_M)
        self.snap("%s_pre" % mid, u)
        # insert
        self.work["phase"] = "insert"
        C.set_stiffness(0.4)
        # the unpowered arm may still swing from the last unit (the left arm turned 1.6 deg/s about J1 with nothing
        # touching it): wait for it to come to rest before going in, else the guard below takes its swing for a jam
        att["arm_settle_s"] = self.wait_arm_rest(mid)
        x_end, moved = self.insert(mid, PRE_M)
        for k_retry in range(2):
            if moved <= ARM_GUARD_DEG or sm.state[mid] in ("ENGAGED", "HOLDING", "LOCKED"):
                break
            att.setdefault("jams", []).append(round(moved, 1))
            T_b = self.cmd_T()
            T_b[:3, 3] -= self.target(mid)["ins"] * 0.012
            self.move_line(T_b, speed=0.01)
            cell.tick(50)
            C.set_stiffness(1.0)
            self.align(mid, max(self.remaining(mid)[0], 0.0), iters=2)
            C.set_stiffness(0.3)
            # (the last try lets the powered joints give 6 deg: they spring back. The right forearm, pushed onto J4,
            # turned J2 3.0-3.1 deg at 5 mm short on every try; in the task image its third try stopped too)
            x_end, moved = self.insert(mid, max(self.remaining(mid)[0], 0.0), speed=0.008,
                                       powered_guard_deg=6.0 if k_retry == 1 else None)
        ax_rem, lat_rem = self.remaining(mid)
        att["after_insert_mm"] = [round(ax_rem * 1000, 2), round(lat_rem * 1000, 2), sm.state[mid]]
        att["bearing_deg"] = {jn: round(math.degrees(float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(jn).id]])), 1)
                              for jn in ("hinge_" + mid[:2] + "J4D",) if jn in [self.m.joint(j_).name for j_ in range(self.m.njnt)]}
        self.snap("%s_inserted" % mid, u)
        if sm.state[mid] not in ("ENGAGED", "HOLDING", "LOCKED"):
            # not on its feature: back out with it and put it back where it was picked
            att["result"] = "not on its feature after the insertion (%.0f mm short, %.0f mm off): backed out" % (ax_rem * 1000, lat_rem * 1000)
            self.snap("%s_jam" % mid, self.mates[mid]["carrier"])
            C.set_stiffness(1.0)
            T_b = self.cmd_T()
            T_b[:3, 3] -= self.target(mid)["ins"] * PRE_M
            if not self.move_line(T_b, speed=0.02, check=(self.held_spec(u), (), None, 3, 0.002, None, 0.001, ())):
                # (the way in reversed: out along the insertion it came in by, checked against penetration only)
                att["back_out"] = self.why["last"]
                self.move_line(T_b, speed=0.02, check=(self.held_spec(u), (), 0.0, 3, 0.0, None, 0.0, ()))
            if back() is False:
                att["result"] += "; could not put it back: " + str(self.why["last"])
                return False
            self.go_home()
            return False
        if self.mates[mid]["kind"] == "seat":
            # on its seat (ENGAGED: the capture holds it): let go, the cell fastens it at rest
            C.set_stiffness(1.0)
            self.retreat(mid)
            for _ in range(HOLD_LOCK_TICKS):
                if sm.rec[mid]["fastened"]:
                    break
                cell.tick()
        else:
            self.work["phase"] = "hold"
            C.set_stiffness(0.6)
            self.hold_until_locked(mid)
            C.set_stiffness(1.0)
            self.retreat(mid)
        self.work["phase"] = "back"
        self.go_home()
        cell.tick(30)
        ok = bool(sm.rec[mid]["fastened"])
        att["result"] = "fastened" if ok else "not fastened (%s)" % sm.state[mid]
        return ok

    # ------------------------------------------------------------------ handover between the two Pandas
    def other_panda(self):
        return [p["name"] for p in self.cell.pandas if p["name"] != self.cur][0]

    def handover_plan(self, mid, picker, n1=12, n2=16):
        """a handover of the mate's unit: the picker takes it from the bench with grasp G2, holds it at T_h (the unit in
        its insertion orientation, its centre of mass at one of a few points in front of the arms), the Panda in use
        takes it there with G1 (a grasp that works at the insertion), the picker lets go. Both Pandas collision-free
        at T_h, each against the other and the unit; the taker's open fingers come in along its approach clear of the
        picker. Returns {G1, w1, q1, G2, w2, q2, T_h} or None."""
        P1 = self.cur
        tg = self.target(mid)
        u = tg["unit"]
        group = sorted(self.group_of(u))
        T_u = self.cell.body_T(u)
        rel = {x: _inv(T_u) @ self.cell.body_T(x) for x in group}
        com = self.com_local(u)
        R_t = tg["T"][:3, :3]
        cands = self.grasp_candidates(u, T_u)
        G1s = []
        for G, w, d in cands:
            if self.place_checks(mid, tg, G, group, width=w, fast=True)[0] is not None:
                G1s.append((G, w, d))
            if len(G1s) >= n1:
                break
        self.use(picker)
        G2s = []
        # the picker may hold the unit by the parts fastened to it as well (the right forearm by its HA)
        cands2 = self.grasp_candidates(u, T_u, with_units=tuple(x for x in group if x != u)) if len(group) > 1 else cands
        for G, w, d in cands2:
            if self.pick_checks(u, T_u, G, w_open=w, fast=True)[0] is not None:
                G2s.append((G, w, d))
            if len(G2s) >= n2:
                break
        self.use(P1)
        why = {"G1": len(G1s), "G2": len(G2s)}
        if not G1s or not G2s:
            return None, why
        cnt = why.setdefault("fail", {})

        def bump(k):
            cnt[k] = cnt.get(k, 0) + 1
        # meeting points in front of the arms, between the two Pandas; the unit in its insertion orientation turned about
        # the vertical (the taker turns it on the way to the arm)
        Hs = [(x, y, z) for z in (0.40, 0.32, 0.48) for x in (-0.45, -0.38, -0.52) for y in (0.0, -0.08, 0.08)]
        for H in Hs:
          for yaw in (0.0, math.pi / 2, -math.pi / 2, math.pi):
            T_h = np.eye(4)
            T_h[:3, :3] = rotz(yaw) @ R_t
            T_h[:3, 3] = np.asarray(H) - T_h[:3, :3] @ com
            poses = {x: T_h @ rel[x] for x in group}
            for G1, w1, d1 in G1s:
                self.use(P1)
                q1, _m1 = self.fast_ik(T_h @ G1)
                if q1 is None:
                    bump("taker IK")
                    continue
                if self.collides(q1, allow=tuple(group), poses=poses, width=w1):
                    bump("taker hits")
                    continue
                P_pg = T_h @ G1
                P_pg[:3, 3] -= P_pg[:3, 2] * 0.08
                q1pg, ok1 = self.K.solve_one(P_pg[:3, 3], P_pg[:3, :3], q1, q1, margin=0.0)
                if not ok1:
                    bump("taker pre-grasp IK")
                    continue
                for G2, w2, d2 in G2s:
                    if float(np.linalg.norm(G1[:3, 3] - G2[:3, 3])) < 0.008:
                        bump("same spot")
                        continue                                   # the two hands on the same spot
                    self.use(picker)
                    q2, _m2 = self.fast_ik(T_h @ G2)
                    if q2 is None:
                        bump("picker IK")
                        continue
                    self.sync(allow=tuple(group), poses=poses, width=w2, q_other={P1: q1}, touch=())
                    if self.PL.hits(q2, finger_clear=0.002):
                        bump("picker hits (the taker / the scene)")
                        continue
                    # the taker coming in, open, past the picker holding the unit
                    self.use(P1)
                    clear = True
                    for k in range(6):
                        Pk = (1 - k / 5) * P_pg[:3, 3] + k / 5 * (T_h @ G1)[:3, 3]
                        qk, okk = self.K.solve_one(Pk, P_pg[:3, :3], q1pg, q1pg, margin=0.0)
                        self.sync(allow=(), poses=poses, width=min(0.08, w1 + PICK_OPEN_M), q_other={picker: q2}, touch=tuple(group))
                        if not okk or self.PL.hits(qk, finger_clear=0.0, link_clear=PICK_LINK_CLEAR_M):
                            clear = False
                            bump("taker's approach")
                            break
                    if clear:
                        self.use(P1)
                        return {"G1": G1, "w1": w1, "d1": d1, "q1": q1, "q1pg": q1pg, "G2": G2, "w2": w2, "d2": d2, "q2": q2,
                                "T_h": T_h, "H": H, "yaw_deg": round(math.degrees(yaw))}, why
        self.use(P1)
        return None, why

    def place_handover(self, mid, pinned=None):
        """the mate's unit picked from the bench by the other Panda and handed over to the Panda in use in the air,
        which then inserts it (`deliver`); falls back to `place` when no handover is found"""
        cell, sm = self.cell, self.sm
        P1 = self.cur
        P2 = self.other_panda()
        tg = self.target(mid)
        u = tg["unit"]
        rec = {"unit": u, "panda": P1, "picker": P2, "placed_by": None}
        t0 = float(self.d.time)
        n_c0 = len(self.contact_log)
        t_plan = time.time()
        hp, why = self.handover_plan(mid, P2)
        rec["handover_plan_wall_s"] = round(time.time() - t_plan, 1)
        rec["handover_candidates"] = why
        if hp is None:
            self.log("[panda] %s: no handover (%s): placed directly" % (mid, why))
            r_ = self.place(mid, pinned)
            r_.update({k: v for k, v in rec.items() if k.startswith("handover")})
            return r_
        att = {"grasp": hp["d1"], "picker_grasp": hp["d2"], "handover_at": list(hp["H"]), "width_mm": round(hp["w1"] * 1000, 1)}
        ok = False
        # 1) the picker takes it from the bench and holds it at the handover pose
        self.use(P2)
        self.work["phase"] = "handover: pick"
        q_pick = self.pick(mid, u, hp["G2"], hp["w2"], att)
        if q_pick is not None:
            hs = self.held_spec(u)
            self.work["phase"] = "handover: to the meeting point"
            if not self.go(hp["q2"], held=hs, speed=0.3):
                att["result"] = "no path to the handover pose (%s)" % self.why["last"]
                self.put_back(q_pick, u)
                self.go_home()
            else:
                cell.tick(30)
                # 2) the taker comes in open, closes on it
                self.use(P1)
                self.work["phase"] = "handover: take"
                C1 = self.C
                C1.command_width(min(0.08, hp["w1"] + PICK_OPEN_M))
                T_now = cell.body_T(u)
                P_t = T_now @ hp["G1"]
                P_pg = P_t.copy()
                P_pg[:3, 3] -= P_t[:3, 2] * 0.08
                q_pg, okp = self.K.solve_one(P_pg[:3, 3], P_pg[:3, :3], hp["q1pg"], hp["q1pg"], margin=0.0)
                if not okp or not self.go(q_pg) or not self.move_line(P_t, speed=0.03, check=((), (), None, 3, PICK_LINK_CLEAR_M, None, PICK_FINGER_CLEAR_M, tuple(self.group_of(u)))):
                    att["result"] = "the taker could not come in (%s)" % self.why["last"]
                    self.go_home()
                    self.use(P2)
                    self.put_back(q_pick, u)
                    self.go_home()
                else:
                    mass = sum(float(self.m.body_mass[b]) for b in range(self.m.nbody) if cell.unit_of_body.get(b) in self.group_of(u))
                    f_finger = grip_force(mass)
                    self.work["touch_ok"] = set(self.group_of(u))
                    C1.command_width(max(0.0, hp["w1"] - 4.0 * f_finger / 2500.0))
                    cell.tick(80)
                    touching = self.touching(u)
                    both = (P1 + "/left_finger") in touching and (P1 + "/right_finger") in touching
                    att["taker_both_fingers"] = both
                    # 3) the picker lets go and backs away
                    self.use(P2)
                    self.work["phase"] = "handover: let go"
                    self.C.command_width(0.08)
                    cell.tick(40)
                    T2 = self.cmd_T()
                    for v in (-T2[:3, 2], np.array([0.0, 0.0, 1.0])):
                        T_b = T2.copy()
                        T_b[:3, 3] += v * 0.08
                        if self.move_line(T_b, speed=0.04, check=((), (), None, 3, 0.002, None, 0.002, tuple(self.group_of(u)))):
                            break
                    self.go_home()
                    self.use(P1)
                    self.work["held"] = sorted(self.group_of(u))
                    if not both:
                        att["result"] = "the taker's fingers did not close on it"
                    else:
                        # 4) the taker inserts it
                        def back():
                            # put it down on the bench where it was picked, if this Panda reaches there
                            T_d = np.asarray(cell.initial[u], float) if u in cell.initial else None
                            G_a = _inv(cell.body_T(u)) @ self.tcp_T()
                            q_d = self.fast_ik(T_d @ G_a)[0] if T_d is not None else None
                            if q_d is None or not self.go(q_d, held=self.held_spec(u), allow=("world",)):
                                self.log("[panda]     %s: cannot put it back, let go over the bench" % mid)
                            self.C.command_width(0.08)
                            cell.tick(40)
                            self.work["held"] = []
                        ok = self.deliver(mid, u, att, 0.3, back)
        st = sm.rec[mid]
        rec.update(grasp_attempts=[att], placed_by="panda" if ok else "nobody", state=sm.state[mid], achieved=bool(st["achieved"]),
                   t_start_s=round(t0, 2), t_locked_s=st["t_s"], score_after=sm.score(), n_unplanned_contacts=len(self.contact_log) - n_c0,
                   unplanned_contacts=self.contact_summary(self.contact_log[n_c0:])[:20])
        return rec

    def place_air(self, mid_p, mid_c, pinned_p=None, pinned_c=None):
        """a part put onto its carrier in the air (the left HB onto J4_D of the upper arm: no rest of the upper arm on the
        bench both takes HB and leaves a grasp that mounts it, and no Panda reaches HB's place on the built arm): the
        carrier's Panda picks the carrier with a grasp that also mounts it with the part on and holds it where the other
        Panda reaches the part's seat; the other Panda places the part (`place`: its target live on the held carrier);
        then the carrier's Panda mounts carrier and part (`deliver`). Returns (the part's record, the carrier's)."""
        from scipy.spatial.transform import Rotation
        cell, sm = self.cell, self.sm
        P2 = self.cur
        u_c, u_p = self.mates[mid_c]["unit"], self.mates[mid_p]["unit"]
        P1 = [p_["name"] for p_ in cell.pandas if p_.get("arm") == ("left" if mid_c.startswith("l_") else "right")][0]
        rec_c = {"unit": u_c, "panda": P1, "placed_by": "nobody", "with_in_air": mid_p}
        rec_p = {"unit": u_p, "panda": P2, "placed_by": "nobody", "in_air_on": u_c}
        t0, n_c0 = float(self.d.time), len(self.contact_log)
        from .cell import design_world
        D = design_world(self.kit, cell.layout["fixture"])
        rel_p = _inv(D[u_c]) @ D[u_p]

        def finish():
            for r_, m_ in ((rec_p, mid_p), (rec_c, mid_c)):
                st = sm.rec[m_]
                r_.update(state=sm.state[m_], achieved=bool(st["achieved"]), t_start_s=round(t0, 2), t_locked_s=st["t_s"],
                          score_after=sm.score())
            rec_c.update(n_unplanned_contacts=len(self.contact_log) - n_c0, unplanned_contacts=self.contact_summary(self.contact_log[n_c0:])[:20])
            self.work["q_other"] = None
            self.work["held_other"] = []
            return rec_p, rec_c
        # 1) the carrier's grasps that work at its pick and at its mount with the part on
        self.use(P1)
        tg_c = self.target(mid_c)
        feas, ncand, _rej = self.plan_grasps(mid_c, tg_c, pinned_c)
        G1s = [f_ for f_ in feas if self.place_checks(mid_c, tg_c, f_[1], [u_c, u_p], width=f_[2], fast=True,
                                                     rel={u_c: np.eye(4), u_p: rel_p})[0] is not None][:4]
        rec_c.update(grasp_candidates=ncand, grasps_feasible=len(feas), grasps_with_part=len(G1s))
        if not G1s:
            self.log("[panda] %s: no grasp of %s that picks it and mounts it with %s on" % (mid_c, u_c, u_p))
            return finish()
        # 2) the other Panda's grasps of the part where it rests
        self.use(P2)
        T_pb = cell.body_T(u_p)
        mm_p = self.mates[mid_p]
        ins_u = (_inv(D[u_p]) @ D[mm_p["carrier"]])[:3, :3] @ np.asarray(mm_p["insertion_in_carrier"], float)
        ins_u /= np.linalg.norm(ins_u)
        # (from behind the part along its insertion: one from its side runs the fingers into the held carrier)
        G2s = [(G, w, d) for G, w, d in self.grasp_candidates(u_p, T_pb) if float(G[:3, 2] @ ins_u) > 0.7
               and self.pick_checks(u_p, T_pb, G, w_open=w, fast=True)[0] is not None][:12]
        rec_p["grasps_pickable"] = len(G2s)
        # 3) the meeting pose: the carrier in front of the arms, the part's insertion down or 45 deg from down
        mm_p = self.mates[mid_p]
        T_cb_rel = _inv(cell.body_T(u_c)) @ cell.body_T(mm_p["carrier"])
        ins_l = T_cb_rel[:3, :3] @ np.asarray(mm_p["insertion_in_carrier"], float)
        ins_l /= np.linalg.norm(ins_l)
        com = self.com_local(u_c)
        found, n_try = None, {"P1 IK": 0, "P1 hits": 0, "P2": 0}
        for H in ((-0.44, 0.0, 0.40), (-0.40, -0.06, 0.40), (-0.48, 0.06, 0.40), (-0.44, 0.0, 0.48)):
            for tilt in (0.0, math.radians(45.0)):
                for spin in range(8):
                    d_ = rotz(spin * math.pi / 4)[:3, :3] @ np.array([math.sin(tilt), 0.0, -math.cos(tilt)])
                    R0 = Rotation.align_vectors([d_], [ins_l])[0].as_matrix()
                    for roll in range(4):
                        R = Rotation.from_rotvec(d_ * roll * math.pi / 2).as_matrix() @ R0
                        T_air = np.eye(4)
                        T_air[:3, :3] = R
                        T_air[:3, 3] = np.asarray(H) - R @ com
                        for _s, G1, w1, d1, q_pick1, _qpg in G1s:
                            self.use(P1)
                            self.work["q_other"] = None
                            q1, _m = self.fast_ik(T_air @ G1)
                            if q1 is None:
                                n_try["P1 IK"] += 1
                                continue
                            if self.collides(q1, allow=(u_c,), poses={u_c: T_air}, width=w1):
                                n_try["P1 hits"] += 1
                                continue
                            T_cb = T_air @ T_cb_rel
                            ins_w = T_cb[:3, :3] @ np.asarray(mm_p["insertion_in_carrier"], float)
                            tg_p = {"T": T_cb @ np.array(mm_p["T_part_in_carrier"], float), "ins": ins_w / np.linalg.norm(ins_w),
                                    "unit": u_p, "carrier": mm_p["carrier"]}
                            self.use(P2)
                            self.work["q_other"] = {P1: q1}
                            for G2, w2, d2 in G2s:
                                n_try["P2"] += 1
                                q2, _w2 = self.place_checks(mid_p, tg_p, G2, [u_p], poses={u_c: T_air}, width=w2, fast=True, rel={u_p: np.eye(4)})
                                if q2 is None:
                                    k_ = "P2: " + str(_w2).split(" at ")[0].split(" going")[0][:60]
                                    n_try[k_] = n_try.get(k_, 0) + 1
                                else:
                                    found = (T_air, G1, w1, d1, q1, H, tilt, spin, roll, d2)
                                    break
                            self.work["q_other"] = None
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        rec_c["meeting_search"] = n_try
        if found is None:
            self.use(P1)
            self.log("[panda] %s: no meeting pose for %s in the air (%s)" % (mid_p, u_p, n_try))
            return finish()
        T_air, G1, w1, d1, q1, H, tilt, spin, roll, d2 = found
        rec_c["meeting"] = {"at": list(H), "tilt_deg": round(math.degrees(tilt)), "spin": spin, "roll": roll, "grasp": d1, "part_grasp": d2}
        self.log("[panda]     %s: meeting pose %s" % (mid_p, rec_c["meeting"]))
        # 4) the carrier's Panda picks it and holds it there
        self.use(P1)
        att_c = {"grasp": d1, "width_mm": round(w1 * 1000, 1)}
        rec_c["grasp_attempts"] = [att_c]
        q_pick = self.pick(mid_c, u_c, G1, w1, att_c)
        if q_pick is None:
            return finish()
        E_l = self.last_lift_E
        if math.degrees(rotation_angle(E_l[:3, :3])) > 3.0:
            # it turned in the fingers as its far end left the bench (the left upper arm, held 80 mm from its centre of
            # mass: 9.5 deg, every time; its fingers then met L2 going in): put it back and take it again turned the
            # other way by as much, so that the lift turns it into the grasp it is mounted with
            self.put_back(q_pick, u_c)
            self.go_home()
            G1c = G1 @ _inv(E_l)
            att_c2 = {"grasp": d1 + " (pre-turned %.1f deg)" % math.degrees(rotation_angle(E_l[:3, :3])), "width_mm": round(w1 * 1000, 1)}
            rec_c["grasp_attempts"].append(att_c2)
            q_pick2 = self.pick(mid_c, u_c, G1c, w1, att_c2)
            if q_pick2 is None:
                att_c2["result"] = att_c2.get("result", "the pre-turned grasp failed")
                q_pick = self.pick(mid_c, u_c, G1, w1, att_c)
                if q_pick is None:
                    return finish()
            else:
                q_pick, att_c = q_pick2, att_c2
        self.work["phase"] = "in the air: to the meeting pose"
        if not self.go(q1, held=self.held_spec(u_c), speed=0.3):
            att_c["result"] = "no path to the meeting pose (%s)" % self.why["last"]
            self.put_back(q_pick, u_c)
            self.go_home()
            return finish()
        cell.tick(50)
        # 5) the other Panda puts the part on (its target live on the held carrier)
        self.work["held_other"] = sorted(self.group_of(u_c))
        held_c = list(self.work["held"])
        self.use(P2)
        r_ = self.place(mid_p, pinned_p)
        rec_p.update({k: v for k, v in r_.items() if k not in ("unit", "panda")})
        self.use(P1)
        self.work["held"] = held_c
        self.work["held_other"] = []
        # 6) the carrier's Panda mounts carrier (and part)
        ok = self.deliver(mid_c, u_c, att_c, 0.3, back=lambda q_=q_pick: self.put_back(q_, u_c))
        if not ok and sm.rec[mid_p]["fastened"] and not self.work["held"]:
            # (put back on the bench with the part on: the upper arm turned 10 deg in the fingers as it was lifted and
            # its fingers then met L2 going in) its own placement from where it lies now, the part fastened to it
            self.go_home()
            r_ = self.place(mid_c, None)
            rec_c["after_put_back"] = {k: v for k, v in r_.items() if k not in ("unit", "panda")}
            ok = r_.get("placed_by") == "panda"
        rec_c["placed_by"] = "panda" if ok else "nobody"
        return finish()

    def touching(self, u):
        m, d = self.m, self.d
        ub = set(b for b in range(m.nbody) if self.cell.unit_of_body.get(b) == u)
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in ub and b2 not in ub:
                out.add(m.body(b2).name)
            elif b2 in ub and b1 not in ub:
                out.add(m.body(b1).name)
        return out

    def bearing_error(self, mid):
        """[(joint id, body id, angle rad)]: the mate's co-welded parts on a hinge (J4_D on its bearing) and the turn
        each needs about its axis to sit where the unit's target puts it"""
        from scipy.spatial.transform import Rotation
        m, d = self.m, self.d
        tg = self.target(mid)
        T_u = self.cell.body_T(tg["unit"])
        out = []
        for c in self.mates[mid].get("co_welds", []):
            b = m.body(c["part"]).id
            if m.body_jntnum[b] != 1 or m.jnt_type[m.body_jntadr[b]] != self.mj.mjtJoint.mjJNT_HINGE:
                continue
            j = int(m.body_jntadr[b])
            T_des = tg["T"] @ _inv(T_u) @ self.cell.body_T(c["carrier"]) @ np.asarray(c["T_part_in_carrier"], float)
            rv = Rotation.from_matrix(T_des[:3, :3] @ d.xmat[b].reshape(3, 3).T).as_rotvec()
            out.append((j, b, float(rv @ d.xaxis[j])))
        return out

    def align_bearing(self, mid, tol_deg=BEARING_TOL_DEG):
        """turn each co-welded bearing part within tol_deg of its target angle: the closed fingertips come down beside
        the part's far end and push it round (a straight 1 mm-step push, the angle watched). [record per part]"""
        out = []
        for j, b, e in self.bearing_error(mid):
            r_ = {"part": self.m.body(b).name, "before_deg": round(math.degrees(e), 1)}
            if abs(math.degrees(e)) > 30.0:
                r_["result"] = "not pushed: %.0f deg off (its carrier not on the arm?)" % math.degrees(e)
            elif abs(math.degrees(e)) > tol_deg:
                r_["result"] = self._push_bearing(mid, j, b, tol_deg)
                r_["after_deg"] = round(math.degrees([e_ for j_, b_, e_ in self.bearing_error(mid) if b_ == b][0]), 1)
            self.log("[panda]     %s: bearing %s" % (mid, r_))
            out.append(r_)
        return out

    def _push_bearing(self, mid, j, b, tol_deg):
        m, d = self.m, self.d
        name = m.body(b).name
        e = [e_ for j_, b_, e_ in self.bearing_error(mid) if b_ == b][0]
        a, o = d.xaxis[j].copy(), d.xanchor[j].copy()
        V = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != self.mj.mjtGeom.mjGEOM_MESH or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
                continue
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            V.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        V = np.vstack(V) - o
        ax = V @ a
        rad = V - np.outer(ax, a)
        r = np.linalg.norm(rad, axis=1)
        band = r >= 0.7 * r.max()
        cen = rad[band].mean(axis=0)
        push = math.copysign(1.0, e) * np.cross(a, cen / np.linalg.norm(cen))
        s_ = V[band] @ push
        edge = V[band][s_ <= s_.min() + 0.004]
        P = o + edge.mean(axis=0)
        r_P = float(np.linalg.norm(edge.mean(axis=0) - (edge.mean(axis=0) @ a) * a))
        self.work["phase"] = "align bearing"
        self.go_home()
        self.C.command_width(0.0)
        for _ in range(200):
            self.cell.tick()
            if self.C.width() < 0.002:
                break
        why = "no posture"
        for q_a, T_s, back, z in self._push_starts(name, P, push, a):
            if not self.go(q_a):
                why = "no path to the push"
                continue
            if not self.move_line(T_s, speed=0.02, check=((), (), None, 3, PRE_LINK_CLEAR_M, None, 0.002, (name,))):
                why = "the way down to the push"
                self.go_home()
                continue
            # push along `push` until the angle is within BEARING_AIM_DEG (or past): 1 mm steps, finer as it comes in
            # (the forearm's fingers pass J4_D's plate 100 mm from the axis: 0.5 deg left there is 0.9 mm)
            self.work["touch_ok"] = {self.cell.unit_of_body.get(b)}
            T_k = T_s.copy()
            for _k in range(int((back + abs(e) * r_P) / 0.0002) + 20):
                e_k = [e_ for j_, b_, e_ in self.bearing_error(mid) if b_ == b][0]
                if abs(math.degrees(e_k)) <= BEARING_AIM_DEG or e_k * e < 0:
                    break
                if float((T_k[:3, 3] - T_s[:3, 3]) @ push) > back + abs(e) * r_P + 0.008:
                    break                                   # (pushed as far as the turn needs, and 8 mm more)
                T_k[:3, 3] += push * min(0.001, max(0.0002, 0.5 * abs(e_k) * r_P))
                if not self.move_line(T_k, speed=0.01, check=((), (name,), None, 3, PRE_LINK_CLEAR_M, None, 0.002, ())):
                    break
                self.cell.tick(3)
            self.cell.tick(25)
            T_b = self.cmd_T()
            T_b[:3, 3] -= push * 0.006
            self.move_line(T_b, speed=0.01)
            T_b[:3, 3] -= z * 0.06
            self.move_line(T_b, speed=0.03)
            self.work["touch_ok"] = set()
            self.go_home()
            e_f = [e_ for j_, b_, e_ in self.bearing_error(mid) if b_ == b][0]
            return "pushed %.0f mm at r %.0f mm: %.1f deg" % ((T_k[:3, 3] - T_s[:3, 3]) @ push * 1000, r_P * 1000, math.degrees(e_f))
        self.go_home()
        return self.why.get("push", why) if why == "no posture" else why

    def _push_starts(self, name, P, push, a):
        """the push's start postures, best first: yields (q_a above, T_s the start, back m, z the fingers' direction);
        on exhaustion self.why["push"] says why none"""
        zs = [-a, a] + ([np.array([0.0, 0.0, -1.0])] if abs(a[2]) < 0.6 else [])
        whys = []
        for z in zs:
            for ysg in (1.0, -1.0):
                y = ysg * (push - (push @ z) * z)
                if np.linalg.norm(y) < 0.5:
                    continue
                y /= np.linalg.norm(y)
                T = np.eye(4)
                T[:3, :3] = np.column_stack([np.cross(y, z), y, z])
                # the start: the closed fingertips 8-26 mm behind the edge, shifted along the fingers (-30..+30 mm) and
                # across them (-20..+20 mm) until they keep 0.5 mm off the part (2 mm off the rest) and a push of the
                # distance + 4 mm would meet it
                q_s = None
                x_ = T[:3, 0]
                grid = sorted(((dz, dx) for dz in np.arange(-0.03, 0.0301, 0.005) for dx in (-0.02, -0.01, 0.0, 0.01, 0.02)),
                              key=lambda o: abs(o[0]) + abs(o[1]))
                T_n = T.copy()
                T_n[:3, 3] = P - push * 0.018
                seeds = [c_[1] for c_ in self.K.candidates(T_n, np.array(self.C.q_cmd, float), extra_seeds=(READY, HOME))[:4]]
                n_eval = 0
                for dz, dx in grid:
                    for back in (0.008, 0.012, 0.018, 0.026):
                        T_s = T.copy()
                        T_s[:3, 3] = P - push * back - z * dz + x_ * dx
                        T_e = T_s.copy()
                        T_e[:3, 3] += push * (back + 0.004)
                        T_a = T_s.copy()
                        T_a[:3, 3] -= z * 0.06
                        for q0 in seeds:
                            q_s_, ok = self.K.solve_one(T_s[:3, 3], T_s[:3, :3], q0, q0, margin=0.0)
                            if not ok:
                                continue
                            n_eval += 1
                            h_s = [h for h in self.collides(q_s_, width=0.0, finger_clear=0.002, link_clear=PRE_LINK_CLEAR_M) if h[0] != name or h[1] < 0.5]
                            if h_s:
                                k_ = "z %s y %+.0f" % (np.round(z, 1), ysg)
                                if not whys or whys[-1][0] != k_:
                                    whys.append((k_, "start %s" % h_s[:2]))
                                break
                            q_e, ok_e = self.K.solve_one(T_e[:3, 3], T_e[:3, :3], q_s_, q_s_, margin=0.0)
                            if not ok_e or not [h for h in self.collides(q_e, width=0.0, finger_clear=0.0, link_clear=0.0) if h[0] == name]:
                                break                                      # the push would miss the part
                            q_a, ok_a = self.K.solve_one(T_a[:3, 3], T_a[:3, :3], q_s_, q_s_, margin=0.0)
                            if not ok_a or self.collides(q_a, width=0.0, link_clear=PRE_LINK_CLEAR_M):
                                break
                            q_s = (q_a, T_s)
                            break
                        if q_s is not None:
                            break
                    if q_s is not None:
                        break
                self.why["push"] = "no posture (%d starts): %s" % (n_eval, "; ".join("%s: %s" % w_ for w_ in whys))
                if q_s is not None:
                    yield q_s[0], q_s[1], back, z

    def align(self, mid, back, iters=3, speed=0.01, tol_mm=0.3, tol_deg=0.3):
        """move the hand until the held unit is `back` before home along the insertion (re-computed from the carrier)"""
        u = self.mates[mid]["unit"]
        for _ in range(iters):
            tg = self.target(mid)
            T_goal = tg["T"].copy()
            T_goal[:3, 3] -= tg["ins"] * back
            T_part, T_act, T_cmd = self.cell.body_T(u), self.tcp_T(), self.cmd_T()
            E = _inv(T_goal) @ T_part
            if np.linalg.norm(E[:3, 3]) * 1000 < tol_mm and math.degrees(rotation_angle(E[:3, :3])) < tol_deg:
                break
            G_act = _inv(T_part) @ T_act
            self.move_line(T_goal @ G_act @ (_inv(T_act) @ T_cmd), speed=speed, ang_speed=math.radians(5.0))
            self.cell.tick(10)

    # ------------------------------------------------------------------ the whole program
    def run(self, seq, plan=None, only="", stop_after=""):
        report = {"mates": {}, "sequence": [list(s) for s in seq]}
        cur_panda = None
        for mid, pn, where in seq:
            if only and mid not in only.split(","):
                continue
            if pn != cur_panda:
                if cur_panda is not None:
                    self.work["phase"] = "park"
                    self.go_home()
                self.use(pn)
                cur_panda = pn
                self.work["phase"] = "unpark"
                self.go_home()
            self.cell.label.update(text="%s by %s (%s)" % (mid, pn, where), part=self.mates[mid]["unit"], mate=mid)
            if mid in report["mates"]:
                continue                                   # (done with the part put on it in the air)
            pinned = (plan or {}).get("grasps", {}).get(mid)
            t_w = time.time()
            if where == "air":
                c_mid = [m_ for m_, _p, _w in seq if self.mates[m_]["unit"] == self.unit_of[self.mates[mid]["carrier"]]][0]
                rec, rec_c = self.place_air(mid, c_mid, pinned, (plan or {}).get("grasps", {}).get(c_mid))
                cur_panda = self.cur
                report["mates"][c_mid] = rec_c
                self.log("[panda] %-12s %-10s by %-6s t=%6.1f s score %.3f  (with %s in the air)" % (
                    c_mid, self.sm.state[c_mid], rec_c["placed_by"], self.d.time, self.sm.score(), mid))
            else:
                rec = self.place_handover(mid, pinned) if where == "handover" else self.place(mid, pinned)
            rec["wall_s"] = round(time.time() - t_w, 1)
            report["mates"][mid] = rec
            self.log("[panda] %-12s %-10s by %-6s t=%6.1f s score %.3f  attempts %s  contacts %d  (%.0f s wall)" % (
                mid, self.sm.state[mid], rec["placed_by"], self.d.time, self.sm.score(),
                [x_.get("result") for x_ in rec.get("grasp_attempts", [])], rec.get("n_unplanned_contacts", 0), rec["wall_s"]))
            if stop_after and mid == stop_after:
                break
        if cur_panda is not None:
            self.work["phase"] = "park"
            self.go_home()
        # assembled: the motors off (the joints free against their friction from here); the arms left 3 s to stand
        self.powered = False
        qa = {jn: self.m.jnt_qposadr[self.sm.joint_info[jn]["hinge"]] for jn in self.sm.limits_set}
        q_on = {jn: float(self.d.qpos[a_]) for jn, a_ in qa.items()}
        self.cell.tick(int(round(3.0 / self.cell.tick_s)))
        report["powered_during_assembly"] = sorted(self.hold_q)
        report["after_power_off_3s_deg"] = {jn: round(math.degrees(float(self.d.qpos[qa[jn]]) - q0), 2) for jn, q0 in q_on.items()}
        by = [r["placed_by"] for r in report["mates"].values()]
        report.update(score=self.sm.score(), states=dict(self.sm.state), placed_by_panda=by.count("panda"), not_placed=by.count("nobody"),
                      unplanned_contacts_total=len(self.contact_log), events=[list(e) for e in self.sm.events], sim_s=float(self.d.time))
        return report


# ---------------------------------------------------------------------- the bench plan
def bench_plan(cell, seq, log=print, region=None, gap=0.02, tries_per_rest=10):
    """the bench layout: every unit's rest {unit: [pos, quat]} and the grasp it is picked with {mate: {G, w, desc}}.
    `cell` a probe cell (every unit parked off the bench). Units are placed in the REVERSE order of `seq`: when a unit's
    place is chosen, every unit still on the bench at its pick is already placed and every earlier one is on the arm
    (its design pose). Per unit: its stable rests x 8 yaws x bench cells in its Panda's reach annulus (0.30-0.62 m from
    the Panda's axis), clear of the keep-outs and 20 mm from every other unit; the first cell where a grasp that works
    at the insertion is feasible at the pick. A unit that takes a part on the bench (the forearm: HA) rests so that
    the part's insertion points down."""
    from .coupling import quat_wxyz
    A = Assembler(cell, log=log)
    A.graze_mm = 0.5                   # (the run grazes 0.8 mm at most: a margin for the arm as it is built; 0.35 left
                                       # no bench grasp of the left forearm; planned at 0.5, run at 0.5, the planned grasp
                                       # went 0.5+ mm into J4_D on the built arm)
    kit = cell.kit
    thin = set(u for u in cell.loose if u in RISER_UNITS)

    def riser_of(u):
        return ("riser_" + u) if u in thin else None
    mates = cell.mates
    D = {}
    from .cell import design_world
    D = design_world(kit, cell.layout["fixture"])
    bx, by = cell.cfg["bench"]["size_xy"][0] / 2 - 0.03, cell.cfg["bench"]["size_xy"][1] / 2 - 0.03
    # keep-outs (world xy boxes): the fixture's plate, the arms' footprint and the Pandas' bases
    keep = []
    keep.append((-0.25, 0.32, -0.30, 0.30))           # the arms hanging over the bench and the gallows' plate
    for pc in cell.pandas:
        x, y = pc["pos"][0], pc["pos"][1]
        keep.append((x - 0.15, x + 0.15, y - 0.15, y + 0.15))
    order = list(seq)
    rest, grasps, boxes, info = {}, {}, [], {}
    bench_T, riser_T = {}, {}
    rh = cell.cfg["riser"]["block_m"][2]

    step_of = {mates[mid_j]["unit"]: (j, where_j, A.unit_of[mates[mid_j]["carrier"]]) for j, (mid_j, _p, where_j) in enumerate(order)}

    def pose_at(u, k):
        """where unit u is while step k is done: on the arm (its design) or on its bench carrier (a sub-assembly) when it
        was placed before, else where it rests (None: not placed yet)"""
        j, where_j, c_u = step_of[u]
        if where_j == "air":
            # (a part put on in the air: on the bench until its carrier is mounted, then on the arm with it)
            return D[u] if step_of[c_u][0] < k else bench_T.get(u)
        if j < k:
            if where_j in ("arm", "handover"):
                return D[u]
            Tc = pose_at(c_u, k) if c_u != "base" else D["base"]
            return None if Tc is None else Tc @ _inv(D[c_u]) @ D[u]
        return bench_T.get(u)

    def scene_at(k):
        """the scene when step k's unit is picked: the units on the arm at their design, a bench sub-assembly's parts on
        their carrier, the units still to come where they rest (the step's own unit left out)"""
        sc = {}
        u_k = mates[order[k][0]]["unit"]
        for u_j in step_of:
            if u_j == u_k:
                continue
            T_ = pose_at(u_j, k)
            if T_ is not None:
                sc[u_j] = T_
        sc.update(riser_T)
        return sc

    def bench_group(u, k):
        """u and the units fastened onto it on the bench before step k (recursively): they move with it"""
        out = [u]
        for j, (mid_j, _p, where_j) in enumerate(order[:k]):
            if where_j in ("bench", "air") and A.unit_of[mates[mid_j]["carrier"]] in out:
                out.append(mates[mid_j]["unit"])
        return out

    part_grasps = {}
    air_units = set(mates[mid_j]["unit"] for mid_j, _p, where_j in order if where_j == "air")
    base_plan = json.load(open(os.environ["PANDA_PLAN_BASE"])) if os.environ.get("PANDA_PLAN_BASE") else None
    plan_only = set(os.environ.get("PANDA_PLAN_ONLY", "").split(","))

    def part_fits(mj, T_carrier_unit, poses):
        """a grasp of the bench sub-assembly's part works at its insertion onto the carrier unit resting at T_carrier_unit"""
        mmj = mates[mj]
        uj, cu = mmj["unit"], A.unit_of[mmj["carrier"]]
        T_c = T_carrier_unit @ _inv(D[cu]) @ D[mmj["carrier"]]
        T_h = T_c @ np.array(mmj["T_part_in_carrier"], float)
        ins_ = T_c[:3, :3] @ np.asarray(mmj["insertion_in_carrier"], float)
        tgj = {"T": T_h, "ins": ins_ / np.linalg.norm(ins_), "unit": uj, "carrier": mmj["carrier"]}
        if mj not in part_grasps:
            part_grasps[mj] = A.grasp_candidates(uj, D[uj], cap=80)
        ps = dict(poses)
        ps[cu] = T_carrier_unit
        n_ev = 0
        for G, w, _d in part_grasps[mj]:
            # the approach along the insertion (a seat: the hand behind the part), or from above pushing it sideways
            # (onto the upper arm lying on the bench: HB goes onto J4_D sideways, a hand behind it would be at the bench)
            a_w = T_h[:3, :3] @ G[:3, 2]
            if float(a_w @ tgj["ins"]) < 0.7 and float(a_w[2]) > -0.7:
                continue
            n_ev += 1
            if n_ev > 16:
                break
            if A.place_checks(mj, tgj, G, [uj], poses=ps, width=w, fast=True, rel={uj: np.eye(4)})[0] is not None:
                return True
        return False

    if base_plan is not None:
        # (development: the base plan's units that are kept are on the bench from the start: the re-planned keep off)
        for mid_b, _p, _w in order:
            u_b = mates[mid_b]["unit"]
            if mid_b not in plan_only and u_b in base_plan["rest"]:
                p_, q_ = base_plan["rest"][u_b]
                R9 = np.zeros(9)
                import mujoco as _mj
                _mj.mju_quat2Mat(R9, np.asarray(q_, float))
                T_b = np.eye(4)
                T_b[:3, :3], T_b[:3, 3] = R9.reshape(3, 3), p_
                bench_T[u_b] = T_b
                rest[u_b] = base_plan["rest"][u_b]
                if mid_b in base_plan["grasps"]:
                    grasps[mid_b] = base_plan["grasps"][mid_b]
                c_ = A.collision_points(u_b) @ T_b[:3, :3].T + T_b[:3, 3]
                boxes.append((c_[:, 0].min(), c_[:, 0].max(), c_[:, 1].min(), c_[:, 1].max()))
                r_b = riser_of(u_b)
                if r_b is not None and r_b in base_plan.get("risers", {}):
                    _mj.mju_quat2Mat(R9, np.asarray(base_plan["risers"][r_b][1], float))
                    T_rb = np.eye(4)
                    T_rb[:3, :3], T_rb[:3, 3] = R9.reshape(3, 3), base_plan["risers"][r_b][0]
                    riser_T[r_b] = T_rb
    for k in range(len(order) - 1, -1, -1):
        mid, pn, where = order[k]
        if where == "handover":
            # the other Panda picks it from the bench: its part of the bench, any grasp it can pick with
            pn = [p["name"] for p in cell.pandas if p["name"] != pn][0]
        A.use(pn)
        u = mates[mid]["unit"]
        if u in bench_T:
            continue
        pc = [p for p in cell.pandas if p["name"] == pn][0]
        px, py = pc["pos"][0], pc["pos"][1]
        sc = scene_at(k)
        # the target with the scene: the carrier at its design (on the arm) or at its bench rest (a bench sub-assembly)
        mm = mates[mid]
        c_u = A.unit_of[mm["carrier"]]
        T_c_unit = sc.get(c_u, D.get(c_u, np.eye(4))) if c_u != "base" else D["base"]
        T_carrier = T_c_unit @ _inv(D[c_u]) @ D[mm["carrier"]] if c_u != "base" else D[mm["carrier"]]
        T_home = T_carrier @ np.array(mm["T_part_in_carrier"], float)
        T_home = T_home @ _inv(_inv(D[u]) @ D[mm["part"]]) if mm["part"] != u else T_home
        ins = T_carrier[:3, :3] @ np.asarray(mm["insertion_in_carrier"], float)
        tg = {"T": T_home, "ins": ins / np.linalg.norm(ins), "unit": u, "carrier": mm["carrier"]}
        # the group the hand carries (a bench sub-assembly's part comes with its carrier)
        group = bench_group(u, k)
        # grasps feasible at the insertion (G in the root frame; the group as designed)
        ok_g = []
        cands = A.grasp_candidates(u, D[u], toward=tg["ins"] if mm["kind"] == "seat" else None)
        rel = {x: _inv(D[u]) @ D[x] for x in group}
        for G, w, desc in cands:
            if where == "handover":
                ok_g.append((G, w, desc, 0.3))
                continue
            if where == "air":
                # its insertion is in the air (checked when it is done): a grasp from behind the part along its insertion
                # (one from its side ran the fingers into the held upper arm at every meeting pose)
                ins_u = (_inv(D[u]) @ D[mm["carrier"]])[:3, :3] @ np.asarray(mm["insertion_in_carrier"], float)
                if float(G[:3, 2] @ (ins_u / np.linalg.norm(ins_u))) > 0.7:
                    ok_g.append((G, w, desc, 0.3))
                continue
            q_pre, why = A.place_checks(mid, tg, G, group, poses=dict(sc), width=w, fast=True, rel=rel)
            if q_pre is not None:
                ok_g.append((G, w, desc, why))
        log("[bench] %-12s %d of %d grasps feasible at the insertion%s" % (mid, len(ok_g), len(cands), " (%s: any)" % where if where in ("handover", "air") else ""))
        if os.environ.get("PANDA_PLAN_DEBUG") == mid:
            for G_, w_, d_, _m in ok_g:
                log("[bench]     ok at the insertion: %s w %.0f mm at %s mm" % (d_, w_ * 1000, np.round(G_[:3, 3] * 1000)))
        if not ok_g:
            info[mid] = "no grasp at the insertion"
            continue
        # the rests: a bench carrier (it takes a part on the bench) with the part's insertion pointing down
        # (a carrier that itself goes onto another carrier on the bench rests the way that one makes it: no constraint)
        carrier_mass = sum(float(cell.m.body_mass[b]) for b in range(cell.m.nbody) if cell.unit_of_body.get(b) == u)
        takes = [] if step_of[u][1] == "bench" else [(mj, mates[mj]) for mj, _p, wj in order if wj == "bench" and A.unit_of[mates[mj]["carrier"]] == u]
        rests = A.stable_rests(u, n=12)
        if u in CRADLE_UNITS and rests:
            hv0 = rests[0][2]
            for k_ in range(4):
                a_ = k_ * math.pi / 2
                Rx = np.array([[1.0, 0, 0], [0, math.cos(a_), -math.sin(a_)], [0, math.sin(a_), math.cos(a_)]])
                rests.append((Rx, 0.0, hv0))          # the axis (the unit's x) level, turned about it
        best = None
        why_n = {"(cells tried)": 0}
        grid = [(x, y) for x in np.arange(-bx, bx + 1e-9, 0.01) for y in np.arange(-by, by + 1e-9, 0.01)]
        r_mid = 0.46
        grid.sort(key=lambda p_: abs(math.hypot(p_[0] - px, p_[1] - py) - r_mid))
        for R0, prob, hv in rests:
            # of the grasps that work at the insertion, those from above in this rest (the approach's world z under
            # any yaw is R0's), most downward first
            ok_r = sorted([g_ for g_ in ok_g if float((R0 @ g_[0][:3, 2])[2]) < -0.3], key=lambda g_: float((R0 @ g_[0][:3, 2])[2]))[:12]
            if not ok_r:
                why_n["no grasp from above in a rest"] = why_n.get("no grasp from above in a rest", 0) + 1
                continue
            for yaw in np.arange(0, 2 * math.pi - 1e-9, math.pi / 4):
                R = rotz(yaw) @ R0
                if takes:
                    mj, mmj = takes[0]
                    dn_c = np.asarray(mmj["insertion_in_carrier"], float)
                    # the insertion in the carrier's frame -> the unit root's frame -> world under R
                    Rcu = (_inv(D[u]) @ D[mmj["carrier"]])[:3, :3]
                    # the part goes on from above or from the side, not from below (the forearm lying with HA's
                    # side up leaves no grasp from above for the forearm with HA)
                    # from above; onto a heavy carrier (the upper arm, 0.92 kg: 7 N of friction on the bench) also from the
                    # side, never from below (the forearm, 0.4 kg, tipped up when HA was pushed onto it from the side)
                    if float((R @ Rcu @ dn_c)[2]) > (0.3 if carrier_mass > 0.6 else -0.7):
                        continue
                corners = hv @ R.T
                cmin, cmax = corners.min(0), corners.max(0)
                if riser_of(u) is not None:
                    r_off = rest_on_riser(A, u, R, corners, riser_of(u))     # (xy offset, yaw)
                    if r_off is None:
                        why_n["no riser spot"] = why_n.get("no riser spot", 0) + 1
                        continue
                n_tried = 0
                for x, y in grid:
                    xb, yb = (x + cmin[0], x + cmax[0]), (y + cmin[1], y + cmax[1])
                    if xb[0] < -bx or xb[1] > bx or yb[0] < -by or yb[1] > by:
                        continue
                    rr = [math.hypot(cx - px, cy - py) for cx in xb for cy in yb]
                    if min(rr) < 0.30 or max(rr) > 0.66:
                        continue
                    if any(xb[1] > q_[0] and xb[0] < q_[1] and yb[1] > q_[2] and yb[0] < q_[3] for q_ in keep):
                        continue
                    if any(xb[0] < q_[1] + gap and xb[1] > q_[0] - gap and yb[0] < q_[3] + gap and yb[1] > q_[2] - gap for q_ in boxes):
                        continue
                    n_tried += 1
                    why_n["(cells tried)"] += 1
                    if n_tried > tries_per_rest * (4 if takes else 1):     # (a carrier of a bench part: more cells)
                        break
                    T_pick = np.eye(4)
                    T_pick[:3, :3] = R
                    T_pick[:3, 3] = [x, y, -cmin[2] + 0.001]
                    psc = dict(sc)
                    T_r = None
                    if riser_of(u) is not None:
                        # on its riser: the pose it settled to on the riser in the drop test
                        T_r = np.eye(4)
                        T_r[:3, :3] = rotz(r_off[1])
                        T_r[:3, 3] = [x + r_off[0][0], y + r_off[0][1], r_off[3]]
                        T_pick = T_r @ r_off[2]
                        psc[riser_of(u)] = T_r
                    for x_ in group[1:]:
                        if x_ not in air_units:                  # (a part put on in the air is on the bench at the pick)
                            psc[x_] = T_pick @ rel[x_]
                    found = None
                    for G, w, desc, mrg_place in ok_r:
                        q_pick, q_pg, mrg1 = A.pick_checks(u, T_pick, G, poses=psc, w_open=w, fast=True, touch_units=tuple(group))
                        if q_pick is None and os.environ.get("PANDA_PLAN_DEBUG") == mid and n_tried <= 2:
                            log("[bench]     rest %s yaw %.0f at (%.2f, %.2f), %s: %s" % (np.round(R0.T @ [0, 0, 1], 1), math.degrees(yaw), x, y, desc, mrg1))
                        if q_pick is None:
                            why_n[mrg1.split(" at ")[0][:60]] = why_n.get(mrg1.split(" at ")[0][:60], 0) + 1
                            continue
                        s_ = min(mrg1, mrg_place) + 0.05 * prob
                        if found is None or s_ > found[0]:
                            found = (s_, G, w, desc)
                    if found is not None and takes and not part_fits(takes[0][0], T_pick, psc):
                        why_n["the bench part cannot go on it there"] = why_n.get("the bench part cannot go on it there", 0) + 1
                        found = None
                    if found is not None:
                        if best is None or found[0] > best[0]:
                            best = (found[0], T_pick, found, (xb[0], xb[1], yb[0], yb[1]), T_r)
                        break
            if best is not None and best[0] > 0.3:
                break
        if best is None:
            info[mid] = {"no bench place": why_n, "rests": len(rests)}
            log("[bench] %-12s NO bench place (%d rests): %s" % (mid, len(rests), sorted(why_n.items(), key=lambda kv: -kv[1])[:6]))
            continue
        T_pick = best[1]
        bench_T[u] = T_pick
        if takes:
            # a bench sub-assembly's carrier: room for the Panda to bring its part, the units placed after it keep away
            b_ = best[3]
            keep.append((b_[0] - 0.15, b_[1] + 0.15, b_[2] - 0.15, b_[3] + 0.15))
        if best[4] is not None:
            riser_T[riser_of(u)] = best[4]
        boxes.append(best[3])
        rest[u] = [T_pick[:3, 3].tolist(), quat_wxyz(T_pick[:3, :3]).tolist()]
        grasps[mid] = {"G": best[2][1].tolist(), "w": float(best[2][2]), "desc": best[2][3]}
        info[mid] = {"xy": [round(T_pick[0, 3], 3), round(T_pick[1, 3], 3)], "grasp": best[2][3], "score": round(best[0], 3)}
        log("[bench] %-12s %-5s at (%.2f, %.2f) by %s" % (mid, u, T_pick[0, 3], T_pick[1, 3], best[2][3]))
        # the carrier of a bench sub-assembly: its part comes onto it (placed with it)
    return {"rest": rest, "grasps": grasps, "info": info,
            "risers": {n: [T[:3, 3].tolist(), quat_wxyz(T[:3, :3]).tolist(), riser_span_kit(cell.kit, n[len("riser_"):])] for n, T in riser_T.items()}}


def riser_span_kit(kit, u):
    """the blocks' spacing under a unit: its longest extent (the kit's mesh of the unit) less 12 mm (each block, 10 mm
    wide, 1 mm in from an end: the fingers grip the unit's edges between them)"""
    import trimesh
    mesh = trimesh.load(os.path.join(kit["dir"], "parts", "%s.stl" % u))
    return round(float(np.max(mesh.extents)) - 0.012, 4)


def rest_on_riser(A, u, R, pts, riser, settle_s=0.6):
    """((xy offset from the unit's root, yaw)) of the riser on which the unit, turned to R, stays put: the riser along
    the unit's longest axis (horizontal in this rest, else None) under its centre of mass, then shifted along that
    axis by up to 10 mm; each dropped in the probe cell (the unit alone, the others parked off the bench), the first
    that leaves the unit at rest within 8 mm / 6 deg of where it was put (it drops 0.8 mm onto the blocks and settles):
    returns (xy offset, yaw, the settled unit pose in the riser's frame, the riser's settled height)"""
    import mujoco
    from .coupling import quat_wxyz
    cell = A.cell
    m = cell.m
    rh = cell.cfg["riser"]["block_m"][2]
    key = (u, tuple(np.round(R, 3).flatten()))
    cache = A.__dict__.setdefault("_riser_cache", {})
    if key in cache:
        return cache[key]
    # the riser along the unit's longest axis when that lies flat in this rest, else along the long axis of its
    # footprint (the principal axis of its lowest points) if that is long enough for the blocks
    span = riser_span_kit(cell.kit, u)
    low = pts[pts[:, 2] < pts[:, 2].min() + 0.004][:, :2]
    lo, hi = A.box[u]
    axes = []
    a_box = R[:, int(np.argmax(hi - lo))]
    if abs(float(a_box[2])) < 0.3:
        axes.append(a_box[:2] / np.linalg.norm(a_box[:2]))
    ev, evec = np.linalg.eigh(np.cov((low - low.mean(0)).T))
    a_pca = evec[:, int(np.argmax(ev))]
    if float(np.ptp(low @ a_pca)) >= span + 0.006:
        axes.append(a_pca)
    com = R @ A.com_local(u)
    best = None
    trials = []
    for a2 in axes:
        s_ = pts[:, :2] @ a2
        mid = 0.5 * (s_.min() + s_.max())
        for shift in (0.0, -0.005, 0.005, -0.01, 0.01):
            trials.append((a2, com[:2] + a2 * (mid - float(com[:2] @ a2) + shift)))
    for a2, c in trials:
        yaw = math.atan2(float(a2[1]), float(a2[0]))
        d = mujoco.MjData(m)
        d.qpos[:] = cell.d.qpos
        a = m.jnt_qposadr[m.joint("free_" + u).id]
        T = np.eye(4); T[:3, :3] = R
        T[:2, 3] = np.array([0.6, -0.5 if "l_" in u else 0.5]) - com[:2]        # an empty corner of the bench
        T[2, 3] = -pts[:, 2].min() + rh + 0.0008
        d.qpos[a:a + 3] = T[:3, 3]; d.qpos[a + 3:a + 7] = quat_wxyz(R)
        ar = m.jnt_qposadr[m.joint("free_" + riser).id]
        d.qpos[ar:ar + 3] = [T[0, 3] + c[0], T[1, 3] + c[1], rh / 2 + 0.0003]
        d.qpos[ar + 3:ar + 7] = quat_wxyz(rotz(yaw))
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        b = m.body(u).id
        p0, R0 = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()
        for _ in range(int(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        dp = float(np.linalg.norm(d.xpos[b] - p0))
        da = math.degrees(rotation_angle(R0.T @ d.xmat[b].reshape(3, 3)))
        rb = m.body(riser).id
        v = float(np.linalg.norm(d.qvel[m.jnt_dofadr[m.joint("free_" + u).id]:][:6]))
        # (a disc in the cradle sinks ~19 mm between the blocks from where it is put: on their tops)
        if dp < (0.025 if u in CRADLE_UNITS else 0.012) and da < 6.0 and v < 0.02:
            T_u = np.eye(4); T_u[:3, :3] = d.xmat[b].reshape(3, 3); T_u[:3, 3] = d.xpos[b]
            T_rr = np.eye(4); T_rr[:3, :3] = d.xmat[rb].reshape(3, 3); T_rr[:3, 3] = d.xpos[rb]
            T_rel = _inv(T_rr) @ T_u
            best = (c, math.atan2(T_rr[1, 0], T_rr[0, 0]), T_rel, float(T_rr[2, 3]))
            break
    cache[key] = best
    return best


def riser_names(loose):
    return ["riser_" + u for u in loose if u in RISER_UNITS]
