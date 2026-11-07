"""The privileged reference assembler (the oracle's route; grader-side only).

It drives the two robots through the cell's own controllers (the same `command` / `step` a program uses) but reads
the true part poses and the joints' states. Choreography (the drawings' sequence, two persons):
  1. robot a takes the left side panel from the panel stand by its top edge (fingers on the outer face), turns it,
     stands it on the sideboard at its place and keeps holding it;
  2. robot b takes the bottom, the fixed shelf and the top from the kitting table by their front edges (fingers on top)
     and pushes each onto the left side panel's dowels;
  3. robot b takes the right side panel from the stand by its top edge (fingers on the outer face) and pushes it onto
     the three right-hand dowel groups; robot a lets go;
  4. robot b takes the back panel by its top edge (fingers on the raw face) and seats it in the rabbets from behind.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import furniture as F
from . import room as RM
from .robot import PELVIS_HEIGHT

MM = 1e-3


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def _yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def target_pose(key):
    """Room pose (p, R) of part `key` in the assembled bookcase at the floor plan's place."""
    tx, ty, tz, tyaw = RM.TARGET
    Rt = _Rz(tyaw)
    return np.array([tx, ty, tz]) + Rt @ (np.array(F.PARTS[key].design_center) * MM), Rt.copy()


def hand_frame(approach, pinch):
    """Grasp-site rotation (columns x, y, z) from its approach (x) and pinch (y) axes (room frame)."""
    x = np.asarray(approach, float)
    x = x / np.linalg.norm(x)
    y = np.asarray(pinch, float)
    y = y - (y @ x) * x
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


class Reference:
    def __init__(self, cell, log=None):
        self.c = cell
        self.m, self.d = cell.m, cell.d
        self.log = log or (lambda s: None)
        self.grip_in_part = {}        # (robot, side) -> grasp site pose in the held part's frame
        self.part_in_base = {}        # robot -> (key, part pose in the robot's base frame) at grasp time

    # ------------------------------------------------------------------------------------------- primitives
    def step(self, n=1):
        return self.c.step(n)

    def base(self, rob):
        r = self.c.robots[rob]
        return np.array(self.d.qpos[r.base_q[:3]], float)

    def site(self, rob, side):
        r = self.c.robots[rob]
        sid = r.grasp_site[side]
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def walk(self, rob, x, y, yaw, stand=None, timeout=40.0, tol=0.006):
        cmd = {"base_goal": [float(x), float(y), float(yaw)]}
        if stand is not None:
            cmd["stand_height"] = float(stand)
        self.c.command(rob, cmd)
        for _ in range(int(timeout / 0.1)):
            self.step(10)
            b = self.base(rob)
            if math.hypot(b[0] - x, b[1] - y) < tol and abs(wrap(b[2] - yaw)) < 0.01:
                break
        self.step(30)
        b = self.base(rob)
        return math.hypot(b[0] - x, b[1] - y)

    def stand(self, rob, h):
        self.c.command(rob, {"stand_height": float(h)})
        self.step(int(abs(h - self.c.ctrl[rob].stand) / 0.15 / 0.01) + 40)

    def hands(self, rob, targets, frame="room", timeout=8.0, tol=0.003, waist=None, path="line"):
        """targets: {side: (pos, R)}; waits until every site is within tol (or timeout); returns the worst error."""
        cmd = {}
        for side, (p, R) in targets.items():
            cmd[f"{side}_arm"] = {"pose": [float(v) for v in p] + [float(v) for v in _quat(R)], "frame": frame,
                                  "path": path}
        if waist is not None:
            cmd["waist"] = [float(v) for v in waist]
        self.c.command(rob, cmd)
        err = 1.0
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            if frame != "room":
                err = 0.0
                continue
            err = max(float(np.linalg.norm(self.site(rob, s)[0] - np.asarray(p))) for s, (p, _) in targets.items())
            if err < tol:
                break
        self.step(10)
        return err

    def grip(self, rob, sides=("left", "right"), close=True):
        self.c.command(rob, {f"{s}_hand": ("close" if close else "open") for s in sides})
        self.step(5)
        held = {s: self.c.grips.holding(rob, s) for s in sides}
        for s in sides:
            key = held[s]
            if close and key:
                p, R = self.site(rob, s)
                pp, Rp = self.c.part_pose(key)
                self.grip_in_part[(rob, s)] = (Rp.T @ (p - pp), Rp.T @ R)
        if not close:
            for s in sides:
                self.grip_in_part.pop((rob, s), None)
        return held

    def hand_targets_for_part(self, rob, key, p_part, R_part):
        out = {}
        for (r, s), (pg, Rg) in self.grip_in_part.items():
            if r == rob and self.c.grips.holding(rob, s) == key:
                out[s] = (p_part + R_part @ pg, R_part @ Rg)
        return out

    def move_part(self, rob, key, p_part, R_part, timeout=8.0, tol=0.003, refine=0):
        """Move the held part to (p_part, R_part) through its grips; `refine` closed-loop corrections from the part's
        measured pose (the arms sag and track with millimetres of error under load). Returns the part's position
        error (m) when refining, else the hands' error."""
        p_cmd, R_cmd = np.array(p_part, float), np.array(R_part, float)
        tg = self.hand_targets_for_part(rob, key, p_cmd, R_cmd)
        if not tg:
            self.log("[ref] %s no longer holds %s" % (rob, key))
            return 1.0
        err = self.hands(rob, tg, timeout=timeout, tol=tol)
        for _ in range(refine):
            pp, Rp = self.c.part_pose(key)
            dp = np.asarray(p_part) - pp
            dR = np.asarray(R_part) @ Rp.T
            if np.linalg.norm(dp) < 0.0006 and math.acos(float(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 0.002:
                break
            p_cmd = p_cmd + dp
            R_cmd = dR @ R_cmd
            tg = self.hand_targets_for_part(rob, key, p_cmd, R_cmd)
            if not tg:
                return 1.0
            self.hands(rob, tg, timeout=3.0, tol=0.0015)
            self.step(30)
        if refine:
            pp, _ = self.c.part_pose(key)
            err = float(np.linalg.norm(np.asarray(p_part) - pp))
        return err

    def seat_pose(self, host, guest):
        """The guest's seated pose given the host's ACTUAL pose (design relative pose)."""
        ph, Rh = self.c.part_pose(host)
        dh = np.array(F.PARTS[host].design_center) * MM
        dg = np.array(F.PARTS[guest].design_center) * MM
        return ph + Rh @ (dg - dh), Rh.copy()

    def lock_hands_to_base(self, rob):
        """Re-issue the current grasp-site poses as base-frame targets: the hands then travel with the body."""
        st = self.c.get_state()["robots"][rob]
        self.c.command(rob, {f"{s}_arm": {"pose": st["hands"][s]["grasp_pose_in_robot"], "frame": "base"}
                             for s in ("left", "right")})
        self.step(5)

    def base_for_part(self, rob, key, p_part, R_part):
        """The base pose (x, y, yaw) that puts the held part (rigid with the body) at (p_part, R_part)."""
        pp, Rp = self.c.part_pose(key)
        b = self.base(rob)
        Rb = _Rz(b[2])
        rel_p, rel_R = Rb.T @ (pp - np.array([b[0], b[1], 0.0])), Rb.T @ Rp
        yaw = _yaw_of(R_part @ rel_R.T)
        pb2 = p_part - _Rz(yaw) @ rel_p
        return float(pb2[0]), float(pb2[1]), float(yaw)

    def carry_to(self, rob, key, p_part, R_part, via=None, stand=None):
        """Walk (through `via` base waypoints, or a route around the furniture, turning on the spot first) so that
        the held part, rigid with the body, ends at (p_part, R_part)."""
        goal = self.base_for_part(rob, key, p_part, R_part)
        self.lock_hands_to_base(rob)
        if via is None:
            back = (goal[0] - 0.40 * math.cos(goal[2]), goal[1] - 0.40 * math.sin(goal[2]))
            via = self.route(rob, *back, inflate=0.42)
        for (x, y) in via:
            b = self.base(rob)
            yaw = math.atan2(y - b[1], x - b[0])
            self.walk(rob, b[0], b[1], yaw, stand=stand)
            self.walk(rob, x, y, yaw, stand=stand)
        b = self.base(rob)
        if math.hypot(goal[0] - b[0], goal[1] - b[1]) > 0.05:
            yaw = math.atan2(goal[1] - b[1], goal[0] - b[0])
            self.walk(rob, b[0], b[1], yaw, stand=stand)
            self.walk(rob, goal[0], goal[1], yaw, stand=stand)
        return self.walk(rob, *goal, stand=stand)

    # ------------------------------------------------------------------------------------------- grasp geometry
    def edge_targets(self, key, edge_local, along_local, offsets, finger_room, inset=0.022):
        """Grasp-site poses on a board edge: the edge's outward normal `edge_local` (part frame), positions `offsets`
        (m, along `along_local` from the part centre); the fingers lie on the face pointing towards `finger_room`."""
        p, R = self.c.part_pose(key)
        half = np.array(F.PARTS[key].half) * MM
        n = np.asarray(edge_local, float)
        a = np.asarray(along_local, float)
        k = int(np.argmin(F.PARTS[key].size))
        thick = np.zeros(3)
        thick[k] = 1.0
        tr = R @ thick
        if tr @ np.asarray(finger_room, float) < 0:
            tr = -tr
        approach = R @ (-n)
        dist = float(np.abs(n) @ half)
        out = {}
        for side, off in offsets.items():
            pos = p + R @ (n * (dist - inset) + a * off)
            out[side] = (pos, hand_frame(approach, tr if side == "left" else -tr))
        return out

    def walk_ready(self, rob, goal, targets_fn, back=0.05, out=0.08, up=0.10, stand=0.79, pre=0.45):
        """Walk to `goal` (x, y, yaw) with each hand held clear of the work: at its pre-grasp pose (as seen from the
        goal), moved `out` sideways away from the body's mid-line and `up` higher, in the base frame. `targets_fn()`
        gives the grasp targets from the parts' current poses."""
        gx, gy, gyaw = goal
        Rg = _Rz(gyaw)
        ready = {}
        for side, (pos, R) in targets_fn().items():
            pb = Rg.T @ (pos - R[:, 0] * back - np.array([gx, gy, 0.0]))
            pb[1] += out if side == "left" else -out
            pb[2] += up
            ready[side] = (pb, Rg.T @ R)
        # arms tucked while walking; turn only away from the work: to a point `pre` behind the goal, facing the
        # goal's way; hands to their ready poses there; then straight in
        self.tuck(rob)
        px, py = gx - pre * math.cos(gyaw), gy - pre * math.sin(gyaw)
        self.walk_path(rob, self.route(rob, px, py), stand=stand)
        self.walk(rob, px, py, gyaw, stand=stand)
        self.hands(rob, ready, frame="base", timeout=3.0, path="joint")
        return self.walk(rob, gx, gy, gyaw, stand=stand)

    def ik_cost(self, rob, side, pos, R, base, stand):
        """IK residual (position + rotation / 4) of a grasp-site pose for the robot standing at `base` (x, y, yaw)."""
        return self.ik_eval(rob, side, pos, R, base, stand)[0]

    def ik_eval(self, rob, side, pos, R, base, stand):
        """(IK residual, the wrist-pitch link's position in that solution) for the robot standing at `base`."""
        r, ctrl = self.c.robots[rob], self.c.ctrl[rob]
        qpos = self.d.qpos.copy()
        qpos[r.base_q[0]], qpos[r.base_q[1]], qpos[r.base_q[2]] = base
        qpos[r.base_q[3]] = stand - PELVIS_HEIGHT
        q, ep, er = ctrl.ik.solve_global(qpos, side, pos, _quat(R), ctrl.arm_cmd[side])
        dk = ctrl.ik.dk
        dk.qpos[:] = qpos
        dk.qpos[r.arm_q[side]] = q
        mujoco.mj_kinematics(self.m, dk)
        wp = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "%s/%s_wrist_pitch_link" % (rob, side))
        return ep + 0.25 * er, dk.xpos[wp].copy()

    def tuck(self, rob):
        from .robot import ARM_HOME
        self.c.command(rob, {f"{s}_arm": {"q": list(ARM_HOME[s])} for s in ("left", "right")})
        self.step(120)

    CORRIDOR_Y = 0.85          # south of the kitting table

    def obstacles(self, inflate=0.30, rob=None):
        """Floor rectangles (x0, y0, x1, y1) the robots keep out of, inflated by their body radius; with `rob`, the
        other robot's footprint too."""
        tx, ty = RM.TABLE["center"]
        lx, ly = RM.TABLE["size"]
        sx, sy = RM.SIDEBOARD["center"]
        wx, wy, _ = RM.SIDEBOARD["size"]
        pc = RM.PANEL_CART
        rects = [(tx - lx / 2, ty - ly / 2, tx + lx / 2, ty + ly / 2),
                 (sx - wx / 2, sy - wy / 2, sx + wx / 2, sy + wy / 2)]
        for ys in pc["slots_y"]:          # each bed block, with the board standing on it
            rects.append((pc["x"] - 0.45, ys - 0.10, pc["x"] + 0.45, ys + 0.10))
        for other in self.c.robots:
            if rob is not None and other != rob:
                ox, oy, _ = self.base(other)
                rects.append((ox - 0.30, oy - 0.30, ox + 0.30, oy + 0.30))
        return [(a - inflate, b - inflate, c + inflate, d + inflate) for a, b, c, d in rects]

    @staticmethod
    def _seg_hits(p, q, r):
        """Does segment p-q pass through rectangle r (Liang-Barsky)?"""
        x0, y0, x1, y1 = r
        dx, dy = q[0] - p[0], q[1] - p[1]
        t0, t1 = 0.0, 1.0
        for pk, qk in ((-dx, p[0] - x0), (dx, x1 - p[0]), (-dy, p[1] - y0), (dy, y1 - p[1])):
            if abs(pk) < 1e-12:
                if qk < 0:
                    return False
                continue
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
        return True

    def route(self, rob, x, y, inflate=0.30):
        """Shortest waypoint path to (x, y) around the obstacle rectangles (corners as nodes)."""
        import heapq
        b = self.base(rob)
        start, goal = (float(b[0]), float(b[1])), (float(x), float(y))
        rects = self.obstacles(inflate, rob)
        inside = lambda pt, r: r[0] < pt[0] < r[2] and r[1] < pt[1] < r[3]
        escape = []
        for r in rects:                   # a start inside an inflated obstacle first leaves it the shortest way
            if inside(start, r) and not inside(goal, r):
                cands = [(start[0] - r[0], (r[0] - 0.02, start[1])), (r[2] - start[0], (r[2] + 0.02, start[1])),
                         (start[1] - r[1], (start[0], r[1] - 0.02)), (r[3] - start[1], (start[0], r[3] + 0.02))]
                start = min(cands)[1]
                escape.append(start)
        live = [r for r in rects if not inside(start, r) and not inside(goal, r)]
        nodes = [start, goal]
        for (x0, y0, x1, y1) in self.obstacles(inflate + 0.06, rob):
            for cx in (x0, x1):
                for cy in (y0, y1):
                    if 0.35 < cx < RM.ROOM_X - 0.35 and 0.35 < cy < RM.ROOM_Y - 0.35 and \
                            not any(inside((cx, cy), r) for r in rects):
                        nodes.append((cx, cy))
        dist = {0: 0.0}
        prev = {}
        heap = [(0.0, 0)]
        while heap:
            dcur, i = heapq.heappop(heap)
            if i == 1:
                break
            if dcur > dist.get(i, 1e9):
                continue
            for j in range(len(nodes)):
                if j == i or any(self._seg_hits(nodes[i], nodes[j], r) for r in live):
                    continue
                nd = dcur + math.dist(nodes[i], nodes[j])
                if nd < dist.get(j, 1e9) - 1e-9:
                    dist[j], prev[j] = nd, i
                    heapq.heappush(heap, (nd, j))
        if 1 not in prev:
            return escape + [goal]
        path, k = [], 1
        while k != 0:
            path.append(nodes[k])
            k = prev[k]
        return escape + path[::-1]

    def walk_path(self, rob, points, stand=None):
        """Turn towards each point on the spot, walk straight to it."""
        for (x, y) in points:
            b = self.base(rob)
            if math.hypot(x - b[0], y - b[1]) < 0.02:
                continue
            heading = math.atan2(y - b[1], x - b[0])
            self.walk(rob, b[0], b[1], heading, stand=stand)
            self.walk(rob, x, y, heading, stand=stand)

    def approach_and_grip(self, rob, key, targets, back=0.03):
        pre = {s: (pos - R[:, 0] * back, R) for s, (pos, R) in targets.items()}
        e1 = self.hands(rob, pre)
        st = self.c.get_state()["robots"][rob]["hands"]
        self.log("[ref]   pre ik residuals %s" % ({s: st[s]["ik_residual"] for s in ("left", "right")},))
        e2 = self.hands(rob, targets)
        st = self.c.get_state()["robots"][rob]["hands"]
        self.log("[ref]   grasp ik residuals %s" % ({s: st[s]["ik_residual"] for s in ("left", "right")},))
        held = self.grip(rob)
        ok = all(v == key for v in held.values())
        self.log("[ref] t=%6.1f %s grips %s: %s (pre %.4f, at %.4f)" % (self.c.sim_time, rob, key, held, e1, e2))
        return ok

    # ------------------------------------------------------------------------------------------- tall boards
    def take_from_cart(self, rob, key, finger_face_local, grip_height=0.70):
        """Grip a board standing on the panel cart, fingers on the face whose outward normal (part frame) is
        `finger_face_local`; the robot stands on that face's side. Side panels are held by their front and back
        edges at `grip_height` above their bottom; the back panel by its top edge. The board is lifted clear of the
        rails and carried out of the cart."""
        p, R = self.c.part_pose(key)
        nf = R @ np.asarray(finger_face_local, float)
        nf[2] = 0.0
        nf /= np.linalg.norm(nf)
        standoff = 0.30 if key == "K" else 0.27
        stand = 0.80 if key == "K" else 0.79
        bx, by = p[0] + nf[0] * standoff, p[1] + nf[1] * standoff
        yaw = math.atan2(-nf[1], -nf[0])
        fn = lambda: self.cart_grasp_targets(key, nf, yaw, (bx, by), grip_height, rob=rob, stand=stand)
        self.walk_ready(rob, (bx, by, yaw), fn, stand=stand, pre=0.22)
        tg = fn()                                                        # the board may lean in its groove
        ok = self.approach_and_grip(rob, key, tg)
        if not ok:
            return False
        # lift clear of the rails (the cart's bed is at the sideboard's height: no further lift for carrying)
        pp, Rp = self.c.part_pose(key)
        # out of the groove along the board's own up axis: a board leaning in its groove lifted straight up wedges
        # between the rails
        e = self.move_part(rob, key, pp + Rp[:, 2] * (0.065 if key == "K" else 0.08), Rp)
        self.lock_hands_to_base(rob)
        # out of the cart without turning (a held board swings wide): a step back from the groove, then sideways
        # along it past the cart's open east end
        b = self.base(rob)
        self.walk(rob, b[0] + nf[0] * 0.12, b[1] + nf[1] * 0.12, b[2], stand=stand)
        b = self.base(rob)
        self.walk(rob, RM.PANEL_CART["x"] + RM.PANEL_CART["rail_len"] / 2 + 0.70, b[1], b[2], stand=stand)
        # upright, in the open: a board taken leaning in its groove keeps that lean in the hands (and would touch
        # the carcass when carried to its place)
        pp, Rp = self.c.part_pose(key)
        self.move_part(rob, key, pp, _Rz(math.atan2(Rp[1, 0], Rp[0, 0])), refine=1)
        self.lock_hands_to_base(rob)
        self.log("[ref] t=%6.1f %s lifted %s out of the cart (hands %.4f)" % (self.c.sim_time, rob, key, e))
        return True

    def cart_grasp_targets(self, key, nf, yaw, base_xy, grip_height, rob="b", stand=0.79):
        """Grasp-site targets on a board standing on the cart, for a robot at base_xy facing yaw."""
        bx, by = base_xy
        hz = F.PARTS[key].half[2] * MM
        if key in "LR":
            # the two vertical edges (part frame +-y), grip_height above the bottom, fingers on the facing face
            z = grip_height - hz
            tg = {}
            for side, sgn in (("left", 1.0), ("right", -1.0)):
                tg[side] = self.edge_targets(key, [0, sgn, 0], [0, 0, 1], {side: z}, finger_room=nf)[side]
            bl = _Rz(yaw).T @ (tg["left"][0] - np.array([bx, by, 0.0]))
            br = _Rz(yaw).T @ (tg["right"][0] - np.array([bx, by, 0.0]))
            if bl[1] < br[1]:              # the left hand must take the edge on the robot's left
                tg = {}
                for side, sgn in (("left", -1.0), ("right", 1.0)):
                    tg[side] = self.edge_targets(key, [0, sgn, 0], [0, 0, 1], {side: z}, finger_room=nf)[side]
        else:
            # the back panel: its vertical edges are too far apart for one robot's hands, and a hand reaching down
            # onto the top edge would need its wrist above the shoulders: the top edge is held from below, the
            # fingers pointing up and 30 deg away from the robot (the pinch axis 30 deg off the board's normal, within
            # the grip rule's 35 deg), so that the wrist stays on the robot's side of the board; 0.15 m either side of
            # the centre, the pinch axis' sign chosen by the IK
            p, R = self.c.part_pose(key)
            half = np.array(F.PARTS[key].half) * MM
            base = np.array([bx, by, 0.0])
            nrm = R @ np.array([0.0, 1.0, 0.0])
            if float(nrm[:2] @ (base[:2] - p[:2])) < 0:
                nrm = -nrm                                                       # towards the robot
            up = R @ np.array([0.0, 0.0, 1.0])
            phi = math.radians(30.0)
            appr = math.cos(phi) * up - math.sin(phi) * nrm
            pinch = math.sin(phi) * up + math.cos(phi) * nrm
            pts = [p + R @ np.array([sx * 0.15, 0.0, half[2] - 0.022]) for sx in (-1.0, 1.0)]
            pts.sort(key=lambda q: -float((_Rz(yaw).T @ (q - base))[1]))        # the robot's left first
            tg = {}
            for side, pos in zip(("left", "right"), pts):
                # of the two pinch-axis signs, the reachable one whose wrist stays farthest on the robot's side
                best = None
                for sg in (1.0, -1.0):
                    Rh = hand_frame(appr, sg * pinch)
                    cost, wpos = self.ik_eval(rob, side, pos, Rh, (bx, by, yaw), stand)
                    clear = float((wpos - p) @ nrm)
                    key_ = (cost > 0.004, -clear if cost <= 0.004 else cost)
                    if best is None or key_ < best[0]:
                        best = (key_, Rh)
                tg[side] = (pos, best[1])
        return tg

    # ------------------------------------------------------------------------------------------- phases
    def place_left_side(self):
        """Robot a: the left side panel, from the stand to its place on the sideboard; a keeps holding it."""
        ok = self.take_from_cart("a", "L", [-1.0, 0.0, 0.0])          # fingers on the outer face (-x)
        if not ok:
            return False
        pt, Rt = target_pose("L")
        above = pt + np.array([0.0, 0.0, 0.06])
        self.carry_to("a", "L", above, Rt, stand=0.79)
        e = self.move_part("a", "L", above, Rt, refine=2)
        e2 = self.move_part("a", "L", pt + np.array([0.0, 0.0, 0.002]), Rt, refine=2)
        load = self.set_down("a", "L", pt, Rt)
        pp, Rp = self.c.part_pose("L")
        self.log("[ref] t=%6.1f L placed: err %.4f m (hands %.4f / %.4f, load %.1f N)"
                 % (self.c.sim_time, float(np.linalg.norm(pp - pt)), e, e2, load))
        return True

    def hand_load(self, rob):
        st = self.c.get_state()["robots"][rob]["hands"]
        return max(st[s]["tactile_N"] for s in ("left", "right"))

    def set_down(self, rob, key, pt, Rt, load_ok=5.0, step=0.0005, max_press=0.004):
        """Lower the held board onto its support in small steps until the hands' tactile load shows that the support
        carries it (a guarded move: the stiff arms would load the grip beyond its limit if pushed further)."""
        best = (self.hand_load(rob), 0.002)
        for dz in np.arange(0.0015, -max_press - 1e-9, -step):
            tg = self.hand_targets_for_part(rob, key, pt + np.array([0.0, 0.0, dz]), Rt)
            if not tg:
                return -1.0
            self.hands(rob, tg, timeout=0.6, tol=0.0005)
            self.step(20)
            load = self.hand_load(rob)
            self.log("[ref]   set down %s: dz %+.4f load %.1f N" % (key, dz, load))
            if load < best[0]:
                best = (load, dz)
            if load < load_ok:
                return load
            if load > best[0] + 4.0:          # past the support: the arms push; go back to the lightest step
                break
        tg = self.hand_targets_for_part(rob, key, pt + np.array([0.0, 0.0, best[1]]), Rt)
        self.hands(rob, tg, timeout=0.6, tol=0.0005)
        self.step(20)
        return self.hand_load(rob)

    def wait_locked(self, host, guest, timeout=3.0):
        for _ in range(int(timeout / 0.05)):
            if self.c.coupling.state[(host, guest)]["state"] == "LOCKED":
                return True
            self.step(5)
        return self.c.coupling.state[(host, guest)]["state"] == "LOCKED"

    def let_go(self, rob, back=0.06, up=0.05, retreat=0.25):
        """Open both hands, draw them back along their approach and up, step back from the work (arms still out, so
        that tucking them sweeps nothing), tuck the arms."""
        held = [self.c.grips.holding(rob, s) for s in ("left", "right")]
        tg = {}
        for s in ("left", "right"):
            p, R = self.site(rob, s)
            tg[s] = (p - R[:, 0] * back + np.array([0.0, 0.0, up]), R)
        self.grip(rob, close=False)
        self.hands(rob, tg, timeout=2.0)
        if retreat:
            self.lock_hands_to_base(rob)
            b = self.base(rob)
            self.walk(rob, b[0] - retreat * math.cos(b[2]), b[1] - retreat * math.sin(b[2]), b[2],
                      stand=self.c.ctrl[rob].stand_cmd)
        self.tuck(rob)
        return held

    def take_from_table(self, rob, key):
        """Grip a horizontal board lying on the table battens by its front (banded) edge, fingers on top, from the
        side of the table the edge faces; lift it and back away from the table."""
        p, R = self.c.part_pose(key)
        front = R @ np.array([0.0, -1.0, 0.0])
        front[2] = 0.0
        front /= np.linalg.norm(front)
        half_d = F.PARTS[key].half[1] * MM
        edge_mid = p + front * half_d
        bx, by = edge_mid[0] + front[0] * 0.28, edge_mid[1] + front[1] * 0.28
        yaw = math.atan2(-front[1], -front[0])

        def targets():
            tg = self.edge_targets(key, [0, -1, 0], [1, 0, 0], {"left": 0.2, "right": -0.2}, finger_room=[0, 0, 1])
            bl = _Rz(yaw).T @ (tg["left"][0] - np.array([bx, by, 0.0]))
            br = _Rz(yaw).T @ (tg["right"][0] - np.array([bx, by, 0.0]))
            if bl[1] < br[1]:
                tg = self.edge_targets(key, [0, -1, 0], [1, 0, 0], {"left": -0.2, "right": 0.2}, finger_room=[0, 0, 1])
            return tg

        self.walk_ready(rob, (bx, by, yaw), targets, stand=0.76)
        if not self.approach_and_grip(rob, key, targets()):
            return False
        pp, Rp = self.c.part_pose(key)
        self.move_part(rob, key, pp + np.array([0.0, 0.0, 0.08]), Rp)
        self.lock_hands_to_base(rob)
        b = self.base(rob)
        self.walk(rob, b[0] + front[0] * 0.35, b[1] + front[1] * 0.35, b[2], stand=0.76)
        return True

    def insert_horizontal(self, rob, key, slot_key, host="L", via=()):
        """Carry a held horizontal board to 25 mm beside its seat in `host` and push it onto the dowels."""
        pt, Rt = self.seat_pose(host, slot_key)
        side = -1.0 if host == "L" else 1.0
        low = pt[2] - F.PARTS[key].half[2] * MM < RM.SIDEBOARD_TOP + 0.01     # lies on the cabinet when seated
        pre = pt + Rt @ np.array([-side * 0.025, 0.0, 0.0]) + np.array([0.0, 0.0, 0.003 if low else 0.0])
        carry = pre + Rt @ np.array([0.0, -0.06, 0.0])          # in front of the seat, then in
        z_hands = carry[2]
        stand = 0.55 if low else float(min(0.79, max(0.55, z_hands + 0.02)))
        waist = [0.0, 0.0, 0.5] if low else ([0.0, 0.0, 0.35] if z_hands < 0.75 else None)
        lift = np.array([0.0, 0.0, max(0.0, 0.86 - carry[2])])   # carried above the cabinet top, lowered in place
        high = carry[2] > 1.2
        walk_stand = 0.79 if high else 0.76
        # raised to its working height in the open (rising under the carcass it would hit the boards already in)
        pp, Rp = self.c.part_pose(key)
        if carry[2] + lift[2] > pp[2] + 0.03:
            self.stand(rob, walk_stand)
            self.move_part(rob, key, np.array([pp[0], pp[1], carry[2] + lift[2]]), Rp)
        self.carry_to(rob, key, carry + lift, Rt, stand=walk_stand)
        if low or high:
            # squatting, the hips come 0.33 m forward: the base 0.42 m behind the seat's front edge keeps them clear
            # of the cabinet (the hands then reach 0.44 m ahead); reaching up to the top, the base 0.30 m behind it
            want = 0.42 if low else 0.30
            b = self.base(rob)
            h = np.array([math.cos(b[2]), math.sin(b[2])])
            front = (pt + Rt @ np.array([0.0, -F.PARTS[key].half[1] * MM, 0.0]))[:2]
            s_ahead = float((front - b[:2]) @ h)
            self.walk(rob, b[0] + h[0] * (s_ahead - want), b[1] + h[1] * (s_ahead - want), b[2], stand=walk_stand)
        if abs(stand - walk_stand) > 1e-3:
            self.stand(rob, stand)
        if waist is not None:
            self.c.command(rob, {"waist": waist})
            self.step(60)
        e1 = self.move_part(rob, key, carry, Rt)
        e2 = self.move_part(rob, key, pre, Rt, refine=3)
        e3 = self.move_part(rob, key, pt + np.array([0.0, 0.0, 0.0008 if low else 0.0]), Rt, timeout=4.0, refine=2)
        ok = self.wait_locked(host, key)
        st = self.c.coupling.state[(host, key)]
        self.log("[ref] t=%6.1f %s -> %s.%s: %s (hands %.4f, part %.4f %.4f)" % (self.c.sim_time, key, host, slot_key,
                                                                              st["state"], e1, e2, e3))
        if not ok:
            return False                  # keep holding: a released unlocked board would fall
        self.let_go(rob)
        if waist is not None:
            self.c.command(rob, {"waist": [0.0, 0.0, 0.0]})
            self.step(60)
        if abs(stand - 0.76) > 1e-3:
            self.stand(rob, 0.76)
        b = self.base(rob)
        h = np.array([math.cos(b[2]), math.sin(b[2])])
        self.walk(rob, b[0] - 0.30 * h[0], b[1] - 0.30 * h[1], b[2])   # step back from the cabinet
        return ok

    def insert_panels(self, which=("C", "D", "T")):
        ok = True
        for key, slot in (("C", "C"), ("D", "D"), ("T", "T")):
            if key not in which:
                continue
            if not self.take_from_table("b", key):
                self.log("[ref] b could not take %s" % key)
                return False
            ok &= self.insert_horizontal("b", key, slot)
        return ok

    def place_right_side(self):
        """Robot b: the right side panel from the cart, pushed from the east onto the three right-hand dowel groups;
        then robot a lets go of the left side panel."""
        if not self.take_from_cart("b", "R", [1.0, 0.0, 0.0]):              # fingers on the outer face (+x)
            return False
        pt, Rt = self.seat_pose("L", "R")
        pre = pt + Rt @ np.array([0.025, 0.0, 0.0])
        self.carry_to("b", "R", pre + Rt @ np.array([0.06, 0.0, 0.05]), Rt, stand=0.79)
        e1 = self.move_part("b", "R", pre + np.array([0.0, 0.0, 0.002]), Rt, refine=3)
        e2 = self.move_part("b", "R", pt + np.array([0.0, 0.0, 0.002]), Rt, timeout=4.0, refine=2)
        ok = all(self.wait_locked("R", g, timeout=3.0) for g in ("T", "D", "C"))
        states = {g: self.c.coupling.state[("R", g)]["state"] for g in ("T", "D", "C")}
        self.log("[ref] t=%6.1f R on the dowels: %s (hands %.4f %.4f)" % (self.c.sim_time, states, e1, e2))
        self.let_go("b")
        b = self.base("b")
        self.walk("b", b[0] + 0.30, b[1], b[2])
        self.let_go("a")
        self.park("a")
        return ok

    # in front of the couch: the parked robot's legs stay 0.2 m clear of its seat (x >= 5.09), and the corner of the
    # other robot's way round the cabinet's north side, (4.16, 3.84), stays outside this robot's routing footprint
    PARK = {"a": (4.70, 4.50, math.pi / 2)}

    def park(self, rob):
        """Walk (arms tucked) to the robot's parking spot in the north-east corner, in front of the couch, out of the
        other robot's routes (between the cart, the table and the cabinet every spot blocks one of them)."""
        x, y, yaw = self.PARK[rob]
        self.tuck(rob)
        self.walk_path(rob, self.route(rob, x, y), stand=0.79)
        self.walk(rob, x, y, yaw, stand=0.79)

    def seat_back(self):
        """Robot b: the back panel from the cart by its top edge (fingers on the raw face), into the rabbets from
        behind."""
        if not self.take_from_cart("b", "K", [0.0, 1.0, 0.0]):              # fingers on the raw face (+y)
            return False
        pt, Rt = self.seat_pose("L", "K")
        pre = pt + Rt @ np.array([0.0, 0.03, 0.003])
        self.carry_to("b", "K", pre, Rt, stand=0.80)          # the panel 30 mm behind its seat, clear of the sides
        e1 = self.move_part("b", "K", pre, Rt, refine=3)
        # in to 2.5 mm short of the seat: inside the rabbets' engagement depth, its front face still clear of the
        # boards' back edges (1 mm behind them when seated); held still until the rabbets engage, then let go: they
        # pull the panel home and lock it (stiff arms pressing it on would load the grip beyond its limit)
        e2 = self.move_part("b", "K", pt + Rt @ np.array([0.0, 0.0025, 0.002]), Rt, timeout=4.0, refine=2)
        for _ in range(60):
            if any(self.c.coupling.state[(h, "K")]["state"] in ("ENGAGED", "LOCKED") for h in ("L", "R")):
                self.grip("b", close=False)
                break
            self.step(5)
        ok = self.wait_locked("L", "K", timeout=4.0) or self.wait_locked("R", "K", timeout=0.5)
        states = {h: self.c.coupling.state[(h, "K")]["state"] for h in ("L", "R")}
        self.log("[ref] t=%6.1f back panel: %s (hands %.4f %.4f)" % (self.c.sim_time, states, e1, e2))
        self.let_go("b")
        b = self.base("b")
        self.walk("b", b[0], b[1] + 0.30, b[2])
        return ok

    def run(self):
        steps = (("left side", self.place_left_side), ("panels", self.insert_panels),
                 ("right side", self.place_right_side), ("back panel", self.seat_back))
        for name, fn in steps:
            try:
                ok = fn()
            except Exception as e:  # noqa: BLE001 - the reference reports, the score tells
                self.log("[ref] %s raised %r" % (name, e))
                ok = False
            self.log("[ref] t=%6.1f %s %s" % (self.c.sim_time, name, "done" if ok else "FAILED"))
        return True
