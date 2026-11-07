"""Forward kinematics of a unit described by a `bimanual-cad-unit/1` mechanism (CONTRACTS 6): a tree of revolute
joints from a grounded root, plus closed four-bar loops whose bodies follow a driven tree joint.

Every pose is the rigid motion of a link from configuration Z (identity at q = 0), in the frame the mechanism is
written in (the base frame B, mm). The same code poses the reference unit and a submitted one (the converter writes the
submission's mechanism in the same format), so both are moved by one rule.

A loop {"joints": [crank joint, pin 1, pin 2, driven joint]} is solved in the plane normal to its (parallel) axes:
the crank turns about its joint axis on the crank joint's parent until the coupler's two pins are again their Z distance
apart, on the root nearest the driven joint's angle times the loop's ratio (for an exact parallelogram the crank angle
equals the driven angle and the coupler translates).
"""
import math

import numpy as np


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot_about(point, d, ang):
    """4x4 rotation by `ang` about the line through `point` along `d` (right-handed)."""
    d = _unit(d)
    K = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    R = np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    p = np.asarray(point, float)
    T[:3, 3] = p - R @ p
    return T


def apply(T, p):
    return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]


class LoopError(ValueError):
    """The four-bar cannot close at the requested configuration."""


class Mechanism(object):
    def __init__(self, unit):
        self.unit = unit
        self.links = [l["name"] for l in unit["links"]]
        self.joints = {j["name"]: j for j in unit["joints"]}
        tree = [j for j in unit["joints"] if j.get("tree", True)]
        children = set(j["child"] for j in tree)
        roots = [n for n in self.links if n not in children]
        loop_bodies = set()
        for lp in unit.get("loops", []):
            crank, p1 = self.joints[lp["joints"][0]], self.joints[lp["joints"][1]]
            loop_bodies.update([crank["child"], p1["child"]])
        roots = [r for r in roots if r not in loop_bodies]
        if len(roots) != 1:
            raise ValueError("the mechanism has %d roots (%s)" % (len(roots), ", ".join(roots)))
        self.root = roots[0]
        # tree joints parents-first
        order, have, todo = [], {self.root}, list(tree)
        while todo:
            nxt = [j for j in todo if j["parent"] in have]
            if not nxt:
                raise ValueError("tree joints not connected to the root: %s" % ", ".join(j["name"] for j in todo))
            for j in nxt:
                order.append(j)
                have.add(j["child"])
                todo.remove(j)
        self.order = order
        self.loops = list(unit.get("loops", []))

    def tree_joints(self):
        return [j["name"] for j in self.order]

    def poses(self, q):
        """{link: 4x4 motion from Z} for joint angles q ({tree joint name: rad}; missing = 0). Raises LoopError when a
        loop cannot close."""
        T = {self.root: np.eye(4)}
        for j in self.order:
            Tp = T[j["parent"]]
            a = float(q.get(j["name"], 0.0))
            T[j["child"]] = Tp @ rot_about(j["axis_point"], j["axis_dir"], a) if a else Tp.copy()
        for lp in self.loops:
            jc, j1, j2, jd = (self.joints[n] for n in lp["joints"])
            Tw = T[jc["parent"]]
            n = _unit(Tw[:3, :3] @ np.asarray(jc["axis_dir"], float))
            a = apply(Tw, jc["axis_point"])
            b0 = apply(Tw, j1["axis_point"])                      # pin 1 if the crank stayed at its Z angle
            P2 = apply(T[j2["child"]], j2["axis_point"])           # pin 2 rides on the driven link
            h1, h2 = np.asarray(j1["axis_point"], float), np.asarray(j2["axis_point"], float)
            nz = _unit(jc["axis_dir"])
            Lc = np.linalg.norm((h2 - h1) - np.dot(h2 - h1, nz) * nz)
            u = (b0 - a) - np.dot(b0 - a, n) * n
            v = (P2 - a) - np.dot(P2 - a, n) * n
            r2 = float(u @ u)
            K = 0.5 * (r2 + float(v @ v) - Lc * Lc)
            A, B = float(u @ v), float(np.cross(n, u) @ v)
            rho = math.hypot(A, B)
            if rho < 1e-12 or abs(K) > rho * (1 + 1e-12):
                raise LoopError("loop %s cannot close" % "/".join(lp["joints"]))
            alpha = math.atan2(B, A)
            beta = math.acos(max(-1.0, min(1.0, K / rho)))
            ref = float(lp.get("ratio", 1.0)) * float(q.get(jd["name"], 0.0))
            cands = [alpha + beta, alpha - beta]
            phi = min(cands, key=lambda c: abs(math.atan2(math.sin(c - ref), math.cos(c - ref))))
            phi = ref + math.atan2(math.sin(phi - ref), math.cos(phi - ref))
            Rc = rot_about(a, n, phi)
            T[jc["child"]] = Rc @ Tw
            b = apply(Rc, b0)
            # the coupler: the planar rigid motion taking its Z pins (h1, h2) to (b, P2)
            e0 = (h2 - h1) - np.dot(h2 - h1, nz) * nz
            e1 = (P2 - b) - np.dot(P2 - b, n) * n
            Rw = Tw[:3, :3]
            e0w = Rw @ e0                                         # the Z direction carried by the parent's rotation
            psi = math.atan2(float(np.cross(e0w, e1) @ n), float(e0w @ e1))
            Tcp = rot_about(np.zeros(3), n, psi) @ np.vstack([np.hstack([Rw, np.zeros((3, 1))]), [0, 0, 0, 1]])
            Tcp[:3, 3] = b - Tcp[:3, :3] @ h1
            T[j1["child"]] = Tcp
        return T

    def point(self, T, link, p):
        return apply(T[link], p)
