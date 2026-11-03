"""Scoring rule: a piece counts iff it passed THROUGH its own matching hole into the box.

Tracked at every physics step (2 ms) on the piece centroid c, expressed in the cube's body frame (the scene's actual
cube placement):
  * a face crossing happens when c moves from the outer side to the inner side of a face's mid-wall plane
    (+x, +y, -y side walls at |n.c| = 0.065 m, the lid at z = plane_offset, and the hole-free -x wall and floor);
    the crossing point is classified as hole h if it lies inside h's outline dilated by HOLE_MARGIN, else as "wall";
  * an outward crossing of the same planes clears the record.
At the end (after the post-plan hold) piece i is PASSED iff
  (1) its last inward crossing was through hole(i), and
  (2) every one of its hull points (geometry.json) is inside the cavity box enlarged by INSIDE_TOL.
Instance score = passed pieces / 11.
"""
import numpy as np

HOLE_MARGIN = 0.003     # m
INSIDE_TOL = 0.001      # m


def _point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xc = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xc:
                inside = not inside
        j = i
    return inside


def _dist_to_poly(pt, poly):
    p = np.asarray(pt, float)
    a = np.asarray(poly, float)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(proj - p, axis=1)))


class Tracker:
    def __init__(self, geometry, box_pos, box_quat=(0.0, 0.0, 0.0, 1.0)):
        from .quat import conj
        self.g = geometry
        self.box_pos = np.asarray(box_pos, float)
        self.box_qinv = conj(np.asarray(box_quat, float))
        cav = geometry["cavity"]
        self.cav_lo = np.array([cav["xmin"], cav["ymin"], cav["zmin"]])
        self.cav_hi = np.array([cav["xmax"], cav["ymax"], cav["zmax"]])
        # planes: (name, normal, offset, u, v, extent box in (u, v), holes)
        holes = geometry["holes"]
        self.planes = []
        defs = {"+x": ([1, 0, 0], [0, 1, 0], [0, 0, 1]), "+y": ([0, 1, 0], [1, 0, 0], [0, 0, 1]),
                "-y": ([0, -1, 0], [1, 0, 0], [0, 0, 1]), "-x": ([-1, 0, 0], [0, 1, 0], [0, 0, 1]),
                "lid": ([0, 0, 1], [1, 0, 0], [0, 1, 0]), "floor": ([0, 0, -1], [1, 0, 0], [0, 1, 0])}
        side_mid = 0.5 * (0.07 + abs(cav["xmax"]))
        lid_mid = [h["plane_offset"] for h in holes if h["face"] == "lid"][0]
        offs = {"+x": side_mid, "+y": side_mid, "-y": side_mid, "-x": side_mid, "lid": lid_mid, "floor": -0.5 * cav["zmin"]}
        for name, (n, u, v) in defs.items():
            hs = [(h["id"], np.array(h["polygon_uv"], float)) for h in holes if h["face"] == name]
            self.planes.append((name, np.array(n, float), float(offs[name]), np.array(u, float), np.array(v, float), hs))
        self.own = {p: geometry["pieces"][p]["hole"] for p in geometry["pieces"]}
        self.hull = {p: np.array(geometry["pieces"][p]["hull"], float) for p in geometry["pieces"]}
        self.prev = {}
        self.last = {}
        self.events = {}

    def _to_box(self, x):
        from .quat import apply
        return apply(self.box_qinv, np.asarray(x, float) - self.box_pos)

    def reset(self, centroids):
        self.prev = {k: self._to_box(v) for k, v in centroids.items()}
        self.last = {k: None for k in centroids}
        self.events = {k: 0 for k in centroids}

    def state(self):
        return dict(prev={k: v.copy() for k, v in self.prev.items()}, last=dict(self.last), events=dict(self.events))

    def set_state(self, st):
        self.prev = {k: v.copy() for k, v in st["prev"].items()}
        self.last = dict(st["last"])
        self.events = dict(st["events"])

    def _in_extent(self, name, pt3):
        x, y, z = pt3
        if name in ("lid", "floor"):
            return abs(x) <= 0.07 and abs(y) <= 0.07
        if name in ("+x", "-x"):
            return abs(y) <= 0.07 and -0.001 <= z <= 0.142
        return abs(x) <= 0.07 and -0.001 <= z <= 0.142

    def update(self, centroids):
        for k, c in centroids.items():
            c = self._to_box(c)
            p = self.prev[k]
            for name, n, off, u, v, hs in self.planes:
                sp_, sc = float(n @ p) - off, float(n @ c) - off
                if (sp_ > 0.0) == (sc > 0.0):
                    continue
                t = sp_ / (sp_ - sc)
                x = p + t * (c - p)
                if not self._in_extent(name, x):
                    continue
                if sp_ > 0.0 >= sc:  # inward
                    uv = (float(u @ x), float(v @ x))
                    hit = "wall"
                    for hid, poly in hs:
                        if _point_in_poly(uv, poly) or _dist_to_poly(uv, poly) <= HOLE_MARGIN:
                            hit = hid
                            break
                    self.last[k] = hit
                    self.events[k] += 1
                else:  # outward
                    self.last[k] = None
                    self.events[k] += 1
            self.prev[k] = c

    def result(self, poses, nonfinite=False, sim_time=None):
        from .quat import apply
        out = {}
        for k in self.own:
            p, q = poses[k]
            pts = np.array([self._to_box(apply(q, h) + p) for h in self.hull[k]])
            inside = bool(np.all(pts >= self.cav_lo - INSIDE_TOL) and np.all(pts <= self.cav_hi + INSIDE_TOL))
            entry = self.last[k]
            passed = bool(inside and entry == self.own[k])
            out[k] = dict(passed=passed, inside=inside, entry=entry, own_hole=self.own[k])
        if nonfinite:   # a scene whose simulation went non-finite scores 0
            for v in out.values():
                v["passed"] = False
        n = sum(1 for v in out.values() if v["passed"])
        return dict(pieces=out, passed=n, score=n / float(len(self.own)), nonfinite=bool(nonfinite), sim_time=sim_time)
