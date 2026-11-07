"""Collision-free motions of one Panda in the OpenArm cell (privileged: the planner sees the cell's exact state).

`Planner(cell, panda)` keeps its own copy of the cell's model and checks a posture of the Panda `panda` (its geoms
grown by `clearance_m`), with the units it holds carried rigidly by the hand, against everything else in the CURRENT
scene: the bench, the fixture, the arms built so far (at their joints' current angles), the units on the bench and the
other Panda where it is. Only the Panda and what it holds are checked against the rest (collision bits 2/1 against
1/2: the built arm's own contacts cost nothing). `free(q0, q1)` checks a joint-space segment, `plan(q0, q1)` returns
a straight segment when it is free, else an RRT-Connect path with shortcuts; `hits(q)` lists the contacts.
(After the full-arm task's `assembly_full/motion.py`; held units are whole units: a unit's child bodies follow its
root through their hinges at their current angles.)
"""
import math

import numpy as np


class Planner(object):
    def __init__(self, cell, panda, clearance_m=0.008, held_clearance_m=0.004, seed=0):
        import copy
        import mujoco
        self.mj, self.cell = mujoco, cell
        self.m = copy.deepcopy(cell.m)
        self.d = mujoco.MjData(self.m)
        self.clear = float(clearance_m)
        self.finger_clear = 0.002                       # (the fingers' clearance on planned paths)
        self.held_clear = float(held_clearance_m)
        m = self.m
        pc = [p for p in cell.pandas if p["name"] == panda][0]
        self.name = panda
        self.panda = set(cell.panda_bodies_of[panda])
        self.fingers = set(b for b in self.panda if "finger" in m.body(b).name)
        self.col = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.base_bits = {g: (1, 2) for g in self.col}
        for g in self.col:
            if m.geom_bodyid[g] in self.panda:
                m.geom_margin[g] = self.clear
                m.geom_contype[g], m.geom_conaffinity[g] = 2, 1
                self.base_bits[g] = (2, 1)
            else:
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 2
        self.held_geoms = []
        self.pj = [m.joint(panda + "/joint%d" % (i + 1)).id for i in range(7)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.pj])
        self.fq = [m.jnt_qposadr[m.joint(panda + "/finger_joint%d" % i).id] for i in (1, 2)]
        self.lo = np.array([m.jnt_range[j][0] for j in self.pj]) + 0.02
        self.hi = np.array([m.jnt_range[j][1] for j in self.pj]) - 0.02
        self.tcp = m.site(pc["tcp"]).id
        self.rng = np.random.default_rng(seed)
        self.held = []                       # [(unit root name, its pose in the tcp frame, qpos address)]
        self.held_bodies = set()
        self.allow = set()
        self.n_checks = 0
        self.unit_bodies = {}
        for b in range(m.nbody):
            u = cell.unit_of_body.get(b)
            if u is not None:
                self.unit_bodies.setdefault(u, set()).add(b)

    # ------------------------------------------------------------------ the scene
    def sync(self, held=(), allow=(), poses=None, width=None, q_other=None, touch=()):
        """... `q_other` {Panda name: posture} where another Panda would be (a handover: both hands on one unit)"""
        """take the cell's current state; `held` [(unit, G = unit root pose in the tcp frame)], `allow` body or unit
        names contacts with which are allowed, `poses` {unit: T} a hypothetical scene, `width` the finger opening"""
        from .coupling import quat_wxyz
        c = self.cell
        self.d.qpos[:] = c.d.qpos
        for name, T in (poses or {}).items():
            a = self.m.jnt_qposadr[self.m.joint("free_" + name).id]
            self.d.qpos[a:a + 3] = T[:3, 3]
            self.d.qpos[a + 3:a + 7] = quat_wxyz(T[:3, :3])
        if width is not None:
            for fq in self.fq:
                self.d.qpos[fq] = width / 2.0
        for pn, q in (q_other or {}).items():
            for i in range(7):
                self.d.qpos[self.m.jnt_qposadr[self.m.joint(pn + "/joint%d" % (i + 1)).id]] = q[i]
        m = self.m
        for g in self.held_geoms:
            m.geom_contype[g], m.geom_conaffinity[g] = self.base_bits[g]
            m.geom_margin[g] = 0.0
        self.held, self.held_geoms, self.held_bodies = [], [], set()
        for u, G in held:
            a = m.jnt_qposadr[m.joint("free_" + u).id]
            self.held.append((u, np.asarray(G, float), a))
            self.held_bodies |= self.unit_bodies[u]
        self.held_geoms = [g for g in self.col if m.geom_bodyid[g] in self.held_bodies]
        for g in self.held_geoms:
            m.geom_contype[g], m.geom_conaffinity[g] = 2, 1
            m.geom_margin[g] = self.held_clear
        al = set()
        for n in allow:
            if n in self.unit_bodies:
                al |= self.unit_bodies[n]
            else:
                al.add(m.body(n).id)
        self.allow = al
        tb = set()
        for n in touch:
            tb |= self.unit_bodies.get(n, {m.body(n).id})
        self.touch = tb

    def _place_held(self):
        from .coupling import quat_wxyz
        T = np.eye(4)
        T[:3, :3] = self.d.site_xmat[self.tcp].reshape(3, 3)
        T[:3, 3] = self.d.site_xpos[self.tcp]
        for _u, G, a in self.held:
            P = T @ G
            self.d.qpos[a:a + 3] = P[:3, 3]
            self.d.qpos[a + 3:a + 7] = quat_wxyz(P[:3, :3])

    def hits(self, q, tol_m=0.0005, finger_clear=0.002, link_clear=None, held_clear=None, world_finger_clear=None):
        """[(Panda or held body, other body, distance m)] closer than the clearance at posture q (a held unit: closer
        than its clearance - tol_m; the fingers: finger_clear, against the bench world_finger_clear; the Panda's links:
        link_clear, default the clearance)"""
        mj, m, d = self.mj, self.m, self.d
        self.n_checks += 1
        d.qpos[self.qadr] = q
        mj.mj_kinematics(m, d)
        if self.held:
            self._place_held()
            mj.mj_kinematics(m, d)
        mj.mj_collision(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            s1 = b1 in self.panda or b1 in self.held_bodies
            s2 = b2 in self.panda or b2 in self.held_bodies
            if s1 == s2:
                continue
            mine, other = (b1, b2) if s1 else (b2, b1)
            if other in self.allow:
                continue
            if other in self.touch and c.dist >= -0.0002:   # (0.2 mm: a finger laid on its face reads -0.0)
                continue                                   # may touch it (the unit being grasped), not go into it
            if mine in self.held_bodies:
                if c.dist < (self.held_clear if held_clear is None else held_clear) - tol_m:
                    out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
            elif mine in self.fingers:
                if c.dist < (finger_clear if world_finger_clear is None or other != 0 else world_finger_clear):
                    out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
            elif c.dist < 0.0 or c.dist < (self.clear if link_clear is None else link_clear):
                out.append((m.body(mine).name, m.body(other).name, float(c.dist)))
        return out

    def ok(self, q):
        return not self.hits(q, finger_clear=self.finger_clear)

    # ------------------------------------------------------------------ segments and paths
    def free(self, q0, q1, step=0.005):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q1 - q0))) / step)))
        return all(self.ok(q0 + (q1 - q0) * k / n) for k in range(1, n + 1))

    def plan(self, q0, q1, max_iter=3000, step=0.25):
        """[q0, ..., q1] collision-free (straight if possible, else RRT-Connect + shortcuts); None if not found"""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if not self.ok(q1):
            return None
        if self.free(q0, q1):
            return [q0, q1]
        ta, tb = [(q0, -1)], [(q1, -1)]

        def nearest(t, q):
            return int(np.argmin([np.linalg.norm(x[0] - q) for x in t]))

        def extend(t, q):
            i = nearest(t, q)
            v = q - t[i][0]
            dist = float(np.linalg.norm(v))
            qn = q if dist <= step else t[i][0] + v / dist * step
            if not self.free(t[i][0], qn):
                return None
            t.append((qn, i))
            return len(t) - 1

        def chain(t, i):
            out = []
            while i >= 0:
                out.append(t[i][0])
                i = t[i][1]
            return out

        swapped = False
        for it in range(max_iter):
            qr = self.lo + (self.hi - self.lo) * self.rng.random(7) if it % 5 else (tb[-1][0] if not swapped else q1)
            ia = extend(ta, qr)
            if ia is None:
                ta, tb, swapped = tb, ta, not swapped
                continue
            qa = ta[ia][0]
            while True:
                ib = extend(tb, qa)
                if ib is None:
                    break
                if np.linalg.norm(tb[ib][0] - qa) < 1e-9:
                    pa, pb = chain(ta, ia)[::-1], chain(tb, ib)
                    path = pa + pb[1:]
                    if swapped:
                        path = path[::-1]
                    return self.shortcut(path)
            ta, tb, swapped = tb, ta, not swapped
        return None

    def shortcut(self, path, rounds=60):
        path = [np.asarray(p, float) for p in path]
        for _ in range(rounds):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.free(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path
