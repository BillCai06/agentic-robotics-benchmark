"""Geometry and scoring helpers of the tree grader: transforms, B-rep reading and moving, watertight tessellation,
manifold booleans, per-part rigid alignment (point-to-plane ICP), band scores, vertex fingerprints, quasi-random draws.

Vendored with the rest of cad_tree into the task's grader folder (tests/), so it imports only the standard library, numpy,
scipy, trimesh, manifold3d and cadquery / OCP. Lengths mm, angles rad unless a name says otherwise.

Provenance (copied, then adapted where noted):
  - fill_small_holes, solid_to_trimesh, trimesh_to_manifold, manifold_transform, Body, band, score_ratio, ring_score,
    vertex_fingerprint, geometry_invariants, invariants_match(_similar): chain/shared/arm/cad_drawings/checks.py
    (Body: OCP shapes as well as cadquery ones; `strict=False` for given bought geometry; the mesh area kept);
  - sample_surface, surface_distances, icp, trimmed_rms, pca_starts, align_part: chain/shared/arm/cad_photo/align.py
    (align_part: the destination sample count grows with the part's area, so a 250 mm plate is sampled as densely as
    a 40 mm bracket; restricted ICP, then the 24-start multistart, unchanged);
  - transform_shape: tests/fcstd_convert.py of robotic_arm_cad_design_from_photos (`_transform_solid`, the rotation
    re-orthonormalised by SVD: skill pitfalls 5).
"""
import hashlib
import itertools
import json
import math
import os

import numpy as np

TESS_TOL, TESS_ANG = 0.05, 0.3
RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.json")


def load_rules(path=None):
    """The grader's rules (rules.json beside this module unless a path is given)."""
    with open(path or RULES_PATH) as fh:
        return json.load(fh)


def limit_cpus(n):
    """Pin this process to n of its CPUs (Linux). manifold3d's TBB pool and OCC's mesher size their thread pools to the
    machine, not to OMP_NUM_THREADS (64 threads on a 64-core host), so a CPU budget needs an affinity mask (or a cgroup
    limit). Call it first, before any boolean or mesh: threads started later inherit the mask. Returns the CPUs used."""
    cpus = sorted(os.sched_getaffinity(0))[:max(1, int(n))]
    os.sched_setaffinity(0, cpus)
    return cpus


# ----------------------------------------------------------------------------- transforms
def unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def inv(T):
    T = np.asarray(T, float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def apply(T, p):
    """T (4x4) applied to one point (3,) or to points (n, 3)."""
    T = np.asarray(T, float)
    p = np.asarray(p, float)
    if p.ndim == 1:
        return T[:3, :3] @ p + T[:3, 3]
    return p @ T[:3, :3].T + T[:3, 3]


def orthonormal(T):
    """T with its rotation replaced by the nearest proper rotation (SVD)."""
    T = np.array(T, float)
    U, _s, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    T[:3, :3] = R
    T[3] = [0.0, 0.0, 0.0, 1.0]
    return T


def frame_z(origin, z):
    """4x4 with the given origin and z axis; x the world axis least parallel to z, made perpendicular."""
    z = unit(z)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = unit(ref - z * float(ref @ z))
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, np.asarray(origin, float)
    return T


def rot_about(point, d, ang):
    """4x4 rotation by `ang` about the line through `point` along `d` (right-handed)."""
    d = unit(d)
    K = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    R = np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    p = np.asarray(point, float)
    T[:3, 3] = p - R @ p
    return T


def angle_deg(a, b):
    """Angle between two directions, 0 .. 180 deg."""
    a, b = unit(a), unit(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def line_angle_deg(a, b):
    """Angle between two lines (direction sign ignored), 0 .. 90 deg."""
    g = angle_deg(a, b)
    return min(g, 180.0 - g)


def signed_angle(x1, x2, z):
    """Angle from x1 to x2 about z (right-handed), -pi .. pi."""
    return math.atan2(float(np.cross(x1, x2) @ z), float(np.dot(x1, x2)))


def point_line_distance(p, a, d):
    """Distance of point p from the line through a along d."""
    d = unit(d)
    v = np.asarray(p, float) - np.asarray(a, float)
    return float(np.linalg.norm(v - d * float(v @ d)))


def project_on_line(p, a, d):
    d = unit(d)
    a = np.asarray(a, float)
    return a + d * float((np.asarray(p, float) - a) @ d)


def line_line_distance(p1, d1, p2, d2):
    """Shortest distance between two lines (parallel lines: their perpendicular distance)."""
    d1, d2 = unit(d1), unit(d2)
    n = np.cross(d1, d2)
    if float(np.linalg.norm(n)) < 1e-9:
        return point_line_distance(p2, p1, d1)
    return float(abs((np.asarray(p2, float) - np.asarray(p1, float)) @ unit(n)))


def closest_point_to_lines(lines):
    """The point minimising the summed squared distances to the lines [(point, dir)] (least squares)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for p, d in lines:
        d = unit(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ np.asarray(p, float)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def rotation_angle_deg(R):
    """Angle of a rotation matrix (atan2 of its skew and symmetric parts: exact near zero, where acos of the trace is not)."""
    R = np.asarray(R, float)[:3, :3]
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return math.degrees(math.atan2(float(np.linalg.norm(v)), 0.5 * (float(np.trace(R)) - 1.0)))


# ----------------------------------------------------------------------------- B-rep shapes (OCP)
def read_brep(path):
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    s = TopoDS_Shape()
    if not BRepTools.Read_s(s, str(path), BRep_Builder()):
        raise IOError("cannot read %s" % path)
    return s


def write_brep(shape, path):
    from OCP.BRepTools import BRepTools
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    BRepTools.Write_s(_wrapped(shape), str(path))
    return path


def write_step(shape, path):
    import cadquery as cq
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cq.exporters.export(cq.Workplane().add(cq.Shape.cast(_wrapped(shape))), str(path), exportType="STEP")
    return path


def _wrapped(shape):
    return shape.wrapped if hasattr(shape, "wrapped") else shape


def load_shape(path):
    """(OCP shape, error) of a shape file: the .brep beside a .step when there is one (a STEP round trip can turn a
    valid solid invalid: skill pitfalls 1), else the STEP through cadquery (all its solids in one compound)."""
    base, ext = os.path.splitext(str(path))
    brep = base + ".brep"
    try:
        if os.path.isfile(brep):
            return read_brep(brep), None
        if ext.lower() in (".step", ".stp") and os.path.isfile(path):
            import cadquery as cq
            wp = cq.importers.importStep(str(path))
            vals = [v.wrapped for v in wp.vals() if hasattr(v, "wrapped")]
            if not vals:
                return None, "no shape in %s" % os.path.basename(path)
            return (vals[0] if len(vals) == 1 else compound(vals)), None
    except Exception as exc:  # noqa: BLE001 - an unreadable file is the caller's error string
        return None, "%s: %s" % (os.path.basename(str(path)), str(exc)[:200])
    return None, "no file %s" % os.path.basename(str(path))


def solids_of(shape):
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    ex = TopExp_Explorer(_wrapped(shape), TopAbs_SOLID)
    out = []
    while ex.More():
        out.append(TopoDS.Solid_s(ex.Current()))
        ex.Next()
    return out


def compound(shapes):
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    c = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, _wrapped(s))
    return c


def is_valid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return bool(BRepCheck_Analyzer(_wrapped(shape)).IsValid())


def brep_volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(_wrapped(shape), g)
    return float(g.Mass())


def brep_centroid(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(_wrapped(shape), g)
    c = g.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()])


def transform_shape(shape, T):
    """A copy of an OCP shape moved by the rigid 4x4 T (the rotation re-orthonormalised: OCC rejects a skewed one)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    Tn = orthonormal(T)
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in Tn[:3, :].ravel()))
    return BRepBuilderAPI_Transform(_wrapped(shape), tr, True).Shape()


# ----------------------------------------------------------------------------- meshes
def fill_small_holes(mesh, max_loop=64):
    """Close boundary loops of at most `max_loop` edges with triangle fans (the OCC mesher leaves a few micro-gaps
    on some imported B-reps). Returns (mesh, n_loops_closed, n_loops_left)."""
    import trimesh
    edges = mesh.edges_sorted
    bidx = trimesh.grouping.group_rows(edges, require_count=1)
    if len(bidx) == 0:
        return mesh, 0, 0
    directed = mesh.edges[bidx]                    # face-directed boundary edges (a -> b)
    nxt = {}
    for a, b in directed:
        nxt.setdefault(int(a), []).append(int(b))
    used = set()
    loops = []
    for a, b in directed:
        key = (int(a), int(b))
        if key in used:
            continue
        loop = [int(a)]
        cur = int(b)
        used.add(key)
        ok = True
        for _ in range(len(directed) + 1):
            if cur == loop[0]:
                break
            loop.append(cur)
            cands = [c for c in nxt.get(cur, []) if (cur, c) not in used]
            if not cands:
                ok = False
                break
            c = cands[0]
            used.add((cur, c))
            cur = c
        if ok and cur == loop[0] and len(loop) >= 3:
            loops.append(loop)
    new_faces = []
    left = 0
    for loop in loops:
        if len(loop) > max_loop:
            left += 1
            continue
        v0 = loop[0]
        for i in range(1, len(loop) - 1):
            new_faces.append([v0, loop[i + 1], loop[i]])   # reversed relative to the boundary direction
    if not new_faces:
        return mesh, 0, len(loops)
    faces = np.vstack([mesh.faces, np.asarray(new_faces, dtype=np.int64)])
    out = trimesh.Trimesh(mesh.vertices.copy(), faces, process=True)
    return out, len(loops) - left, left


def fresh_copy(shape):
    """A copy of an OCP shape without any stored triangulation (from chain/shared/arm/gt_arm/tessellate.py `_fresh`):
    the incremental mesher reuses a triangulation it finds, so a tessellation would depend on the call history, and
    re-meshing a B-spline face over its own earlier, broken triangulation can run without end."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepTools import BRepTools
    c = BRepBuilderAPI_Copy(_wrapped(shape), True, False).Shape()
    BRepTools.Clean_s(c)
    return c


def solid_to_trimesh(solid, tol=TESS_TOL, ang=TESS_ANG):
    """Watertight trimesh of a solid (OCP or cadquery; a fresh copy meshed; degenerate faces dropped, micro-gaps
    closed). (mesh, notes)."""
    import cadquery as cq
    import trimesh
    s = cq.Shape.cast(fresh_copy(solid))
    vs, tris = s.tessellate(tol, ang)
    verts = np.array([v.toTuple() for v in vs], dtype=float)
    faces = np.array(tris, dtype=np.int64)
    if len(faces) == 0:
        return None, ["tessellation produced no faces"]
    mesh = trimesh.Trimesh(verts, faces, process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    notes = []
    if not mesh.is_watertight:
        mesh, closed, left = fill_small_holes(mesh)
        if closed:
            notes.append("closed {} micro-gap loop(s) of the tessellation".format(closed))
        if left:
            notes.append("{} boundary loop(s) too large to close".format(left))
    return mesh, notes


def trimesh_to_manifold(mesh):
    import manifold3d as m3
    return m3.Manifold(m3.Mesh(vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
                               tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint32)))


def manifold_transform(man, T):
    return man.transform(np.ascontiguousarray(np.asarray(T, float)[:3, :], dtype=np.float32))


def manifold_ok(man):
    return man is not None and str(man.status()) == "Error.NoError" and not man.is_empty()


def union_manifolds(mans):
    import manifold3d as m3
    mans = [m for m in mans if manifold_ok(m)]
    if not mans:
        return None
    if len(mans) == 1:
        return mans[0]
    return m3.Manifold.batch_boolean(mans, m3.OpType.Add)


def aabb_of(T, bounds):
    """World AABB (6,) of a box `bounds` (2, 3) moved by T (the eight corners)."""
    lo, hi = np.asarray(bounds, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = apply(T, c)
    return np.concatenate([w.min(axis=0), w.max(axis=0)])


def aabb_overlap(a, b, margin=0.0):
    return all(a[i] - margin <= b[i + 3] and b[i] - margin <= a[i + 3] for i in range(3))


RETESSELLATE = ((0.1, 0.5), (0.03, 0.25), (0.05, 0.2), (0.02, 0.2))   # tried in order when the mesher leaves a hole


def open_edges(mesh):
    import trimesh
    return len(trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1))


def closed_mesh(solid):
    """(mesh, notes) of a solid with no open edge: the default tessellation, else the first of RETESSELLATE that closes
    (OCC's mesher can leave a face's triangulation unstitched at one deflection and not at another)."""
    mesh, notes = solid_to_trimesh(solid)
    if mesh is None or open_edges(mesh) == 0:
        return mesh, notes
    for tol, ang in RETESSELLATE:
        m2, n2 = solid_to_trimesh(solid, tol, ang)
        if m2 is not None and open_edges(m2) == 0:
            return m2, n2 + ["re-tessellated at {} mm / {} rad (the default mesh had open edges)".format(tol, ang)]
    return mesh, notes


class Body(object):
    """One solid, or one bought piece (a solid or a compound of solids): the OCP shape, its tessellation, its manifold,
    volume, centroid, bounds and area (mm). `strict` (made parts): one BRepCheck-valid solid whose tessellation is closed
    (no open edge; see closed_mesh), consistently wound, of positive volume, and accepted by manifold3d (edges shared by
    more than two triangles, which the mesher leaves where faces meet along a seam, are noted, not refused: the boolean
    kernel takes them); given bought geometry (`strict=False`) needs only a manifold that manifold3d accepts per solid."""

    def __init__(self, name, shape, strict=True):
        import trimesh
        self.name = name
        self.shape = _wrapped(shape)
        self.errors, self.notes = [], []
        self.mesh = self.manifold = None
        self.volume, self.area = 0.0, 0.0
        self.centroid, self.bounds = np.zeros(3), np.zeros((2, 3))
        sols = solids_of(self.shape)
        if not sols:
            self.errors.append("no solid")
            return
        if strict and len(sols) != 1:
            self.errors.append("{} solids (one expected)".format(len(sols)))
            return
        if strict and not is_valid(sols[0]):
            self.errors.append("BRepCheck_Analyzer reports an invalid solid")
        meshes, mans = [], []
        for s in sols:
            mesh, notes = closed_mesh(s) if strict else solid_to_trimesh(s)
            self.notes += notes
            if mesh is None or len(mesh.faces) < 4:
                self.errors.append("tessellation produced no faces")
                return
            if strict:
                if open_edges(mesh):
                    self.errors.append("tessellated mesh is not closed ({} open edges)".format(open_edges(mesh)))
                elif not mesh.is_watertight:
                    self.notes.append("the tessellation has edges shared by more than two triangles")
                if not mesh.is_winding_consistent:
                    self.errors.append("tessellated mesh has inconsistent winding")
                if mesh.volume <= 0:
                    self.errors.append("tessellated mesh has non-positive volume")
            man = trimesh_to_manifold(mesh)
            if not manifold_ok(man):
                if strict:
                    self.errors.append("manifold construction failed: {}".format(man.status()))
                else:
                    self.notes.append("a solid's tessellation is not a manifold: left out of the booleans")
                    man = None
            meshes.append(mesh)
            mans.append(man)
        self.mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        if self.errors:
            return
        self.manifold = mans[0] if len(mans) == 1 else union_manifolds(mans)
        if self.manifold is None:
            self.errors.append("no manifold")
            return
        with np.errstate(divide="ignore", invalid="ignore"):
            vols = [abs(float(m.volume)) for m in meshes]
            cms = [np.asarray(m.center_mass, float) if v > 0 else np.zeros(3) for m, v in zip(meshes, vols)]
        self.volume = float(sum(vols))
        acc = [c * v for c, v in zip(cms, vols) if np.all(np.isfinite(c))]
        self.centroid = (np.sum(acc, axis=0) if acc else np.zeros(3)) / max(self.volume, 1e-12)
        self.bounds = np.asarray(self.mesh.bounds, float)
        self.area = float(sum(float(m.area) for m in meshes))

    @property
    def ok(self):
        return not self.errors and self.manifold is not None

    def contains(self, points):
        """Point-in-solid on the watertight tessellation (ray parity, batched)."""
        pts = np.asarray(points, float).reshape(-1, 3)
        if len(pts) == 0 or self.mesh is None:
            return np.zeros(len(pts), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):     # trimesh's barycentrics of degenerate sliver triangles
            return np.asarray(self.mesh.contains(pts), dtype=bool)


class BoughtClass(object):
    """A bought class (CONTRACTS 5): its JSON record (`meta`), its geometry in the class frame (the given STEP, or the
    .brep beside it) as a Body with strict=False, and its keep-out (the keep-out STEP the record names, else the
    geometry itself). Fields used by the grader: kind, mass_kg, com_mm, output_face_z, output_bolts, mount_faces, and
    for a gripper module hinges, open_stop_rad, tcp, fingertips."""

    def __init__(self, name, meta, geom_path, keepout_path=None):
        self.name = name
        self.meta = dict(meta)
        shape, err = load_shape(geom_path)
        if err:
            raise RuntimeError("bought class %s: %s" % (name, err))
        self.body = Body(name, shape, strict=False)
        if not self.body.ok:
            raise RuntimeError("bought class %s: unusable geometry (%s)" % (name, "; ".join(self.body.errors)))
        self.volume = self.body.volume
        self.com = np.asarray(meta["com_mm"], float) if meta.get("com_mm") is not None else self.body.centroid.copy()
        if meta.get("mass_kg") is None:
            raise RuntimeError("bought class %s: no mass_kg in its record" % name)
        self.mass = float(meta["mass_kg"])
        self.kind = meta.get("kind")
        self.output_face_z = meta.get("output_face_z")
        self.keepout = self.body
        if keepout_path and (os.path.isfile(keepout_path) or os.path.isfile(os.path.splitext(keepout_path)[0] + ".brep")):
            kshape, kerr = load_shape(keepout_path)
            if kerr:
                raise RuntimeError("bought class %s keep-out: %s" % (name, kerr))
            self.keepout = Body(name + "_keepout", kshape, strict=False)
            if not self.keepout.ok:
                raise RuntimeError("bought class %s: unusable keep-out (%s)" % (name, "; ".join(self.keepout.errors)))

    @property
    def is_motor(self):
        return self.kind == "motor" or (self.kind is None and self.output_face_z is not None)

    def output_centre(self, P):
        """The output-face centre (on the axis) of this motor placed at P (4x4), and its +z."""
        P = np.asarray(P, float)
        return apply(P, [0.0, 0.0, float(self.output_face_z or 0.0)]), unit(P[:3, 2])


def load_classes(bought_dir):
    """{class: BoughtClass} of a bought folder: CONTRACTS 5.1's <class>/<class>.json + <pn>.step (+ keep-out), or a flat
    folder of <class>.json + <class>.step / .brep."""
    out = {}
    if not bought_dir or not os.path.isdir(bought_dir):
        raise RuntimeError("no bought-class folder %s" % bought_dir)
    for entry in sorted(os.listdir(bought_dir)):
        p = os.path.join(bought_dir, entry)
        if os.path.isdir(p):
            js = os.path.join(p, entry + ".json")
            folder = p
        elif entry.endswith(".json"):
            js, folder = p, bought_dir
        else:
            continue
        if not os.path.isfile(js):
            continue
        with open(js) as fh:
            meta = json.load(fh)
        name = meta.get("class") or os.path.splitext(os.path.basename(js))[0]
        pn = meta.get("pn") or name
        geom = None
        for cand in (pn + ".step", pn + ".stp", pn + ".brep", name + ".step", name + ".brep"):
            q = os.path.join(folder, cand)
            if os.path.isfile(q):
                geom = q
                break
        if geom is None:
            raise RuntimeError("bought class %s: no geometry file (%s.step) in %s" % (name, pn, folder))
        ko = os.path.join(folder, meta["keepout"]) if meta.get("keepout") else None
        out[name] = BoughtClass(name, meta, geom, ko)
    return out


def outer_shell(mesh):
    """The outer shell of a solid's tessellation: enclosed voids (inner shells) dropped (cad_photo grade.outer_shell)."""
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:  # noqa: BLE001
        return mesh
    if len(comps) <= 1:
        return mesh
    with np.errstate(divide="ignore", invalid="ignore"):         # a degenerate sliver component has no volume
        return max(comps, key=lambda c: abs(float(c.volume)) if np.isfinite(c.volume) else 0.0)


# ----------------------------------------------------------------------------- scores
def band(value, b):
    """1 when |value| <= b[0], 0 when |value| >= b[1], linear between (b = [full, zero]); non-finite -> 0."""
    v = abs(float(value))
    if not math.isfinite(v):
        return 0.0
    full, zero = float(b[0]), float(b[1])
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return (zero - v) / (zero - full)


def score_ratio(ratio, thr):
    """1 at ratio <= thr, 0 at ratio >= 10 thr, log-linear between."""
    if ratio <= thr:
        return 1.0
    if ratio >= 10 * thr:
        return 0.0
    return max(0.0, 1.0 - math.log10(ratio / thr))


def ring_score(frac, lo=0.25, hi=0.75):
    return max(0.0, min(1.0, (float(frac) - lo) / (hi - lo)))


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def weighted(items):
    """sum(w s) / sum(w) of [(id, w, s)] (0 when empty)."""
    sw = sum(w for _i, w, _s in items)
    return sum(w * s for _i, w, s in items) / sw if sw > 0 else 0.0


def group_means(items):
    acc = {}
    for iid, w, s in items:
        a = acc.setdefault(iid.split(":")[0], [0.0, 0.0])
        a[0] += w * s
        a[1] += w
    return {g: (a[0] / a[1] if a[1] > 0 else 0.0) for g, a in acc.items()}


# ----------------------------------------------------------------------------- draws
def child_seed(ss, key):
    """A child SeedSequence of `ss` keyed by `key` (an int), independent of how many children were spawned before."""
    return np.random.SeedSequence(entropy=ss.entropy, spawn_key=tuple(ss.spawn_key) + (int(key),), pool_size=ss.pool_size)


def unit_cube_draws(n, d, ss):
    """n points of [0, 1)^d: a scrambled Sobol sequence seeded by the SeedSequence `ss` (n a power of two keeps it
    balanced)."""
    from scipy.stats import qmc
    if d == 0:
        return np.zeros((n, 0))
    eng = qmc.Sobol(d, scramble=True, seed=np.random.default_rng(ss))
    return eng.random(int(n))


# ----------------------------------------------------------------------------- per-part rigid alignment (align.py)
N_SRC = 20000
N_DST = 200000
N_DST_PER_MM2 = 3.0          # the destination density: a 100 000 mm2 plate gets 300 000 samples (align.py: 200 000 flat)
N_DST_MAX = 600000
TRIM = 0.90
ITERS = 40
MAX_T_MM = 15.0
MAX_R_DEG = 15.0
MULTISTART_RMS_MM = 3.0
QUERY_BOUND_MM = 40.0        # nearest-neighbour search radius (far points are clamped: a KD-tree query of far points is slow)
STEP_ROT = 0.15              # per-iteration step limits (rad, mm) of the unrestricted ICP
STEP_T_MM = 15.0


def n_dst_for(mesh, params=None):
    P = params or {}
    return int(min(P.get("n_dst_max", N_DST_MAX), max(P.get("n_dst", N_DST), P.get("n_dst_per_mm2", N_DST_PER_MM2) * float(mesh.area))))


def sample_surface(mesh, n, seed):
    import trimesh
    pts, fid = trimesh.sample.sample_surface(mesh, int(n), seed=int(seed) % (2 ** 32))
    return np.asarray(pts, float), np.asarray(mesh.face_normals[fid], float)


def surface_distances(query, samples, normals, tree, disk_mm=0.4):
    """Distance of each query point to the surface represented by dense samples with normals: the distance to the
    tangent disk (radius disk_mm) of the nearest sample (point-to-plane inside the disk, Euclidean beyond its rim), so
    points ON the sampled surface read ~0 whatever the sample spacing."""
    d, idx = tree.query(query)
    v = query - samples[idx]
    dn = np.abs(np.einsum("ij,ij->i", v, normals[idx]))
    dl = np.sqrt(np.maximum(0.0, d * d - dn * dn))
    return np.sqrt(dn * dn + np.maximum(0.0, dl - disk_mm) ** 2)


def _rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _query(tree, P):
    d, idx = tree.query(P, distance_upper_bound=QUERY_BOUND_MM)
    far = ~np.isfinite(d)
    d = np.where(far, QUERY_BOUND_MM, d)
    idx = np.where(far, 0, idx)
    return d, idx


def trimmed_rms(T, src, tree, trim=TRIM):
    d, _ = _query(tree, apply(T, src))
    k = max(1, int(len(d) * trim))
    return float(math.sqrt(np.mean(np.sort(d)[:k] ** 2)))


def icp(src, dst, dst_n, tree, T0, iters=ITERS, trim=TRIM, restrict=True, pivot=None, max_t_mm=MAX_T_MM, max_r_deg=MAX_R_DEG):
    """Point-to-plane ICP of `src` onto (dst, dst_n); returns (T, trimmed RMS). `restrict`: the total correction
    relative to T0 is clamped to max_t_mm / max_r_deg (rotation about `pivot`, default the source centroid)."""
    T = np.array(T0, float)
    piv = np.asarray(src.mean(axis=0) if pivot is None else pivot, float)
    for _ in range(int(iters)):
        P = apply(T, src)
        if not np.all(np.isfinite(P)):
            break
        d, idx = _query(tree, P)
        k = max(6, int(len(d) * trim))
        keep = np.argsort(d)[:k]
        p, q, n = P[keep], dst[idx[keep]], dst_n[idx[keep]]
        c = apply(T, piv)
        A = np.c_[np.cross(p - c, n), n]
        b = np.einsum("ij,ij->i", q - p, n)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(x)):
            break
        w = float(np.linalg.norm(x[:3]))
        if w > STEP_ROT:
            x[:3] *= STEP_ROT / w
        tn = float(np.linalg.norm(x[3:]))
        if tn > STEP_T_MM:
            x[3:] *= STEP_T_MM / tn
        R = _rodrigues(x[:3])
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = c - R @ c + x[3:]
        Tn = dT @ T
        if restrict:
            D = Tn @ np.linalg.inv(T0)
            r = rotation_angle_deg(D[:3, :3])
            t = float(np.linalg.norm(apply(D, piv) - piv))
            if r > max_r_deg or t > max_t_mm:
                break
        T = Tn
        if float(np.linalg.norm(x)) < 1e-7:
            break
    return T, trimmed_rms(T, src, tree, trim)


def _principal(P):
    c = P.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return c, V[:, np.argsort(w)[::-1]]


def pca_starts(src, dst):
    """The 24 proper rotations mapping the source principal axes onto the destination's (axis permutations x sign flips
    with det +1), each with the centroids matched."""
    cs, Vs = _principal(src)
    cd, Vd = _principal(dst)
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[:, i] = s * Vd[:, p]
            R = M @ Vs.T
            if np.linalg.det(R) < 0:
                continue
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = cd - R @ cs
            out.append(T)
    return out


def align_part(agent_mesh, gt_mesh, seed, gt_samples=None, params=None):
    """Align the submitted part onto the reference part (both in B at configuration Z): restricted point-to-plane ICP
    from the identity (N_SRC samples against the reference's area-scaled set, trimmed to the best 90 %, 40 iterations,
    total correction within 15 mm / 15 deg); if the trimmed RMS stays above 3 mm, a multistart from the 24 principal-
    axis rotations (each on a 1:10 subsample for 15 iterations, the two best refined in full); the lowest RMS wins.
    Proper rotations only: a mirrored part is not matched to its mirror image. `params`: rules.json part_alignment (the
    module constants by default). Returns {T (agent -> reference), rms_mm, multistart, offset_mm, rotation_deg, gt_pts,
    gt_normals, gt_tree, src_pts}."""
    from scipy.spatial import cKDTree
    P = params or {}
    trim, iters = float(P.get("trim", TRIM)), int(P.get("iters", ITERS))
    src, _src_n = sample_surface(agent_mesh, int(P.get("n_src", N_SRC)), seed * 2 + 1)
    if gt_samples is None:
        dst, dst_n = sample_surface(gt_mesh, n_dst_for(gt_mesh, P), seed * 2 + 2)
    else:
        dst, dst_n = gt_samples
    tree = cKDTree(dst)
    T, rms = icp(src, dst, dst_n, tree, np.eye(4), iters=iters, trim=trim, restrict=True,
                 max_t_mm=float(P.get("max_t_mm", MAX_T_MM)), max_r_deg=float(P.get("max_r_deg", MAX_R_DEG)))
    multistart = False
    if rms > float(P.get("multistart_rms_mm", MULTISTART_RMS_MM)):
        multistart = True
        sc, dc, dcn = src[::10], dst[::10], dst_n[::10]
        tc = cKDTree(dc)
        coarse = []
        for T0 in pca_starts(src, dst):
            Tk, rk = icp(sc, dc, dcn, tc, T0, iters=15, trim=trim, restrict=False)
            coarse.append((rk, Tk))
        coarse.sort(key=lambda x: x[0])
        best = (rms, T)
        for _rk, Tk in coarse[:2]:
            Tf, rf = icp(src, dst, dst_n, tree, Tk, iters=iters, trim=trim, restrict=False)
            if rf < best[0]:
                best = (rf, Tf)
        rms, T = best
    piv = src.mean(axis=0)
    return {"T": T, "rms_mm": float(rms), "multistart": multistart,
            "offset_mm": float(np.linalg.norm(apply(T, piv) - piv)), "rotation_deg": float(rotation_angle_deg(T[:3, :3])),
            "gt_pts": dst, "gt_normals": dst_n, "gt_tree": tree, "src_pts": src}


def refine_planar(src, src_n, dst, dst_n, tree, T0, iters=30, agree_deg=0.05, trim=TRIM):
    """Point-to-plane ICP from T0 on the correspondences whose two normals agree within agree_deg (flat regions), so the
    residuals carry no curvature bias: sample-based ICP elsewhere leaves a few thousandths of a degree where surfaces
    curve (the nearest sample's tangent plane misses the surface); on flat faces a submission that coincides with the
    reference aligns to the identity. Pairs whose residual exceeds max(0.05 mm, 4 x the median) are left out (two
    parallel faces a step apart agree in normal but are not the same face). Falls back to T0 when fewer than 200
    correspondences remain. (T, n used)."""
    T = np.array(T0, float)
    piv = src.mean(axis=0)
    cos_t = math.cos(math.radians(agree_deg))
    used = 0
    for _ in range(int(iters)):
        P = apply(T, src)
        Nn = src_n @ T[:3, :3].T
        d, idx = _query(tree, P)
        ok = np.einsum("ij,ij->i", Nn, dst_n[idx]) >= cos_t
        if int(ok.sum()) < 200:
            return np.array(T0, float), 0
        sel = np.nonzero(ok)[0]
        sel = sel[np.argsort(d[sel])[:max(6, int(len(sel) * trim))]]
        r = np.abs(np.einsum("ij,ij->i", dst[idx[sel]] - P[sel], dst_n[idx[sel]]))
        sel = sel[r <= max(0.05, 4.0 * float(np.median(r)))]      # parallel faces a step apart are not pairs
        if len(sel) < 200:
            return np.array(T0, float), 0
        used = len(sel)
        p, q, n = P[sel], dst[idx[sel]], dst_n[idx[sel]]
        c = apply(T, piv)
        A = np.c_[np.cross(p - c, n), n]
        b = np.einsum("ij,ij->i", q - p, n)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        if not np.all(np.isfinite(x)):
            break
        R = _rodrigues(x[:3])
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = c - R @ c + x[3:]
        T = dT @ T
        if float(np.linalg.norm(x)) < 1e-12:
            break
    return orthonormal(T), used


def register(src_pts, dst_pts, dst_normals, starts, iters=ITERS):
    """Best unrestricted point-to-plane ICP of src onto dst over the initial transforms `starts`: (T, trimmed RMS)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst_pts)
    sc = src_pts[::5]
    best = None
    coarse = []
    for T0 in starts:
        Tk, rk = icp(sc, dst_pts, dst_normals, tree, T0, iters=15, restrict=False)
        coarse.append((rk, Tk))
    coarse.sort(key=lambda x: x[0])
    for _rk, Tk in coarse[:3]:
        Tf, rf = icp(src_pts, dst_pts, dst_normals, tree, Tk, iters=iters, restrict=False)
        if best is None or rf < best[1]:
            best = (Tf, rf)
    return best


# ----------------------------------------------------------------------------- public-model reuse
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vertex_fingerprint(mesh, quant=0.01):
    """sha256 of the sorted, quantised vertex set (frame-dependent; catches re-exports of the same B-rep in the same frame)."""
    v = np.round(np.asarray(mesh.vertices, float) / quant).astype(np.int64)
    v = np.unique(v, axis=0)
    return hashlib.sha256(v.tobytes()).hexdigest()


def geometry_invariants(mesh):
    """Pose-invariant coarse signature: volume, area, sorted principal inertia (per unit mass), sorted AABB extents (frame
    dependent, informative only), and the pose- AND scale-free `shape` vector (area / volume^(2/3), principal inertia shares)."""
    vol = max(float(mesh.volume), 1e-9)
    I = np.asarray(mesh.moment_inertia, float) / vol
    eig = np.sort(np.linalg.eigvalsh(I))
    ext = sorted(float(v) for v in (mesh.bounds[1] - mesh.bounds[0]))
    tr = max(float(eig.sum()), 1e-12)
    shape = [float(mesh.area) / vol ** (2.0 / 3.0)] + [float(e) / tr for e in eig]
    return {"volume": float(mesh.volume), "area": float(mesh.area), "inertia": [float(e) for e in eig], "extents": ext, "shape": shape}


def invariants_match(a, b, rel=2e-3):
    def close(x, y):
        return abs(x - y) <= rel * max(abs(x), abs(y), 1e-9)
    return (close(a["volume"], b["volume"]) and close(a["area"], b["area"])
            and all(close(x, y) for x, y in zip(a["inertia"], b["inertia"])))


def invariants_match_similar(a, b, rel=2e-3):
    """Same geometry up to pose AND uniform scale (the scale-free shape vector agrees)."""
    sa, sb = a.get("shape"), b.get("shape")
    if not sa or not sb or len(sa) != len(sb):
        return False
    return all(abs(x - y) <= rel * max(abs(x), abs(y), 1e-9) for x, y in zip(sa, sb))
