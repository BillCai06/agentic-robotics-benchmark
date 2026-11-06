"""Classical, deterministic perception for the localisation check (round 4, spec section 5.3): NumPy and OpenCV from
/opt/venv only, nothing trained. Every function takes delivered frames, the manifest's sensor entries and the robots'
own numbers (never the true state); `check_localisation.py` scores what they return against the true state.

- geometry: a sensor's world pose from its body's pose and its mount (`sensor_pose`), pixel rays (`pixel_ray`), a ray
  on a plane (`ray_plane`), rays triangulated (`triangulate`);
- the vehicle's start pose by landmark fit on the twin map (`fit_start_pose`: x, y, heading, the camera height known);
- colour blobs (`blob`: HSV thresholds, the largest connected component), line fits (`line_angle`), thermal blobs;
- lidar points in the world with the robot's own returns masked (`lidar_points`), clusters near an expected place.
"""

import math

import numpy as np

try:
    import cv2
except ImportError:  # (the host's checks import the geometry without OpenCV)
    cv2 = None


# --------------------------------------------------------------------------------------------- geometry
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotm(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_z(a):
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def sensor_pose(body_p, body_q, entry):
    """A sensor's world position and rotation (camera axes as columns: +X right, +Y up, -Z forward) from its body's
    pose and the manifest's mount."""
    m = entry["pose_in_body"]
    R = rotm(body_q)
    return np.asarray(body_p, float) + R @ np.asarray(m["pos"], float), rotm(qmul(body_q, np.asarray(m["quat"], float)))


def pixel_ray(entry, u, v, R):
    """The world direction of pixel (u, v) (the manifest's pinhole: u right, v down)."""
    K = entry["intrinsics"]
    d_cam = np.array([(u - K["cx"]) / K["fx"], -(v - K["cy"]) / K["fy"], -1.0])
    d = R @ d_cam
    return d / np.linalg.norm(d)


def project(entry, R, C, X):
    """World points X (N x 3) -> pixels (N x 2) and their depth along the optical axis."""
    K = entry["intrinsics"]
    pc = (np.asarray(X, float) - C) @ R
    z = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K["cx"] + K["fx"] * pc[:, 0] / z
        v = K["cy"] - K["fy"] * pc[:, 1] / z
    return np.stack([u, v], axis=1), z


def ray_plane(C, d, z0):
    """Where the ray C + s d meets the horizontal plane z = z0 (None if it does not, forward)."""
    if abs(d[2]) < 1e-9:
        return None
    s = (z0 - C[2]) / d[2]
    return None if s <= 0 else C + s * d


def triangulate(origins, dirs):
    """The least-squares point nearest a set of rays."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.lstsq(A, b, rcond=None)[0]


# --------------------------------------------------------------------------------------------- images
def blob(rgb, hsv_lo, hsv_hi, min_px=30, mask=None):
    """The largest connected blob of pixels within the HSV box (OpenCV's H 0-179): (centroid u, v, area, its mask)."""
    if cv2 is None:
        raise RuntimeError("OpenCV is needed")
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    if mask is not None:
        m = m & mask
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_px:
        return None
    return float(cents[k][0]), float(cents[k][1]), int(stats[k, cv2.CC_STAT_AREA]), lab == k


def line_angle(mask):
    """The principal direction of a blob's pixels (image angle, radians, of its long axis) and its elongation."""
    v, u = np.nonzero(mask)
    if u.size < 5:
        return None, 0.0
    pts = np.stack([u - u.mean(), v - v.mean()], axis=1).astype(float)
    w, V = np.linalg.eigh(pts.T @ pts / len(pts))
    a = V[:, 1]
    return math.atan2(a[1], a[0]), float(math.sqrt(w[1] / max(w[0], 1e-9)))


def hot(thermal, k_min):
    """Pixels hotter than k_min: (centroid u, v, count, max K)."""
    m = thermal > k_min
    if not m.any():
        return None
    v, u = np.nonzero(m)
    return float(u.mean()), float(v.mean()), int(m.sum()), float(thermal.max())


# --------------------------------------------------------------------------------------------- the start pose
def fit_start_pose(obs_list, entry, mount_body_to_root, root_z, guess, max_iter=60):
    """The vehicle's (x, y, heading) from landmark sightings: obs_list = [(world point, (u, v), weight)], the camera's
    entry, the body's pose in the vehicle's root frame (identity when the camera rides on the root), the root's height
    and a first guess (x, y, heading in rad: the odometry's nominal pose). Gauss-Newton on the reprojection error."""
    x = np.array(guess, dtype=float)
    Pb, Qb = mount_body_to_root

    def residual(x):
        p = np.array([x[0], x[1], root_z])
        q = quat_z(x[2])
        bp = p + rotm(q) @ np.asarray(Pb, float)
        bq = qmul(q, np.asarray(Qb, float))
        C, R = sensor_pose(bp, bq, entry)
        X = np.array([o[0] for o in obs_list])
        uv, z = project(entry, R, C, X)
        r = (uv - np.array([o[1] for o in obs_list])) * np.array([o[2] for o in obs_list])[:, None]
        r[z <= 0] = 1e3
        return r.reshape(-1)
    for _ in range(max_iter):
        r0 = residual(x)
        J = np.zeros((r0.size, 3))
        for k, h in enumerate((1e-4, 1e-4, 1e-5)):
            dx = np.zeros(3)
            dx[k] = h
            J[:, k] = (residual(x + dx) - r0) / h
        step = np.linalg.lstsq(J.T @ J + 1e-6 * np.eye(3), -J.T @ r0, rcond=None)[0]
        x = x + step
        if np.linalg.norm(step) < 1e-7:
            break
    return x, float(np.sqrt(np.mean(residual(x) ** 2)))


# --------------------------------------------------------------------------------------------- lidar
def lidar_points(ranges, entry, C, R):
    """A range image's returns as world points (the manifest's beams)."""
    b = entry["beams"]
    rows, cols = b["rows"], b["columns"]
    e = np.radians(np.linspace(b["elevation_first_deg"], b["elevation_last_deg"], rows)) if rows > 1 else np.radians([b["elevation_first_deg"]])
    a = np.radians(np.arange(cols) * b["azimuth_step_deg"])
    ee, aa = np.meshgrid(e, a, indexing="ij")
    d = np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], axis=-1)
    ok = ranges > 0
    pts = (d[ok] * ranges[ok][:, None]) @ R.T + C
    return pts


def cluster_near(points, expect, radius=0.6, own=None, own_radius=0.55):
    """The centroid of the returns within `radius` of an expected place, the robot's own returns (within `own_radius`
    of its own position) left out; None when there are fewer than 5."""
    p = points
    if own is not None:
        p = p[np.linalg.norm(p - own, axis=1) > own_radius]
    sel = p[np.linalg.norm(p - expect, axis=1) < radius]
    return (sel.mean(axis=0), len(sel)) if len(sel) >= 5 else (None, len(sel))


# --------------------------------------------------------------------------------------------- the twin's edges
def twin_shapes(scene, seed, sigma=0.02):
    """The twin map: the reference scene's fixed furniture (`/World/Facility`: the tables, the crate, the steam line's
    pipe, supports, valve body and gauge, the tank, the pads' and the mat's floor markings) as boxes and cylinders in
    world axes, every landmark group moved by N(0, sigma) per axis (seeded). Returns (shapes, offsets): shape =
    dict(group, kind "box" | "cyl", M (4x4 world, column vectors), half (box half extents) | r, h, axis)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(scene)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    skip = ("/Items/", "/Bucket", "/FirePan", "/SteamLine/Handle", "/Gauge/needle", "/SteamLine/Leak", "/SteamLine/Plume", "/Floor")
    groups = {}
    out = []
    for p in st.Traverse():
        path = p.GetPath().pathString
        if not path.startswith("/World/Facility/") or any(s in path for s in skip):
            continue
        if not (p.IsA(UsdGeom.Cube) or p.IsA(UsdGeom.Cylinder)):
            continue
        if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        rel = path[len("/World/Facility/"):].split("/")
        if rel[0] in ("Tables", "Pads"):
            g = rel[1]
        elif rel[0] == "SteamLine" and rel[1] == "Pipe":
            g = "pipe_" + rel[2].split("_")[-1] if rel[2].startswith(("support_", "foot_")) else "pipe"
        elif rel[0] == "SteamLine":
            g = rel[1]
        else:
            g = rel[0]
        M = np.array(xc.GetLocalToWorldTransform(p), dtype=float).T  # (Gf is row-vector: transpose to column vectors)
        if p.IsA(UsdGeom.Cube):
            s = float(UsdGeom.Cube(p).GetSizeAttr().Get())
            out.append(dict(group=g, kind="box", M=M, half=np.array([s / 2] * 3), path=path))
        else:
            c = UsdGeom.Cylinder(p)
            out.append(dict(group=g, kind="cyl", M=M, r=float(c.GetRadiusAttr().Get()), h=float(c.GetHeightAttr().Get()),
                            axis=str(c.GetAxisAttr().Get()), path=path))
        groups.setdefault(g, None)
    rng = np.random.default_rng(seed)
    offsets = {g: rng.normal(0.0, sigma, 3) for g in sorted(groups)}
    for s in out:
        s["M"] = s["M"].copy()
        s["M"][:3, 3] += offsets[s["group"]]
    return out, {g: [round(float(v), 4) for v in o] for g, o in offsets.items()}


def shape_edges(shapes, eye=None, step=0.01, groups=None):
    """Points along the shapes' edges (world, N x 3) for edge matching: a box's 12 edges; a cylinder's two rims and,
    seen from `eye`, its two silhouette lines."""
    pts = []
    for s in shapes:
        if groups is not None and s["group"] not in groups:
            continue
        M = s["M"]
        if s["kind"] == "box":
            h = s["half"]
            corners = np.array([[sx * h[0], sy * h[1], sz * h[2], 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            W = (M @ corners.T).T[:, :3]
            for i in range(8):
                for j in range(i + 1, 8):
                    if bin(i ^ j).count("1") != 1:
                        continue
                    L = np.linalg.norm(W[j] - W[i])
                    n = max(2, int(L / step) + 1)
                    pts.append(W[i] + np.linspace(0, 1, n)[:, None] * (W[j] - W[i]))
        else:
            ax = {"X": 0, "Y": 1, "Z": 2}[s["axis"]]
            a = M[:3, ax] / np.linalg.norm(M[:3, ax])
            c = M[:3, 3]
            scale = np.linalg.norm(M[:3, ax])
            hh = s["h"] / 2 * scale
            r = s["r"] * np.linalg.norm(M[:3, (ax + 1) % 3])
            u = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
            u /= np.linalg.norm(u)
            w = np.cross(a, u)
            k = max(8, int(2 * math.pi * r / step))
            ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
            ring = (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w) * r
            pts += [c + a * hh + ring, c - a * hh + ring]
            if eye is not None:
                v = c - np.asarray(eye, float)
                v -= a * (v @ a)
                sdir = np.cross(a, v)
                if np.linalg.norm(sdir) > 1e-9:
                    sdir /= np.linalg.norm(sdir)
                    n = max(2, int(2 * hh / step) + 1)
                    line = np.linspace(-hh, hh, n)[:, None] * a
                    pts += [c + r * sdir + line, c - r * sdir + line]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def edge_map(rgb, lo=40, hi=120, blur=1.2):
    """A frame's distance transform to its Canny edges (px)."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    e = cv2.Canny(g, lo, hi)
    return cv2.distanceTransform(255 - e, cv2.DIST_L2, 3).astype(np.float32), e


def chamfer(dt, entry, C, R, X, tau=8.0, min_pts=60):
    """The mean truncated distance (px) from the projections of model points X to the frame's edges, and the number
    of points in the image (in front, within the frame)."""
    uv, z = project(entry, R, C, X)
    H, W = dt.shape
    ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    if ok.sum() < min_pts:
        return tau, int(ok.sum())
    d = dt[np.round(uv[ok, 1]).astype(int), np.round(uv[ok, 0]).astype(int)]
    return float(np.mean(np.minimum(d, tau))), int(ok.sum())


def fit_pose(views, pose_fn, x0, ranges, steps, refine_steps=None, tau=8.0):
    """A pose by edge matching: `views` = [(dt, entry, X model points, mount)] (one per camera frame), `pose_fn(x,
    mount)` -> (C, R) of that camera; a grid search over x0 +- ranges at `steps`, then finer grids around the best
    (`refine_steps`: a list of step vectors, each searched over +- 3 steps). Returns (x, score)."""
    import itertools

    def score(x):
        tot, n = 0.0, 0
        for dt, entry, X, mount in views:
            C, R = pose_fn(x, mount)
            s, k = chamfer(dt, entry, C, R, X, tau)
            tot += s
            n += 1
        return tot / max(1, n)
    x0 = np.asarray(x0, float)
    axes = [np.arange(-r, r + 1e-9, st) for r, st in zip(ranges, steps)]
    best = (None, 1e9)
    for d in itertools.product(*axes):
        x = x0 + np.array(d)
        s = score(x)
        if s < best[1]:
            best = (x, s)
    for st in refine_steps or []:
        st = np.asarray(st, float)
        c = best[0]
        for d in itertools.product(*[np.arange(-3, 4) * v for v in st]):
            x = c + np.array(d)
            s = score(x)
            if s < best[1]:
                best = (x, s)
    return best


def vehicle_body(x, y, heading_deg, z=0.0635):
    """The vehicle's root body pose (world position, quaternion w, x, y, z) from (x, y, heading): heading 0 faces +y,
    counter-clockwise positive; the body's +X is forward (a turn of heading + 90 degrees about z)."""
    return np.array([x, y, z]), quat_z(math.radians(heading_deg + 90.0))


# --------------------------------------------------------------------------------------------- the vehicle's arm (FK)
class Chain:
    """Forward kinematics of the vehicle's articulation from the program's own scene (pxr): the joints' frames on
    their two bodies (localPos/localRot), revolute about the joint frame's axis by the encoder's position. `pose(body,
    joints)` gives a body's pose in the root body's frame (position, quaternion w, x, y, z)."""

    def __init__(self, scene, root="/World/Fleet/UGV"):
        from pxr import Usd, UsdPhysics
        st = Usd.Stage.Open(scene)
        self.parent = {}  # child body -> (parent body, joint name, kind, axis, (p0, q0), (p1, q1))
        for p in Usd.PrimRange(st.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if not b0 or not b1:
                continue
            kind = "revolute" if p.IsA(UsdPhysics.RevoluteJoint) else "fixed" if p.IsA(UsdPhysics.FixedJoint) else "other"
            ax = p.GetAttribute("physics:axis").Get() if p.GetAttribute("physics:axis") else None

            def pq(pos, rot):
                return np.array([float(v) for v in pos]), np.array([float(rot.GetReal())] + [float(v) for v in rot.GetImaginary()])
            self.parent[b1[0]] = (b0[0], p.GetName(), kind, str(ax) if ax else None, pq(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                                  pq(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()))

    def pose(self, body, joints):
        """(p, q) of `body` in the root body's frame; `joints`: {name: [position, velocity]} (the wire's ugv.joints)."""
        chain = []
        b = body
        while b in self.parent:
            chain.append(self.parent[b])
            b = self.parent[b][0]
        p, q = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        for par, name, kind, ax, (p0, q0), (p1, q1) in reversed(chain):
            # body1 = body0 * T(p0, q0) * R(axis, angle) * T(p1, q1)^-1
            p = p + rotm(q) @ p0
            q = qmul(q, q0)
            if kind == "revolute":
                ang = float((joints.get(name) or [0.0])[0])
                a = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[ax]
                q = qmul(q, np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * a]))
            qi = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
            q = qmul(q, qi)
            p = p - rotm(q) @ p1
        return p, q
