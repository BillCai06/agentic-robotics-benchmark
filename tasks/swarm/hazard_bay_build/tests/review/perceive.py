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
