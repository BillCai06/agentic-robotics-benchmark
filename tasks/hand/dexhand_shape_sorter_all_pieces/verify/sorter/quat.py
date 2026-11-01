"""Minimal quaternion helpers (xyzw). Interpolation and clamping use only +, *, / and sqrt (bit-reproducible)."""
import math

import numpy as np


def normalize(q):
    q = np.asarray(q, float)
    return q / math.sqrt(float(q @ q))


def conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], float)


def mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], float)


def apply(q, v):
    x, y, z, w = q
    v = np.asarray(v, float)
    t = 2.0 * np.array([y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]])
    return v + w * t + np.array([y * t[2] - z * t[1], z * t[0] - x * t[2], x * t[1] - y * t[0]])


def to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def to_rotvec(q):
    q = normalize(q)
    if q[3] < 0:
        q = -q
    s = math.sqrt(float(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]))
    if s < 1e-12:
        return 2.0 * q[:3]
    ang = 2.0 * math.atan2(s, float(q[3]))
    return q[:3] * (ang / s)


def angle_between(a, b):
    d = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(min(1.0, d))


def clamp_towards(cur, target, max_angle):
    """Return target if within max_angle of cur, else the nlerp point from cur towards target whose angle is
    about max_angle (bisection on the half-angle cosine; no trigonometry in the loop)."""
    return clamp_towards_cos(cur, target, math.cos(0.5 * max_angle))


def clamp_towards_cos(cur, target, cmin):
    """clamp_towards with the limit given as the cosine of the half angle (cmin = cos(max_angle / 2))."""
    cur = normalize(cur)
    target = normalize(target)
    if float(cur @ target) < 0.0:
        target = -target
    if float(cur @ target) >= cmin:
        return target
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        qm = normalize(cur + mid * (target - cur))
        if float(cur @ qm) >= cmin:
            lo = mid
        else:
            hi = mid
    return normalize(cur + lo * (target - cur))
