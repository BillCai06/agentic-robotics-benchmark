"""Plan format `shape_sorter_plan/v1` and its validator (the grader uses exactly this code).

A plan is a JSON object:
  {"format": "shape_sorter_plan/v1",
   "waypoints": [[x, y, z, qx, qy, qz, qw, f0, ..., f19], ...]}
One waypoint per control period (25 Hz, 0.04 s), at most MAX_WAYPOINTS (180 s). Each waypoint is the wrist (root
link `link_00_mount`) target pose in the world frame (position in metres, unit quaternion xyzw) and the 20 finger
joint targets (radians, SuperDex DOF order 6..25 of the Wuji Hand 2). Limits (checked against the previous waypoint; the
first waypoint is checked against the park pose and the default finger pose):
  |dp| <= MAX_LIN_SPEED*0.04, rotation angle <= MAX_ANG_SPEED*0.04, |df_i| <= MAX_FINGER_SPEED*0.04,
  position inside WRIST_BOUNDS, fingers inside the joint limits, quaternion norm within 1e-3 of 1, all finite, and
  the wrist orientation limit: palm angle (palm_angle_deg) <= C.WRIST_MAX_PALM_DEG, and the finger direction limit:
  finger pitch (finger_pitch_deg) <= C.FINGER_MAX_PITCH_DEG.
Any violation makes the whole plan invalid (that instance scores 0).
"""
import json
import math

import numpy as np

from . import config as C
from . import quat as Q

FORMAT = "shape_sorter_plan/v1"
MAX_BYTES = 8 << 20
EPS = 1e-9

# Wuji Hand 2 finger joint limits and default pose (SuperDex 1.0.0 public asset, DOFs 6..25), copied here so that
# validation needs no simulator. DOF order: thumb (cmc_flex, cmc_abd, mcp, ip), then index, middle, ring, pinky
# (mcp_flex, mcp_abd, pip, dip each).
FINGER_LO = [-1.187000036239624, -1.4839999675750732, 0.0, 0.0, 0.0, -0.6980000138282776, 0.0, 0.0, 0.0,
             -0.6980000138282776, 0.0, 0.0, 0.0, -0.6980000138282776, 0.0, 0.0, 0.0, -0.6980000138282776, 0.0, 0.0]
# (flexion joints: lower limit 0 = no hyperextension, see config.NO_HYPEREXTENSION_SUFFIXES)
FINGER_HI = [1.2910000085830688, 0.6980000138282776, 1.5700000524520874, 1.5700000524520874, 1.5700000524520874,
             0.6980000138282776, 2.0940001010894775, 1.5700000524520874, 1.5700000524520874, 0.6980000138282776,
             2.0940001010894775, 1.5700000524520874, 1.5700000524520874, 0.6980000138282776, 2.0940001010894775,
             1.5700000524520874, 1.5700000524520874, 0.6980000138282776, 2.0940001010894775, 1.5700000524520874]
FINGER_DEFAULT = [0.1926, -0.37, 0.2088, 0.2429, 0.2542, 0.0756, 0.1179, 0.1173, 0.2511, 0.1542, 0.1594, 0.16, 0.1083,
                  0.1505, 0.2391, 0.1346, 0.1007, 0.2693, 0.2425, 0.1008]


class PlanError(ValueError):
    pass


# Palm normal (unit vector out of the palm) in the wrist frame: straight down (0, 0, -1) at the park pose.
PALM_NORMAL_LOCAL = Q.apply(Q.conj(Q.normalize(C.PARK_QUAT)), [0.0, 0.0, -1.0])


def palm_cos(quat):
    """cos of the palm angle of wrist orientation `quat` (xyzw, unit): 1 = palm down, 0 = sideways, -1 = palm up."""
    return -float(Q.apply(quat, PALM_NORMAL_LOCAL)[2])


def palm_angle_deg(quat):
    """Angle in degrees between the palm normal of wrist orientation `quat` (xyzw) and world down (0, 0, -1)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, palm_cos(Q.normalize(quat))))))


# Finger axis (wrist -> fingertips) in the wrist frame: horizontal towards +x at the park pose.
FINGER_AXIS_LOCAL = Q.apply(Q.conj(Q.normalize(C.PARK_QUAT)), [1.0, 0.0, 0.0])


def finger_sin_pitch(quat):
    """sin of the finger pitch of wrist orientation `quat` (xyzw, unit): +1 = fingers straight up, 0 = horizontal."""
    return float(Q.apply(quat, FINGER_AXIS_LOCAL)[2])


def finger_pitch_deg(quat):
    """Angle in degrees of the finger axis (wrist -> fingertips) above the horizontal for wrist orientation `quat`
    (xyzw): + = fingers pointing up, - = down."""
    return math.degrees(math.asin(max(-1.0, min(1.0, finger_sin_pitch(Q.normalize(quat))))))


def finger_limit_project(quat):
    """The simulator's finger direction limit: a unit quaternion whose finger pitch exceeds C.FINGER_MAX_PITCH_DEG is
    rotated (about a horizontal axis, the smallest rotation) so that the finger axis lies on the limit cone; otherwise
    it is returned unchanged. Uses only +, *, / and sqrt (bit-reproducible)."""
    f = Q.apply(quat, FINGER_AXIS_LOCAL)
    if float(f[2]) <= C.SIN_MAX_PITCH:
        return quat
    quat = Q.normalize(quat)
    f = Q.apply(quat, FINGER_AXIS_LOCAL)
    hx, hy = float(f[0]), float(f[1])
    h = math.sqrt(hx * hx + hy * hy)
    if h < 1e-12:
        hx, hy, h = 1.0, 0.0, 1.0
    b = np.array([C.COS_MAX_PITCH * hx / h, C.COS_MAX_PITCH * hy / h, C.SIN_MAX_PITCH])   # on the cone, same azimuth
    ax = np.array([f[1] * b[2] - f[2] * b[1], f[2] * b[0] - f[0] * b[2], f[0] * b[1] - f[1] * b[0]])
    r = Q.normalize(np.array([ax[0], ax[1], ax[2], 1.0 + float(f @ b)]))
    return Q.normalize(Q.mul(r, quat))


def orientation_limit_project(quat):
    """Both orientation limits of the simulator. First the palm limit exactly as before r6 (wrist_limit_project); if the
    finger pitch is then within its limit the result is returned (so inside both limits `quat` comes back unchanged).
    Otherwise the finger projection and the palm projection alternate (each can move the other angle; the error
    shrinks about 25-fold per round) up to 24 times, ending with the palm projection, so the palm limit always holds;
    over 200 000 random orientations at most 15 rounds were needed and the finger pitch ended within 1e-6 deg of its
    limit. Set-points of valid plans need at most a round or two (they start within a degree of the limits)."""
    quat = wrist_limit_project(quat)
    for _ in range(24):
        if finger_sin_pitch(quat) <= C.SIN_MAX_PITCH + 1e-12:
            return quat
        quat = wrist_limit_project(finger_limit_project(quat))
    return quat


def wrist_limit_project(quat):
    """The simulator's wrist limit: a unit quaternion whose palm angle exceeds C.WRIST_MAX_PALM_DEG is rotated (about a
    horizontal axis, the smallest rotation) onto the limit cone; inside the cone it is returned unchanged. Uses only
    +, *, / and sqrt (bit-reproducible)."""
    n = Q.apply(quat, PALM_NORMAL_LOCAL)
    if -float(n[2]) >= C.COS_MAX_PALM:
        return quat
    quat = Q.normalize(quat)
    n = Q.apply(quat, PALM_NORMAL_LOCAL)
    hx, hy = float(n[0]), float(n[1])
    h = math.sqrt(hx * hx + hy * hy)
    if h < 1e-12:
        hx, hy, h = 1.0, 0.0, 1.0
    b = np.array([C.SIN_MAX_PALM * hx / h, C.SIN_MAX_PALM * hy / h, -C.COS_MAX_PALM])   # on the cone, above n's foot
    ax = np.array([n[1] * b[2] - n[2] * b[1], n[2] * b[0] - n[0] * b[2], n[0] * b[1] - n[1] * b[0]])
    r = Q.normalize(np.array([ax[0], ax[1], ax[2], 1.0 + float(n @ b)]))
    return Q.normalize(Q.mul(r, quat))


def parse_bytes(data):
    if len(data) > MAX_BYTES:
        raise PlanError("plan file larger than %d bytes" % MAX_BYTES)

    def no_const(c):
        raise PlanError("non-finite constant %s" % c)
    try:
        obj = json.loads(data.decode("utf-8"), parse_constant=no_const)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PlanError("not valid JSON: %s" % str(exc)[:80])
    return validate(obj)


def validate(obj, finger_default=None):
    """Return the list of waypoints as (pos[3], quat[4], fingers[20]) numpy tuples, or raise PlanError."""
    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise PlanError("format must be %r" % FORMAT)
    wps = obj.get("waypoints")
    if not isinstance(wps, list) or not (1 <= len(wps) <= C.MAX_WAYPOINTS):
        raise PlanError("waypoints must be a list of 1..%d rows" % C.MAX_WAYPOINTS)
    lo, hi = np.array(FINGER_LO), np.array(FINGER_HI)
    dt = 1.0 / C.CONTROL_HZ
    max_dp, max_da, max_df = C.MAX_LIN_SPEED * dt + EPS, C.MAX_ANG_SPEED * dt + EPS, C.MAX_FINGER_SPEED * dt + EPS
    prev_p = np.array(C.PARK_POS, float)
    prev_q = np.array(C.PARK_QUAT, float)
    prev_f = np.array(FINGER_DEFAULT if finger_default is None else finger_default, float)
    out = []
    for i, row in enumerate(wps):
        if not isinstance(row, list) or len(row) != 27:
            raise PlanError("waypoint %d: expected 27 numbers" % i)
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in row):
            raise PlanError("waypoint %d: non-numeric entry" % i)
        v = np.array(row, float)
        if not np.all(np.isfinite(v)):
            raise PlanError("waypoint %d: non-finite entry" % i)
        p, q, f = v[0:3], v[3:7], v[7:27]
        nq = math.sqrt(float(q @ q))
        if abs(nq - 1.0) > 1e-3:
            raise PlanError("waypoint %d: quaternion norm %.6f" % (i, nq))
        # q is returned exactly as written; World.step_waypoint normalises it once (bit-identical replays)
        for k, (a, b) in enumerate(C.WRIST_BOUNDS):
            if not (a - EPS <= p[k] <= b + EPS):
                raise PlanError("waypoint %d: wrist position outside the workspace" % i)
        c = palm_cos(Q.normalize(q))
        if c < C.COS_MAX_PALM - EPS:
            raise PlanError("waypoint %d: palm angle %.2f deg exceeds the wrist limit of %.0f deg (palm normal vs straight "
                            "down, see plan.palm_angle_deg)" % (i, math.degrees(math.acos(max(-1.0, min(1.0, c)))),
                                                               C.WRIST_MAX_PALM_DEG))
        sp = finger_sin_pitch(Q.normalize(q))
        if sp > C.SIN_MAX_PITCH + EPS:
            raise PlanError("waypoint %d: finger pitch %.2f deg exceeds the finger direction limit of %.0f deg (finger axis "
                            "above the horizontal, see plan.finger_pitch_deg)" % (i, math.degrees(math.asin(min(1.0, sp))),
                                                                                  C.FINGER_MAX_PITCH_DEG))
        if np.any(f < lo - EPS) or np.any(f > hi + EPS):
            k = int(np.argmax(np.maximum(lo - f, f - hi)))
            raise PlanError("waypoint %d: finger %d target %.4f outside its range [%.4f, %.4f]" % (i, k, f[k], lo[k], hi[k]))
        if float(np.linalg.norm(p - prev_p)) > max_dp:
            raise PlanError("waypoint %d: wrist moves faster than %.2f m/s" % (i, C.MAX_LIN_SPEED))
        if Q.angle_between(Q.normalize(prev_q), Q.normalize(q)) > max_da:
            raise PlanError("waypoint %d: wrist rotates faster than %.2f rad/s" % (i, C.MAX_ANG_SPEED))
        if float(np.max(np.abs(f - prev_f))) > max_df:
            raise PlanError("waypoint %d: a finger moves faster than %.2f rad/s" % (i, C.MAX_FINGER_SPEED))
        out.append((p, q, f))
        prev_p, prev_q, prev_f = p, q, f
    return out


def to_json(waypoints):
    """Serialize (pos, quat, fingers) tuples to a plan object (floats written with repr precision)."""
    rows = [[float(x) for x in np.concatenate([np.asarray(p, float), np.asarray(q, float), np.asarray(f, float)])]
            for p, q, f in waypoints]
    return {"format": FORMAT, "waypoints": rows}
